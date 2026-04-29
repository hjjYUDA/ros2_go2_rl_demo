"""Standalone ONNX exporter — no IsaacSim required.

Reads a rsl-rl checkpoint (model_xxxx.pt), reconstructs the actor MLP from
the agent.yaml (sibling file in `params/`), and exports `policy.onnx` next
to the checkpoint, plus `policy.pt` (Torch JIT) for sanity.

Usage:
    python scripts/rsl_rl/export_onnx_standalone.py \\
        --checkpoint logs/rsl_rl/unitree_go2_velocity_robust/<run>/model_2200.pt
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.nn as nn
import yaml


ACT_MAP = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


def build_mlp(in_dim: int, hidden: List[int], out_dim: int, activation: str) -> nn.Sequential:
    act_cls = ACT_MAP[activation.lower()]
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(act_cls())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--obs-dim", type=int, default=None,
                   help="defaults to the in_features of actor.0.weight")
    p.add_argument("--act-dim", type=int, default=None,
                   help="defaults to the out_features of the last actor layer")
    p.add_argument("--filename", default="policy.onnx")
    args = p.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    log_dir = os.path.dirname(ckpt_path)
    agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")
    if not os.path.exists(agent_yaml_path):
        agent_yaml_path = os.path.join(os.path.dirname(log_dir), "params", "agent.yaml")
    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] agent.yaml: {agent_yaml_path}")

    with open(agent_yaml_path, "r") as f:
        agent_cfg = yaml.safe_load(f)

    hidden_dims = list(agent_cfg["policy"]["actor_hidden_dims"])
    activation = agent_cfg["policy"]["activation"]
    if agent_cfg.get("empirical_normalization", False):
        raise NotImplementedError(
            "Empirical normalization is enabled in agent.yaml; this standalone exporter does not "
            "ship its state. Run scripts/rsl_rl/export_onnx.py (with IsaacSim) instead."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    actor_sd = {k.removeprefix("actor."): v for k, v in sd.items() if k.startswith("actor.")}

    in_dim = args.obs_dim or actor_sd["0.weight"].shape[1]
    last_lin = max(int(k.split(".")[0]) for k in actor_sd.keys() if k.endswith(".weight"))
    out_dim = args.act_dim or actor_sd[f"{last_lin}.weight"].shape[0]
    print(f"[INFO] actor MLP: in={in_dim}  hidden={hidden_dims}  out={out_dim}  act={activation}")

    actor = build_mlp(in_dim, hidden_dims, out_dim, activation)
    missing, unexpected = actor.load_state_dict(actor_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state-dict mismatch missing={missing} unexpected={unexpected}")
    actor.eval()

    out_dir = os.path.join(log_dir, "exported")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, args.filename)
    pt_path = os.path.join(out_dir, "policy.pt")

    dummy = torch.zeros(1, in_dim)
    torch.onnx.export(
        actor, dummy, onnx_path,
        export_params=True, opset_version=11,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    traced = torch.jit.trace(actor, dummy)
    traced.save(pt_path)
    print(f"[OK] wrote {onnx_path}")
    print(f"[OK] wrote {pt_path}")


if __name__ == "__main__":
    main()
