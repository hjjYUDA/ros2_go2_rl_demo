// rl_inference_node.cpp (ROS 2) — Go2 RL policy bridge.
//
// Loads a trained policy (ONNX) and the unitree_rl_lab deploy.yaml metadata
// produced by export_deploy_cfg.py, runs the policy at 1/step_dt (50 Hz by
// default), converts joint position targets to torques via PD, and publishes
// /go2/cmd_torque.
//
//   joint_states + imu  ---> obs vector (deploy.yaml term order)
//                              ---> ONNX ---> q_des(asset order)
//   /go2/cmd_vel  ---> velocity_commands (3 floats appended to obs)
//   tau = kp*(q_des - q) - kd*dq  (SDK order, 12)
//   /go2/mode  ---> stand / walk / passive
//
// Joint orderings:
//   * SDK order (matches /go2/joint_states / /go2/cmd_torque in this repo):
//       FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
//       RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
//   * Asset order (used inside the ONNX policy) is determined by IsaacLab's
//     loaded USD; deploy.yaml records the mapping as
//         joint_ids_map[asset_idx] = sdk_idx
//     so for any per-joint vector:
//         v_asset[asset_idx] = v_sdk[joint_ids_map[asset_idx]]
//         v_sdk[joint_ids_map[asset_idx]] = v_asset[asset_idx]
//   * stiffness, damping  -> SDK order (length 12)
//   * default_joint_pos, action_scale, action_offset, action_clip -> asset order

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {

constexpr int N_JOINTS = 12;

enum class Mode { STAND, WALK, PASSIVE };

struct DeployCfg {
    std::vector<int> joint_ids_map;          // sdk_idx -> asset_idx (size 12)
    std::vector<double> stiffness;           // SDK order
    std::vector<double> damping;             // SDK order
    std::vector<double> default_joint_pos;   // asset order
    std::vector<double> action_offset;       // asset order
    std::vector<double> action_scale;        // asset order
    std::vector<std::pair<double,double>> action_clip;
    double step_dt = 0.02;

    double lin_vel_x_min = -1.0, lin_vel_x_max = 1.0;
    double lin_vel_y_min = -0.4, lin_vel_y_max = 0.4;
    double ang_vel_z_min = -1.0, ang_vel_z_max = 1.0;

    struct Obs {
        std::string name;
        int dim;
        std::vector<double> scale;
        std::pair<double,double> clip;
    };
    std::vector<Obs> obs_terms;
    int total_obs_dim = 0;
};

DeployCfg loadDeployCfg(const std::string& yaml_path) {
    DeployCfg cfg;
    YAML::Node y = YAML::LoadFile(yaml_path);

    for (const auto& v : y["joint_ids_map"]) cfg.joint_ids_map.push_back(v.as<int>());
    for (const auto& v : y["stiffness"])     cfg.stiffness.push_back(v.as<double>());
    for (const auto& v : y["damping"])       cfg.damping.push_back(v.as<double>());
    for (const auto& v : y["default_joint_pos"]) cfg.default_joint_pos.push_back(v.as<double>());
    cfg.step_dt = y["step_dt"].as<double>();

    if (y["commands"] && y["commands"]["base_velocity"] && y["commands"]["base_velocity"]["ranges"]) {
        auto r = y["commands"]["base_velocity"]["ranges"];
        cfg.lin_vel_x_min = r["lin_vel_x"][0].as<double>();
        cfg.lin_vel_x_max = r["lin_vel_x"][1].as<double>();
        cfg.lin_vel_y_min = r["lin_vel_y"][0].as<double>();
        cfg.lin_vel_y_max = r["lin_vel_y"][1].as<double>();
        cfg.ang_vel_z_min = r["ang_vel_z"][0].as<double>();
        cfg.ang_vel_z_max = r["ang_vel_z"][1].as<double>();
    }

    auto act = y["actions"]["JointPositionAction"];
    for (const auto& v : act["scale"])  cfg.action_scale.push_back(v.as<double>());
    for (const auto& v : act["offset"]) cfg.action_offset.push_back(v.as<double>());
    if (act["clip"]) {
        for (const auto& pair : act["clip"]) {
            cfg.action_clip.emplace_back(pair[0].as<double>(), pair[1].as<double>());
        }
    }

    for (auto it : y["observations"]) {
        DeployCfg::Obs o;
        o.name = it.first.as<std::string>();
        for (const auto& v : it.second["scale"]) o.scale.push_back(v.as<double>());
        o.dim = static_cast<int>(o.scale.size());
        if (it.second["clip"]) {
            o.clip.first = it.second["clip"][0].as<double>();
            o.clip.second = it.second["clip"][1].as<double>();
        } else {
            o.clip = {-100.0, 100.0};
        }
        cfg.total_obs_dim += o.dim;
        cfg.obs_terms.push_back(std::move(o));
    }
    return cfg;
}

inline Eigen::Vector3d projectedGravity(const Eigen::Quaterniond& q) {
    return q.toRotationMatrix().transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
}

}  // namespace

class Go2RLNode : public rclcpp::Node {
public:
    Go2RLNode() : rclcpp::Node("go2_rl_inference_node") {
        // ---- parameters ----
        std::string onnx_path = declare_parameter<std::string>(
            "onnx_path", "policy/policy.onnx");
        std::string yaml_path = declare_parameter<std::string>(
            "deploy_yaml", "policy/deploy.yaml");
        std::string init_mode = declare_parameter<std::string>("mode", "stand");
        mode_ = parseMode(init_mode);

        RCLCPP_INFO(get_logger(), "Loading deploy.yaml: %s", yaml_path.c_str());
        cfg_ = loadDeployCfg(yaml_path);
        RCLCPP_INFO(get_logger(), "  step_dt=%.3fs  obs_dim=%d  action_dim=%zu",
                    cfg_.step_dt, cfg_.total_obs_dim, cfg_.action_scale.size());

        if ((int)cfg_.joint_ids_map.size() != N_JOINTS) {
            RCLCPP_FATAL(get_logger(), "joint_ids_map has %zu entries, expected %d",
                         cfg_.joint_ids_map.size(), N_JOINTS);
            throw std::runtime_error("bad deploy.yaml");
        }

        // pre-compute SDK-ordered default joint pose for stand mode:
        //   default_joint_pos is in asset order, so default_sdk[joint_ids_map[a]] = default_asset[a].
        default_joint_pos_sdk_.assign(N_JOINTS, 0.0);
        for (int a = 0; a < N_JOINTS; ++a) {
            int sdk = cfg_.joint_ids_map[a];
            default_joint_pos_sdk_[sdk] = cfg_.default_joint_pos[a];
        }

        // ---- ONNX ----
        RCLCPP_INFO(get_logger(), "Loading ONNX: %s", onnx_path.c_str());
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "go2_rl");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*ort_env_, onnx_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i)
            input_strs_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
        for (size_t i = 0; i < session_->GetOutputCount(); ++i)
            output_strs_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
        for (auto& s : input_strs_)  input_names_.push_back(s.c_str());
        for (auto& s : output_strs_) output_names_.push_back(s.c_str());
        RCLCPP_INFO(get_logger(), "ONNX inputs=%zu outputs=%zu",
                    input_names_.size(), output_names_.size());

        last_action_.assign(N_JOINTS, 0.0);

        // ---- ROS 2 pub/sub ----
        torque_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/go2/cmd_torque", 1);
        joint_sub_  = create_subscription<sensor_msgs::msg::JointState>(
            "/go2/joint_states", 1,
            std::bind(&Go2RLNode::jointCb, this, std::placeholders::_1));
        imu_sub_    = create_subscription<sensor_msgs::msg::Imu>(
            "/go2/imu", 1,
            std::bind(&Go2RLNode::imuCb, this, std::placeholders::_1));
        cmd_sub_    = create_subscription<geometry_msgs::msg::Twist>(
            "/go2/cmd_vel", 1,
            std::bind(&Go2RLNode::cmdVelCb, this, std::placeholders::_1));
        mode_sub_   = create_subscription<std_msgs::msg::String>(
            "/go2/mode", 1,
            std::bind(&Go2RLNode::modeCb, this, std::placeholders::_1));

        const auto period = std::chrono::nanoseconds(static_cast<int64_t>(cfg_.step_dt * 1e9));
        timer_ = create_wall_timer(period, std::bind(&Go2RLNode::step, this));
        RCLCPP_INFO(get_logger(), "Go2 RL inference node ready. mode=%s rate=%.0f Hz",
                    init_mode.c_str(), 1.0 / cfg_.step_dt);
    }

private:
    static Mode parseMode(const std::string& s) {
        if (s == "walk")    return Mode::WALK;
        if (s == "passive") return Mode::PASSIVE;
        return Mode::STAND;
    }

    void jointCb(const sensor_msgs::msg::JointState::SharedPtr m) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        for (int i = 0; i < N_JOINTS && i < (int)m->position.size(); ++i) {
            joint_pos_sdk_[i] = m->position[i];
            if (i < (int)m->velocity.size()) joint_vel_sdk_[i] = m->velocity[i];
        }
        joint_received_ = true;
    }
    void imuCb(const sensor_msgs::msg::Imu::SharedPtr m) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        base_quat_ = Eigen::Quaterniond(m->orientation.w, m->orientation.x,
                                        m->orientation.y, m->orientation.z);
        base_ang_vel_ = Eigen::Vector3d(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z);
        imu_received_ = true;
    }
    void cmdVelCb(const geometry_msgs::msg::Twist::SharedPtr m) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        cmd_vx_ = std::clamp(m->linear.x,  cfg_.lin_vel_x_min, cfg_.lin_vel_x_max);
        cmd_vy_ = std::clamp(m->linear.y,  cfg_.lin_vel_y_min, cfg_.lin_vel_y_max);
        cmd_wz_ = std::clamp(m->angular.z, cfg_.ang_vel_z_min, cfg_.ang_vel_z_max);
    }
    void modeCb(const std_msgs::msg::String::SharedPtr m) {
        Mode new_mode = parseMode(m->data);
        if (new_mode != mode_) {
            RCLCPP_INFO(get_logger(), "Mode -> %s", m->data.c_str());
            mode_ = new_mode;
            std::fill(last_action_.begin(), last_action_.end(), 0.0);
        }
    }

    void step() {
        if (!joint_received_ || !imu_received_) return;

        std::array<double, N_JOINTS> q_sdk{}, dq_sdk{};
        Eigen::Quaterniond bq;
        Eigen::Vector3d bw;
        double cmdx, cmdy, cmdwz;
        Mode mode;
        {
            std::lock_guard<std::mutex> lk(state_mutex_);
            q_sdk = joint_pos_sdk_;
            dq_sdk = joint_vel_sdk_;
            bq = base_quat_;
            bw = base_ang_vel_;
            cmdx = cmd_vx_; cmdy = cmd_vy_; cmdwz = cmd_wz_;
            mode = mode_;
        }

        // SDK -> asset reorder (joint_ids_map[asset] = sdk).
        std::vector<double> q_asset(N_JOINTS), dq_asset(N_JOINTS);
        for (int a = 0; a < N_JOINTS; ++a) {
            int sdk = cfg_.joint_ids_map[a];
            q_asset[a]  = q_sdk[sdk];
            dq_asset[a] = dq_sdk[sdk];
        }

        std::vector<float> obs(cfg_.total_obs_dim, 0.0f);
        Eigen::Vector3d g_proj = projectedGravity(bq);
        int off = 0;
        for (const auto& term : cfg_.obs_terms) {
            int s = off;
            if (term.name == "base_ang_vel") {
                double v[3] = {bw.x(), bw.y(), bw.z()};
                for (int i = 0; i < 3; ++i) obs[s + i] = static_cast<float>(v[i] * term.scale[i]);
            } else if (term.name == "projected_gravity") {
                for (int i = 0; i < 3; ++i) obs[s + i] = static_cast<float>(g_proj[i] * term.scale[i]);
            } else if (term.name == "velocity_commands") {
                double v[3] = {cmdx, cmdy, cmdwz};
                if (mode != Mode::WALK) v[0] = v[1] = v[2] = 0.0;
                for (int i = 0; i < 3; ++i) obs[s + i] = static_cast<float>(v[i] * term.scale[i]);
            } else if (term.name == "joint_pos_rel") {
                for (int i = 0; i < N_JOINTS; ++i)
                    obs[s + i] = static_cast<float>((q_asset[i] - cfg_.default_joint_pos[i]) * term.scale[i]);
            } else if (term.name == "joint_vel_rel") {
                for (int i = 0; i < N_JOINTS; ++i)
                    obs[s + i] = static_cast<float>(dq_asset[i] * term.scale[i]);
            } else if (term.name == "last_action") {
                for (int i = 0; i < N_JOINTS; ++i)
                    obs[s + i] = static_cast<float>(last_action_[i] * term.scale[i]);
            } else {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                    "Unknown observation term '%s' — zero-filled.", term.name.c_str());
            }
            for (int i = 0; i < term.dim; ++i)
                obs[s + i] = std::clamp(obs[s + i], static_cast<float>(term.clip.first),
                                                   static_cast<float>(term.clip.second));
            off += term.dim;
        }

        std::vector<int64_t> in_shape{1, cfg_.total_obs_dim};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            mem, obs.data(), obs.size(), in_shape.data(), in_shape.size());

        std::vector<Ort::Value> outs;
        try {
            outs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_.data(), &in_tensor, 1,
                                 output_names_.data(), output_names_.size());
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                                  "ONNX Run failed: %s", e.what());
            return;
        }
        const float* out_data = outs[0].GetTensorData<float>();

        std::vector<double> q_des_asset(N_JOINTS);
        for (int i = 0; i < N_JOINTS; ++i) {
            double raw = static_cast<double>(out_data[i]);
            if (!cfg_.action_clip.empty())
                raw = std::clamp(raw, cfg_.action_clip[i].first, cfg_.action_clip[i].second);
            last_action_[i] = raw;
            q_des_asset[i] = cfg_.action_offset[i] + raw * cfg_.action_scale[i];
        }

        // PD control in SDK order (stiffness / damping are SDK-ordered, q_des
        // from policy is asset-ordered — fold the reorder into the loop).
        std::vector<double> tau_sdk(N_JOINTS, 0.0);
        if (mode != Mode::PASSIVE) {
            for (int a = 0; a < N_JOINTS; ++a) {
                int sdk = cfg_.joint_ids_map[a];
                double q_des_sdk = (mode == Mode::STAND) ? default_joint_pos_sdk_[sdk]
                                                         : q_des_asset[a];
                tau_sdk[sdk] = cfg_.stiffness[sdk] * (q_des_sdk - q_sdk[sdk])
                             - cfg_.damping[sdk]   * dq_sdk[sdk];
            }
        }

        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(tau_sdk.begin(), tau_sdk.end());
        torque_pub_->publish(msg);
    }

    // ---- members ----
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_strs_, output_strs_;
    std::vector<const char*> input_names_, output_names_;

    DeployCfg cfg_;
    std::vector<double> default_joint_pos_sdk_;
    std::vector<double> last_action_;

    std::mutex state_mutex_;
    std::array<double, N_JOINTS> joint_pos_sdk_{}, joint_vel_sdk_{};
    Eigen::Quaterniond base_quat_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d base_ang_vel_ = Eigen::Vector3d::Zero();
    double cmd_vx_ = 0.0, cmd_vy_ = 0.0, cmd_wz_ = 0.0;
    Mode mode_ = Mode::STAND;
    bool joint_received_ = false;
    bool imu_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Go2RLNode>());
    rclcpp::shutdown();
    return 0;
}
