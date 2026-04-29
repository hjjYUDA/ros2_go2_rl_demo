cmake_minimum_required(VERSION 3.10)
project(go2_rl_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# ONNX Runtime (manually installed under /opt/onnxruntime — see README).
set(ONNXRUNTIME_ROOT /opt/onnxruntime CACHE PATH "Path to onnxruntime install")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  REQUIRED)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  REQUIRED)

# -- C++ node: RL inference -------------------------------------------------
add_executable(rl_inference_node src/rl_inference_node.cpp)
target_include_directories(rl_inference_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)
ament_target_dependencies(rl_inference_node
  rclcpp sensor_msgs std_msgs geometry_msgs
)
target_link_libraries(rl_inference_node
  ${ONNXRUNTIME_LIB}
  yaml-cpp
)
install(TARGETS rl_inference_node
  DESTINATION lib/${PROJECT_NAME}
)

# -- Python nodes -----------------------------------------------------------
install(PROGRAMS
  scripts/mujoco_sim_node.py
  scripts/keyboard_teleop_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# -- Launch / description / policy ------------------------------------------
install(DIRECTORY launch description policy
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
