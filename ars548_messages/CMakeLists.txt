cmake_minimum_required(VERSION 3.8)
project(ars548_messages)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖项
find_package(catkin REQUIRED
  message_generation
  std_msgs
)

# 添加消息文件
add_message_files(
  FILES
  Detection.msg
  DetectionList.msg
  Object.msg
  ObjectList.msg
  Status.msg
  AccelerationLateralCoG.msg
  AccelerationLongitudinalCoG.msg
  CharacteristicSpeed.msg
  DrivingDirection.msg
  SensorConfiguration.msg
  SteeringAngleFrontAxle.msg
  VelocityVehicle.msg
  YawRate.msg
)

# 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
)

# catkin 包设置
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)

# 测试设置
if(CATKIN_ENABLE_TESTING)
  find_package(catkin REQUIRED COMPONENTS
    roslint
  )
  roslint_cpp()
  roslint_add_test()
endif()
