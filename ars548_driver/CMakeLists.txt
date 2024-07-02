cmake_minimum_required(VERSION 3.8)
project(ars548_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(catkin REQUIRED
  roscpp
  ars548_messages
  sensor_msgs
  geometry_msgs
  tf2_geometry_msgs
)

catkin_package(
  # CATKIN_DEPENDS roscpp ars548_messages sensor_msgs geometry_msgs tf2_geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(ars548_driver_node src/ars548_driver_node.cpp)
target_link_libraries(ars548_driver_node ${catkin_LIBRARIES})
add_dependencies(ars548_driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(ars548_driver_filter src/ars548_driver_filter.cpp)
target_link_libraries(ars548_driver_filter ${catkin_LIBRARIES})
add_dependencies(ars548_driver_filter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

install(TARGETS
  ars548_driver_node
  ars548_driver_filter
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(roslint)
  if(roslint_FOUND)
    roslint_cpp()
    roslint_add_test()
  endif()
endif()
