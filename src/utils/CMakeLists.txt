cmake_minimum_required(VERSION 3.14)
project(dandelion-utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 UTF-8 编码
if(MSVC)
    add_compile_options(/utf-8)
endif()

# 查找 Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 设置包含目录
set(DANDELION_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories(${DANDELION_ROOT_DIR})

# 生成 dandelion-ray 静态库
add_library(dandelion-ray STATIC ray.cpp)
target_link_libraries(dandelion-ray PUBLIC Eigen3::Eigen)

# 生成 dandelion-bvh 静态库
add_library(dandelion-bvh STATIC aabb.cpp bvh.cpp)
target_link_libraries(dandelion-bvh PUBLIC Eigen3::Eigen)

# 安装路径可选（生成的 lib 文件会放在 build/{Debug,Release}）
