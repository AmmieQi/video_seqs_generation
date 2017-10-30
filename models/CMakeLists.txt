cmake_minimum_required(VERSION 3.7)
project(deform3d_new_pytorch)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        "/home/lin/anaconda2/lib/python2.7/site-packages/torch/lib/include"
        "/usr/local/cuda/include"
        "./"
)

add_subdirectory(lib)
