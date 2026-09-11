cmake_minimum_required(VERSION 3.13)

# Use the SDK pinned as a submodule, not whatever is installed on the host.
set(PICO_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../third_party/pico-sdk)
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Must come before project() so the SDK can set up the cross toolchain.
include(${PICO_SDK_PATH}/pico_sdk_init.cmake)

project(pico_person_detection C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_subdirectory(hello)
