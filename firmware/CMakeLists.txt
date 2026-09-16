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

# Our changes to the pinned pico-tflmicro live as patches rather than as commits
# in the submodule, so the submodule stays on an upstream commit and the diff is
# reviewable in our own history. Applying them dirties the submodule's working
# tree, so `git status` will report it as modified - that is expected.
# See docs/journal/02-tflm-embedded.md.
set(TFLM_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/pico-tflmicro)
file(GLOB TFLM_PATCHES ${CMAKE_CURRENT_LIST_DIR}/../third_party/patches/*.patch)
list(SORT TFLM_PATCHES)
find_package(Git REQUIRED)
foreach(patch ${TFLM_PATCHES})
    get_filename_component(patch_name ${patch} NAME)
    # A patch that reverse-applies cleanly is already in the tree.
    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --reverse --check ${patch}
        WORKING_DIRECTORY ${TFLM_DIR}
        RESULT_VARIABLE already_applied
        ERROR_QUIET)
    if(already_applied EQUAL 0)
        message(STATUS "pico-tflmicro patch already applied: ${patch_name}")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply ${patch}
            WORKING_DIRECTORY ${TFLM_DIR}
            RESULT_VARIABLE apply_result
            ERROR_VARIABLE apply_error)
        if(NOT apply_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to apply ${patch_name} to pico-tflmicro: ${apply_error}\n"
                "If the submodule was bumped, the patches need rebasing.")
        endif()
        message(STATUS "pico-tflmicro patch applied: ${patch_name}")
    endif()
endforeach()

# Provides the pico-tflmicro library target. EXCLUDE_FROM_ALL so its ~100
# tests and examples are only defined, not built.
add_subdirectory(../third_party/pico-tflmicro pico-tflmicro EXCLUDE_FROM_ALL)

add_subdirectory(hello)
add_subdirectory(person_detect_embedded)
add_subdirectory(person_detect_profile)
add_subdirectory(person_detect_serial)
