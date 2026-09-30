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

# The *_custom targets embed a trained model in place of pico-tflmicro's: the main
# one by default, or -DCUSTOM_MODEL=<path to .tflite>. The build converts it to a
# C array, and converts it again whenever the file changes.
set(CUSTOM_MODEL ${CMAKE_CURRENT_LIST_DIR}/../models/v1-80k.tflite
    CACHE FILEPATH "The .tflite the *_custom targets embed")

# v3 and MCUNet need ADD, MUL and HARD_SWISH, and EXTENDED_OPS also makes room for
# their larger arena, which leaves a model buffer too small for v1.
# tools/check_ops.py says which a model needs. Both settings are cached, so
# change them together.
option(CUSTOM_EXTENDED_OPS "Build the *_custom targets with EXTENDED_OPS=1" OFF)
message(STATUS "*_custom targets: ${CUSTOM_MODEL}, CUSTOM_EXTENDED_OPS=${CUSTOM_EXTENDED_OPS}")

# Adds CUSTOM_MODEL, converted to a C array, to a firmware target's sources.
# tools/tflite_to_c.py uses only the standard library, so any python3 will do.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TFLITE_TO_C ${CMAKE_CURRENT_LIST_DIR}/../tools/tflite_to_c.py)
function(target_embed_custom_model target)
    set(source ${CMAKE_CURRENT_BINARY_DIR}/model_data.cpp)
    add_custom_command(
        OUTPUT ${source}
        COMMAND Python3::Interpreter ${TFLITE_TO_C} ${CUSTOM_MODEL} -o ${source}
        DEPENDS ${CUSTOM_MODEL} ${TFLITE_TO_C}
    )
    target_sources(${target} PRIVATE ${source})
endfunction()

add_subdirectory(hello)
add_subdirectory(person_detect_embedded)
add_subdirectory(person_detect_profile)
add_subdirectory(person_detect_serial)
