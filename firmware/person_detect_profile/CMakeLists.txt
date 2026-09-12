# Model, sample image and model_settings come straight from pico-tflmicro's
# person_detection example; only main.cpp is ours.
set(TFLM_EXAMPLE ${CMAKE_CURRENT_LIST_DIR}/../../third_party/pico-tflmicro/examples/person_detection)
set(TFLM_PERSON_MODEL ${TFLM_EXAMPLE}/tensorflow/lite/micro/tools/make/downloads/person_model)

# Two variants of the same source: the model read from flash over XIP, and the
# model copied into SRAM. The comparison is the point - see the step 02 journal.
foreach(variant person_detect_profile person_detect_profile_sram)
    add_executable(${variant}
        main.cpp
        ${TFLM_EXAMPLE}/model_settings.cpp
        ${TFLM_PERSON_MODEL}/person_detect_model_data.cpp
        ${TFLM_PERSON_MODEL}/person_image_data.cpp
    )

    target_include_directories(${variant} PRIVATE ${TFLM_EXAMPLE})
    target_link_libraries(${variant} pico_stdlib pico-tflmicro)

    pico_enable_stdio_usb(${variant} 1)
    pico_enable_stdio_uart(${variant} 0)

    pico_add_extra_outputs(${variant})
endforeach()

target_compile_definitions(person_detect_profile_sram PRIVATE MODEL_IN_SRAM=1)
