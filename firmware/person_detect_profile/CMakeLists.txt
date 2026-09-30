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

# Per-op profile of a trained model, for filling gaps in the cost model.
# CUSTOM_MODEL and CUSTOM_EXTENDED_OPS: see firmware/CMakeLists.txt.
add_executable(person_detect_profile_custom
    main.cpp
    ${TFLM_EXAMPLE}/model_settings.cpp
    ${TFLM_PERSON_MODEL}/person_image_data.cpp
)
target_embed_custom_model(person_detect_profile_custom)
target_include_directories(person_detect_profile_custom PRIVATE ${TFLM_EXAMPLE} ${TFLM_PERSON_MODEL})
target_compile_definitions(person_detect_profile_custom PRIVATE MODEL_IN_SRAM=1)
target_compile_definitions(person_detect_profile_custom PRIVATE
    EXTENDED_OPS=$<BOOL:${CUSTOM_EXTENDED_OPS}>)
target_link_libraries(person_detect_profile_custom pico_stdlib pico-tflmicro)
pico_enable_stdio_usb(person_detect_profile_custom 1)
pico_enable_stdio_uart(person_detect_profile_custom 0)
pico_add_extra_outputs(person_detect_profile_custom)
