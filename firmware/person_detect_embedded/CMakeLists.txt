# Model, sample images and model_settings come straight from pico-tflmicro's
# person_detection example; only main.cpp is ours.
set(TFLM_EXAMPLE ${CMAKE_CURRENT_LIST_DIR}/../../third_party/pico-tflmicro/examples/person_detection)
set(TFLM_PERSON_MODEL ${TFLM_EXAMPLE}/tensorflow/lite/micro/tools/make/downloads/person_model)

add_executable(person_detect_embedded
    main.cpp
    ${TFLM_EXAMPLE}/model_settings.cpp
    ${TFLM_PERSON_MODEL}/person_detect_model_data.cpp
    ${TFLM_PERSON_MODEL}/person_image_data.cpp
    ${TFLM_PERSON_MODEL}/no_person_image_data.cpp
)

target_include_directories(person_detect_embedded PRIVATE ${TFLM_EXAMPLE})

target_link_libraries(person_detect_embedded pico_stdlib pico-tflmicro)

pico_enable_stdio_usb(person_detect_embedded 1)
pico_enable_stdio_uart(person_detect_embedded 0)

pico_add_extra_outputs(person_detect_embedded)
