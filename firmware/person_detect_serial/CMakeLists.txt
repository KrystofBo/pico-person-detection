# Model and model_settings come from pico-tflmicro's person_detection example;
# the sample image arrays are not needed - images arrive over USB.
set(TFLM_EXAMPLE ${CMAKE_CURRENT_LIST_DIR}/../../third_party/pico-tflmicro/examples/person_detection)
set(TFLM_PERSON_MODEL ${TFLM_EXAMPLE}/tensorflow/lite/micro/tools/make/downloads/person_model)

# At full speed the USB endpoint buffer defaults to a single 64-byte packet, and
# the Pico took one about every 120 us: 17 ms to receive a frame. A 512-byte
# buffer takes eight at a time, and a frame arrives in 9 ms. TinyUSB re-arms the
# endpoint only while the receive FIFO has a whole buffer free, so the FIFO holds
# two, or every transfer would wait for the firmware to read the last one.
set(USB_BUFFERS CFG_TUD_CDC_EP_BUFSIZE=512 CFG_TUD_CDC_RX_BUFSIZE=1024)

add_executable(person_detect_serial
    main.cpp
    ${TFLM_EXAMPLE}/model_settings.cpp
    ${TFLM_PERSON_MODEL}/person_detect_model_data.cpp
)

target_include_directories(person_detect_serial PRIVATE ${TFLM_EXAMPLE})
target_link_libraries(person_detect_serial pico_stdlib pico-tflmicro)
target_compile_definitions(person_detect_serial PRIVATE ${USB_BUFFERS})

pico_enable_stdio_usb(person_detect_serial 1)
pico_enable_stdio_uart(person_detect_serial 0)

pico_add_extra_outputs(person_detect_serial)

# Same firmware, linking a trained model instead of pico-tflmicro's. The
# generated file defines the same symbols, so main.cpp is unchanged. Only
# defined when the model has been generated:
#   python tools/tflite_to_c.py <run>/model_int8.tflite -o firmware/generated/model_data.cpp
set(CUSTOM_MODEL ${CMAKE_CURRENT_LIST_DIR}/../generated/model_data.cpp)
if(EXISTS ${CUSTOM_MODEL})
    add_executable(person_detect_serial_custom
        main.cpp
        ${TFLM_EXAMPLE}/model_settings.cpp
        ${CUSTOM_MODEL}
    )
    target_include_directories(person_detect_serial_custom PRIVATE ${TFLM_EXAMPLE} ${TFLM_PERSON_MODEL})
    # CUSTOM_EXTENDED_OPS: see firmware/CMakeLists.txt.
    target_compile_definitions(person_detect_serial_custom PRIVATE
        EXTENDED_OPS=$<BOOL:${CUSTOM_EXTENDED_OPS}> ${USB_BUFFERS})
    target_link_libraries(person_detect_serial_custom pico_stdlib pico-tflmicro)
    pico_enable_stdio_usb(person_detect_serial_custom 1)
    pico_enable_stdio_uart(person_detect_serial_custom 0)
    pico_add_extra_outputs(person_detect_serial_custom)
    message(STATUS "person_detect_serial_custom: using ${CUSTOM_MODEL}, "
                   "CUSTOM_EXTENDED_OPS=${CUSTOM_EXTENDED_OPS}")
endif()
