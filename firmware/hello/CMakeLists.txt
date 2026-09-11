add_executable(hello main.c)

target_link_libraries(hello pico_stdlib)

# printf goes over USB CDC (shows up as /dev/ttyACM0), not the UART pins.
pico_enable_stdio_usb(hello 1)
pico_enable_stdio_uart(hello 0)

# Produce .uf2 / .bin / .dis alongside the .elf.
pico_add_extra_outputs(hello)
