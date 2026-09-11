// Smallest useful firmware: proves that building, flashing and USB serial
// output all work. Blinks the LED and prints once per second.
#include <inttypes.h>
#include <stdio.h>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

// Linker symbol marking the end of the program image in flash.
extern char __flash_binary_end;

int main(void) {
    stdio_init_all();

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    uint32_t binary_size = (uint32_t)&__flash_binary_end - XIP_BASE;

    for (uint32_t n = 0;; n++) {
        gpio_put(PICO_DEFAULT_LED_PIN, n & 1);
        printf("hello from RP2350 #%" PRIu32 ": clk_sys=%" PRIu32 " Hz, binary=%" PRIu32 " bytes\n",
               n, clock_get_hz(clk_sys), binary_size);
        sleep_ms(1000);
    }
}
