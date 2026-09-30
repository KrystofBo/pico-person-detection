// Runs the TFLM person detection model on 96x96 int8 images streamed from the
// host over USB serial, so a whole folder can be tested without reflashing.
//
// Protocol
//   host -> pico : 'P','I','M','G' then kInputBytes raw int8 pixels
//   pico -> host : one text line, either
//                    OK person=<int8> no_person=<int8> time=<us> verdict=<person|no_person>
//                  or
//                    ERR <reason>
//
// The verdict is the argmax of the two scores, so a tie is not a person, and
// the onboard LED shows it: on while the latest frame contains a person.
//
// Binary one way, text the other, deliberately: the SDK's USB stdio translates
// \n to \r\n on output but leaves input alone, so a binary reply would be
// silently corrupted while a binary request is safe.
#include <cstdio>
#include <cstring>

#include "pico/stdlib.h"

#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// EXTENDED_OPS trades flash and SRAM for the three operators MobileNetV3 needs.
// Off by default: a MobileNet v1 model uses none of them, and enabling it costs
// ~30 KB of flash and ~50 KB of SRAM for nothing. The person_detect_*_custom
// targets turn it on when configured with -DCUSTOM_EXTENDED_OPS=ON.
#ifndef EXTENDED_OPS
#define EXTENDED_OPS 0
#endif

// v1 uses 82,308 bytes of arena; v3-Small measured 79,428 but needs a much
// larger model buffer. The firmware prints what was actually used.
constexpr int kTensorArenaSize = (EXTENDED_OPS ? 184 : 88) * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// The CMSIS-NN kernels split each conv across both cores; reading weights from
// flash makes them contend for the shared XIP cache. See the step 02 journal.
// v1 is 303,496 bytes; v3-Small is 341,312, because its 110 operators carry far
// more flatbuffer overhead than v1's 31 despite having fewer parameters.
alignas(16) static uint8_t model_sram[(EXTENDED_OPS ? 296 : 301) * 1024];

constexpr int kInputBytes = kNumCols * kNumRows * kNumChannels;

// No timeout while waiting for a new frame to start, but once a frame is in
// flight a stall means the host died mid-send and we should resynchronise.
constexpr uint32_t kFrameByteTimeoutUs = 2 * 1000 * 1000;

// Reads whatever the USB FIFO holds per call - up to 64 bytes at full speed -
// rather than one getchar_timeout_us() per byte. The FIFO is all the device can
// buffer, so the host can only send the next packet once it is drained, and
// draining it a byte at a time made moving a frame cost ~56 ms.
static bool read_exact(int8_t* dst, int count) {
    char* buf = reinterpret_cast<char*>(dst);
    while (count > 0) {
        int n = stdio_get_until(buf, count, make_timeout_time_us(kFrameByteTimeoutUs));
        if (n == PICO_ERROR_TIMEOUT) {
            return false;
        }
        buf += n;
        count -= n;
    }
    return true;
}

// Scans the stream until the magic has been seen, so a desynchronised host can
// recover just by sending another frame.
static void wait_for_magic() {
    static const char kMagic[] = {'P', 'I', 'M', 'G'};
    int matched = 0;
    while (matched < 4) {
        int c = getchar_timeout_us(1000 * 1000);
        if (c == PICO_ERROR_TIMEOUT) {
            // No frame for a second: the last verdict is stale, stop showing it.
            gpio_put(PICO_DEFAULT_LED_PIN, false);
            continue;
        }
        matched = (c == kMagic[matched]) ? matched + 1 : (c == kMagic[0] ? 1 : 0);
    }
}

int main() {
    stdio_init_all();
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    // Unchecked, a model larger than the buffer overwrites whatever follows it.
    if ((size_t)g_person_detect_model_data_len > sizeof(model_sram)) {
        while (true) {
            printf("ERR model_too_large need=%d have=%u\n", g_person_detect_model_data_len,
                   (unsigned)sizeof(model_sram));
            sleep_ms(2000);
        }
    }
    memcpy(model_sram, g_person_detect_model_data, g_person_detect_model_data_len);
    const tflite::Model* model = tflite::GetModel(model_sram);

    static tflite::MicroMutableOpResolver<EXTENDED_OPS ? 8 : 5> resolver;
    resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
    resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
    resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
    resolver.AddReshape();
    resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
#if EXTENDED_OPS
    // MobileNetV3: residual connections, the squeeze-excite rescale, and h-swish.
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddHardSwish();
#endif

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        while (true) {
            printf("ERR allocate_tensors\n");
            sleep_ms(2000);
        }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if ((int)input->bytes != kInputBytes) {
        while (true) {
            printf("ERR input_size %d expected %d\n", (int)input->bytes, kInputBytes);
            sleep_ms(2000);
        }
    }

    printf("READY person_detect_serial %dx%d arena=%u\n", kNumCols, kNumRows,
           (unsigned)interpreter.arena_used_bytes());

    while (true) {
        wait_for_magic();

        // Read straight into the input tensor rather than through a staging
        // buffer: that buffer cost 9,216 bytes of SRAM, which matters when a
        // model and its arena are competing for 520 KB. A truncated frame
        // leaves the tensor half-written, which is harmless because we report
        // the error and the host resends before anything reads it.
        if (!read_exact(input->data.int8, kInputBytes)) {
            printf("ERR short_frame\n");
            continue;
        }

        uint64_t start = time_us_64();
        TfLiteStatus status = interpreter.Invoke();
        uint64_t elapsed_us = time_us_64() - start;

        if (status != kTfLiteOk) {
            printf("ERR invoke\n");
            continue;
        }

        const int8_t person = output->data.int8[kPersonIndex];
        const int8_t no_person = output->data.int8[kNotAPersonIndex];
        const bool is_person = person > no_person;
        gpio_put(PICO_DEFAULT_LED_PIN, is_person);
        printf("OK person=%d no_person=%d time=%llu verdict=%s\n", person, no_person, elapsed_us,
               is_person ? "person" : "no_person");
    }
}
