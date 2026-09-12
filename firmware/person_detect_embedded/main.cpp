// Runs the pretrained TFLM person detection model on the two sample images
// that ship with pico-tflmicro, and reports scores, latency and memory use.
// Runs in a loop so the output can be read whenever a terminal is opened.
#include <cstdio>
#include <cstring>

#include "pico/stdlib.h"

#include "model_settings.h"
#include "no_person_image_data.h"
#include "person_detect_model_data.h"
#include "person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 82,308 bytes are actually used; the rest is headroom the SRAM copy below needs.
constexpr int kTensorArenaSize = 88 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// The CMSIS-NN kernels now split each conv across both cores, and both cores
// stream weights over the same QSPI port through one shared 16 KB XIP cache.
// Reading the model from flash makes the larger layers slower than single-core;
// copying it into SRAM is what lets the split actually pay off. See the step 02
// journal for the measurements.
alignas(16) static uint8_t model_sram[301 * 1024];

// Linker symbol marking the end of the program image in flash.
extern "C" char __flash_binary_end;

int main() {
    stdio_init_all();

    if ((size_t)g_person_detect_model_data_len > sizeof(model_sram)) {
        while (true) {
            printf("model_sram too small: need %d bytes\n", g_person_detect_model_data_len);
            sleep_ms(2000);
        }
    }
    memcpy(model_sram, g_person_detect_model_data, g_person_detect_model_data_len);

    const tflite::Model* model = tflite::GetModel(model_sram);

    // Only the ops this model uses, with the CMSIS-NN optimized int8 kernels.
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
    resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
    resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
    resolver.AddReshape();
    resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    TfLiteStatus alloc_status = interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    struct Sample {
        const char* name;
        const uint8_t* data;
    };
    const Sample samples[] = {
        {"person", g_person_image_data},
        {"no_person", g_no_person_image_data},
    };

    while (true) {
        if (alloc_status != kTfLiteOk) {
            printf("AllocateTensors failed (arena too small?)\n");
            sleep_ms(2000);
            continue;
        }

        printf("--- model=%d bytes, arena used=%u of %d bytes, binary=%lu bytes\n",
               g_person_detect_model_data_len, (unsigned)interpreter.arena_used_bytes(),
               kTensorArenaSize, (unsigned long)((uintptr_t)&__flash_binary_end - XIP_BASE));

        for (const Sample& s : samples) {
            memcpy(input->data.int8, s.data, input->bytes);

            uint64_t start = time_us_64();
            TfLiteStatus status = interpreter.Invoke();
            uint64_t elapsed_us = time_us_64() - start;

            if (status != kTfLiteOk) {
                printf("%-9s Invoke failed\n", s.name);
                continue;
            }

            int8_t person = output->data.int8[kPersonIndex];
            int8_t no_person = output->data.int8[kNotAPersonIndex];
            float scale = output->params.scale;
            int zero_point = output->params.zero_point;
            printf("%-9s person=%4d (%.3f) no_person=%4d (%.3f) time=%llu us\n", s.name, person,
                   (person - zero_point) * scale, no_person, (no_person - zero_point) * scale,
                   elapsed_us);
        }
        sleep_ms(2000);
    }
}
