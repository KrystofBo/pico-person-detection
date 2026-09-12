// Per-operator profile of the TFLM person detection model, to find out where the
// ~190 ms of an Invoke() actually goes. Same model and images as
// person_detect_embedded; the only addition is a profiler hooked into the
// interpreter. Runs in a loop so the output can be read whenever a terminal is opened.
#include <cstdio>
#include <cstring>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

#include "model_settings.h"
#include "person_detect_model_data.h"
#include "person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 82,308 bytes are actually used; the rest of step 02's 136 KB is headroom we
// need back when MODEL_IN_SRAM also wants ~294 KB of the 520 KB of SRAM.
constexpr int kTensorArenaSize = 88 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

#if MODEL_IN_SRAM
// Both cores stream weights over the same QSPI port through one shared 16 KB
// XIP cache, so splitting a layer whose weights exceed the cache makes it
// slower, not faster. Copying the model into SRAM removes flash from the inner
// loop entirely. Sized for the known 300,568-byte model and checked at startup.
alignas(16) static uint8_t model_sram[301 * 1024];
#endif

// TFLM's own MicroProfiler statically reserves ~80 KB for 4096 events. This model
// has 31 operators, so a fixed 64 slots is enough and costs about 1 KB.
class OpProfiler : public tflite::MicroProfilerInterface {
  public:
    uint32_t BeginEvent(const char* tag) override {
        if (count_ >= kMaxEvents) {
            return kMaxEvents;  // overflow sink, EndEvent ignores it
        }
        tags_[count_] = tag;
        start_us_[count_] = time_us_32();
        return count_++;
    }

    void EndEvent(uint32_t handle) override {
        if (handle < kMaxEvents) {
            elapsed_us_[handle] = time_us_32() - start_us_[handle];
        }
    }

    void Clear() { count_ = 0; }

    void Print() const {
        uint32_t total = 0;
        for (uint32_t i = 0; i < count_; i++) {
            printf("%2lu,%s,%lu\n", (unsigned long)i, tags_[i],
                   (unsigned long)elapsed_us_[i]);
            total += elapsed_us_[i];
        }
        printf("sum of %lu ops = %lu us\n", (unsigned long)count_, (unsigned long)total);
    }

  private:
    static constexpr uint32_t kMaxEvents = 64;
    const char* tags_[kMaxEvents];
    uint32_t start_us_[kMaxEvents];
    uint32_t elapsed_us_[kMaxEvents];
    uint32_t count_ = 0;
};

static OpProfiler profiler;

int main() {
    stdio_init_all();

    const uint8_t* model_data = g_person_detect_model_data;
#if MODEL_IN_SRAM
    if ((size_t)g_person_detect_model_data_len > sizeof(model_sram)) {
        while (true) {
            printf("model_sram too small: need %d bytes\n", g_person_detect_model_data_len);
            sleep_ms(2000);
        }
    }
    memcpy(model_sram, g_person_detect_model_data, g_person_detect_model_data_len);
    model_data = model_sram;
#endif

    const tflite::Model* model = tflite::GetModel(model_data);

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
    resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
    resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
    resolver.AddReshape();
    resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                                kTensorArenaSize, nullptr, &profiler);
    TfLiteStatus alloc_status = interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    while (true) {
        if (alloc_status != kTfLiteOk) {
            printf("AllocateTensors failed (arena too small?)\n");
            sleep_ms(2000);
            continue;
        }

        printf("--- clk_sys=%lu Hz, arena used=%u bytes, model in %s\n",
               (unsigned long)clock_get_hz(clk_sys), (unsigned)interpreter.arena_used_bytes(),
               model_data == g_person_detect_model_data ? "flash" : "SRAM");

        memcpy(input->data.int8, g_person_image_data, input->bytes);

        profiler.Clear();
        uint64_t start = time_us_64();
        TfLiteStatus status = interpreter.Invoke();
        uint64_t elapsed_us = time_us_64() - start;

        if (status != kTfLiteOk) {
            printf("Invoke failed\n");
            sleep_ms(2000);
            continue;
        }

        printf("idx,op,us\n");
        profiler.Print();
        printf("Invoke total = %llu us, person=%d no_person=%d\n", elapsed_us,
               output->data.int8[kPersonIndex], output->data.int8[kNotAPersonIndex]);
        sleep_ms(2000);
    }
}
