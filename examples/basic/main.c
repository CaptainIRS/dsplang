#if __hexagon__
#include "q6sim_timer.h"
#include "HAP_perf.h"
#endif
#include "stdio.h"
#include "stdint.h"

#define LOG_TAG         "dsplanglib"      
#define LOG(...)        printf(LOG_TAG ": " __VA_ARGS__)

#if __hexagon__
#define RESET_PMU()     __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU()      __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define READ_PCYCLES    q6sim_read_pcycles
#endif

void dsplanglib(
    float *data1, float *aligned1, int offset1, uint64_t rank1, uint64_t stride1,
    float *data2, float *aligned2, int offset2, uint64_t rank2, uint64_t stride2,
    float *data3, float *aligned3, int offset3, uint64_t rank3, uint64_t stride3
);

int main(int argc, char *argv[]) {
#if __hexagon__
    long long start_cycles, total_cycles, start_time, end_time, duration;
    RESET_PMU();
    start_time = HAP_perf_get_qtimer_count();
    start_cycles = READ_PCYCLES();
#endif
    float data1[10000];
    float data2[10000];
    float data3[10000];
    for (int i = 0; i < 10000; i++) {
        data1[i] = 5.0;
        data2[i] = 6.0;
    }
    uint64_t rank = 10000;
    uint64_t stride = 1;
    dsplanglib(data1, data1, 0, rank, stride, data2, data2, 0, rank, stride, data3, data3, 0, rank, stride);
#if __hexagon__
    total_cycles = READ_PCYCLES() - start_cycles;
    end_time = HAP_perf_get_qtimer_count();
    DUMP_PMU();
    LOG("Total cycles: %lld\n", total_cycles);
    duration = HAP_perf_qtimer_count_to_us(end_time - start_time);
    LOG("Duration: %lld us\n", duration);
#endif
    LOG();
    for (int i = 0; i < 100; i++) {
        printf("%f ", data3[i]);
    }
    printf("\n");
    LOG();
    for (int i = 9900; i < 10000; i++) {
        printf("%f ", data3[i]);
    }
    printf("\n");
    return 0;
}
