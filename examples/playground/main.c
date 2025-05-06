#if __hexagon__
#include "q6sim_timer.h"
#endif
#include "stdio.h"
#include "stdint.h"

#if __hexagon__
#define RESET_PMU()     __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU()      __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define READ_PCYCLES    q6sim_read_pcycles
#endif

void dsplanglib(
    short *data1, short *aligned1, int offset1, uint64_t rank1, uint64_t stride1,
    short *data2, short *aligned2, int offset2, uint64_t rank2, uint64_t stride2,
    short *data3, short *aligned3, int offset3, uint64_t rank3, uint64_t stride3
);

int main() {
#if __hexagon__
    long long start_time, total_cycles;
    RESET_PMU();
    start_time = READ_PCYCLES();
#endif
    short data1[70];
    short data2[70];
    short data3[70];
    for (int i = 0; i < 70; i++) {
        data1[i] = 5;
        data2[i] = 6;
    }
    uint64_t rank = 70;
    uint64_t stride = 1;
    dsplanglib(data1, data1, 0, rank, stride, data2, data2, 0, rank, stride, data3, data3, 0, rank, stride);
#if __hexagon__
    total_cycles = READ_PCYCLES() - start_time;
    DUMP_PMU();
    printf("Total cycles: %lld\n", total_cycles);
#endif
    for (int i = 0; i < 70; i++) {
        printf("%d ", data3[i]);
    }
    printf("\n");
    return 0;
}
