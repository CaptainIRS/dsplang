#include <stdint.h>

int dsplanglib(
    float *data1, float *aligned1, int offset1, uint64_t rank1, uint64_t stride1,
    float *data2, float *aligned2, int offset2, uint64_t rank2, uint64_t stride2,
    float *data3, float *aligned3, int offset3, uint64_t rank3, uint64_t stride3
) {
    for (uint64_t i = 0; i < rank1; i++) {
        data3[i] = (data1[i] * data2[i]) * (data1[i] + data2[i]);
    }
    return 0;
}