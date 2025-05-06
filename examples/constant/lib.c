#include <stdint.h>
#include <memory.h>

int dsplanglib(
    short *data1, short *aligned1, int offset1, uint64_t rank1, uint64_t stride1,
    short *data2, short *aligned2, int offset2, uint64_t rank2, uint64_t stride2,
    short *data3, short *aligned3, int offset3, uint64_t rank3, uint64_t stride3
) {
    short a[70] = {70, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4};
    short b[70] = {70, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 0, 1, 2, 3, 4};
    for (uint64_t i = 0; i < rank1; i++) {
        data3[i] = (data1[i] * data2[i]) * (a[i] + b[i]);
    }
    return 0;
}
