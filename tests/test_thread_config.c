#include "thread_config.h"

int main(void) {
    /* CTest supplies BITNET_NUM_THREADS=99.  Every consumer must clamp the
     * shared requested value to its own fixed worker capacity. */
    if (bitnet_thread_count(16) != 16) return 1;
    if (bitnet_thread_count(8) != 8) return 2;
    if (bitnet_thread_count(1) != 1) return 3;
    return 0;
}
