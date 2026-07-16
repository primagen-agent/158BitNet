#include "thread_config.h"

#include <pthread.h>
#include <stdlib.h>

static pthread_once_t g_thread_count_once = PTHREAD_ONCE_INIT;
static int g_requested_threads = 3;

static void bitnet_read_thread_count_once(void) {
    const char *env = getenv("BITNET_NUM_THREADS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env && *end == '\0' && parsed > 0) {
            if (parsed > 256) parsed = 256;
            g_requested_threads = (int)parsed;
        }
    }
}

int bitnet_thread_count(int max_threads) {
    int result;
    (void)pthread_once(&g_thread_count_once, bitnet_read_thread_count_once);
    result = g_requested_threads;
    if (result < 1) result = 1;
    if (max_threads > 0 && result > max_threads) result = max_threads;
    return result;
}
