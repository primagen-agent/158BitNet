#ifndef BITNET_THREAD_CONFIG_H
#define BITNET_THREAD_CONFIG_H

/* Process-wide inference thread setting.  The environment is intentionally
 * read once because the persistent worker pools cannot be resized safely. */
int bitnet_thread_count(int max_threads);

#endif
