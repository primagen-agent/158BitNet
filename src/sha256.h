#ifndef BITNET_SHA256_H
#define BITNET_SHA256_H

#include <stdint.h>

int bitnet_sha256_file(const char *path, uint8_t digest[32]);

#endif
