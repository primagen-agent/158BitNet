#ifndef UNICODE_DATA_H
#define UNICODE_DATA_H

#include <stdint.h>

/* Codepoint category flags (same values as llama.cpp's unicode_cpt_flags). */
enum {
    BQ_U_UNDEFINED   = 0x0001,
    BQ_U_NUMBER      = 0x0002,  /* \p{N} */
    BQ_U_LETTER      = 0x0004,  /* \p{L} */
    BQ_U_SEPARATOR   = 0x0008,  /* \p{Z} */
    BQ_U_ACCENT_MARK = 0x0010,  /* \p{M} */
    BQ_U_PUNCTUATION = 0x0020,  /* \p{P} */
    BQ_U_SYMBOL      = 0x0040,  /* \p{S} */
    BQ_U_CONTROL     = 0x0080,  /* \p{C} */
    BQ_U_WHITESPACE  = 0x0100
};

typedef struct {
    uint32_t start;
    uint32_t flags;
} bq_urange_t;

extern const bq_urange_t bq_unicode_ranges[];
extern const int bq_unicode_ranges_count;

/* Returns the category flags of a codepoint (0 for out-of-range). */
uint32_t bq_unicode_flags(uint32_t cpt);

#endif
