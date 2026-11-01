/* Read-decision head artifact parity transport.
 * Input: uint32 count followed by count rows of NS_READ_HEAD_FEATURES floats.
 * Output: uint32 count followed by count head logits. No backbone inference. */
#include "neural_read_head.h"
#include "neural_slots.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s MEMORY_MODEL BACKBONE HEAD INPUT OUTPUT\n", argv[0]);
        return 2;
    }
    ns_model_t *model = ns_model_load(argv[1], argv[2]);
    ns_read_head_t *head = model ? ns_read_head_load(argv[3], model) : NULL;
    FILE *in = fopen(argv[4], "rb"), *out = fopen(argv[5], "wb");
    int rc = 1;
    uint32_t n;
    if (!head || !in || !out || fread(&n, 4, 1, in) != 1 || n > 1000000 ||
        fwrite(&n, 4, 1, out) != 1)
        goto done;
    for (uint32_t i = 0; i < n; i++) {
        float features[NS_READ_HEAD_FEATURES], logit;
        if (fread(features, sizeof features, 1, in) != 1 ||
            ns_read_head_logit(head, features, &logit) ||
            fwrite(&logit, sizeof logit, 1, out) != 1)
            goto done;
    }
    if (fgetc(in) != EOF || ferror(in))
        goto done;
    rc = 0;
done:
    if (in)
        fclose(in);
    if (out && fclose(out))
        rc = 1;
    ns_read_head_free(head);
    ns_model_free(model);
    return rc;
}
