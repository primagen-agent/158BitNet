#include "neural_slots.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Feature transport for supervised training. Input contains only texts. */
int main(int argc, char **argv) {
    if (argc != 4)
        return 2;
    bitnet_model_t *backbone = bitnet_load_model(argv[1]);
    if (!backbone)
        return 3;
    FILE *in = fopen(argv[2], "rb"), *out = fopen(argv[3], "wbx");
    int rc = 1;
    uint32_t count;
    if (!in || !out || fread(&count, 4, 1, in) != 1 || !count || count > 10000)
        goto done;
    if (fwrite("BNSFEAT1", 1, 8, out) != 8 || fwrite(&count, 4, 1, out) != 1)
        goto done;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bytes;
        if (fread(&bytes, 4, 1, in) != 1 || !bytes || bytes > 1024 * 1024)
            goto done;
        char *text = calloc((size_t)bytes + 1, 1);
        if (!text)
            goto done;
        if (fread(text, 1, bytes, in) != bytes || memchr(text, 0, bytes)) {
            free(text);
            goto done;
        }
        ns_features_t features = {0};
        int status = ns_encode(backbone, text, &features);
        free(text);
        if (status)
            goto done;
        uint32_t n = (uint32_t)features.count;
        if (fwrite(&n, 4, 1, out) != 1) {
            ns_features_clear(&features);
            goto done;
        }
        for (size_t t = 0; t < features.count; t++) {
            uint32_t offsets[2] = {(uint32_t)features.starts[t], (uint32_t)features.ends[t]};
            if (fwrite(offsets, 4, 2, out) != 2) {
                ns_features_clear(&features);
                goto done;
            }
        }
        if (fwrite(features.rows, NS_INPUT * sizeof(float), features.count, out) != features.count) {
            ns_features_clear(&features);
            goto done;
        }
        ns_features_clear(&features);
        if ((i + 1) % 100 == 0)
            fprintf(stderr, "features %u/%u\n", i + 1, count);
    }
    if (fgetc(in) != EOF || ferror(in))
        goto done;
    rc = 0;
done:
    if (in)
        fclose(in);
    if (out && fclose(out))
        rc = 1;
    bitnet_free_model(backbone);
    return rc;
}
