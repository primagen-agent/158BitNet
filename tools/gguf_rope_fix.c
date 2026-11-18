/* gguf_rope_fix -- inject the LongRoPE factor tensors a MiniCPM GGUF is missing.
 *
 *   gguf_rope_fix <in.gguf> <out.gguf> <short.txt> [long.txt]
 *
 * The official MiniCPM4.1 / BitCPM-CANN 1B and 3B GGUFs declare
 * rope.scaling.type = longrope -- and their HF configs carry 64-entry
 * short/long factor lists -- but the files contain no
 * rope_factors_short.weight / rope_factors_long.weight tensors.  Loading one
 * silently runs plain RoPE at every position, which costs measurable MMLU and
 * ARC accuracy.  This tool rewrites the GGUF with those two F32 tensors
 * appended; every existing tensor stays byte-for-byte where it was, because
 * tensor offsets are relative to the data section and only the tensor-info
 * block grows.
 *
 * <short.txt> is one factor per line (or whitespace separated) -- exactly the
 * rope_scaling.short_factor array in the model's config.json:
 *
 *   python3 -c 'import json;c=json.load(open("config.json"));
 *   print("\n".join(map(str,c["rope_scaling"]["short_factor"])))' > f.txt
 *   ./gguf_rope_fix in.gguf out.gguf f.txt
 *
 * With no <long.txt> the short values are used for both tensors, which is what
 * the MiniCPM4.1 configs specify (short_factor == long_factor).  The runtime
 * picks short or long by context length, as HF does.
 */
/* fileno()/fstat() are POSIX; request them explicitly so the file compiles
 * under a strict ISO C front end and on any libc (glibc, bionic, musl). */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#define GGUF_MAGIC 0x46554747u

static uint32_t rd32(FILE *f)
{
    uint8_t b[4];
    if (fread(b,1,4,f)!=4) { fprintf(stderr,"short read\n"); exit(1); }
    return (uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
}
static uint64_t rd64(FILE *f) { uint64_t lo=rd32(f), hi=rd32(f); return lo|(hi<<32); }
static void wr32(FILE *f, uint32_t v)
{
    uint8_t b[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)};
    fwrite(b,1,4,f);
}
static void wr64(FILE *f, uint64_t v) { wr32(f,(uint32_t)v); wr32(f,(uint32_t)(v>>32)); }

static char *read_string(FILE *f)
{
    uint64_t n=rd64(f);
    char *s=(char *)malloc((size_t)n+1);
    if (!s) { fprintf(stderr,"oom\n"); exit(1); }
    if (fread(s,1,(size_t)n,f)!=(size_t)n) { fprintf(stderr,"short read\n"); exit(1); }
    s[n]=0;
    return s;
}

static void skip_value(FILE *f, uint32_t type)
{
    switch (type) {
    case 0: case 1: case 7: fseek(f,1,SEEK_CUR); break;
    case 2: case 3: fseek(f,2,SEEK_CUR); break;
    case 4: case 5: case 6: fseek(f,4,SEEK_CUR); break;
    case 10: case 11: case 12: fseek(f,8,SEEK_CUR); break;
    case 8: { uint64_t n=rd64(f); fseek(f,(long)n,SEEK_CUR); break; }
    case 9: {
        uint32_t et=rd32(f);
        uint64_t n=rd64(f), i;
        for (i=0;i<n;i++) skip_value(f,et);
        break;
    }
    default: fprintf(stderr,"unknown metadata type %u\n",type); exit(1);
    }
}

static float *read_factors(const char *path, int *n_out)
{
    FILE *f=fopen(path,"r");
    int cap=256, n=0;
    float *v;
    if (!f) { fprintf(stderr,"cannot open %s\n",path); exit(1); }
    v=(float *)malloc((size_t)cap*sizeof(float));
    if (!v) { fprintf(stderr,"oom\n"); exit(1); }
    for (;;) {
        float x;
        if (fscanf(f,"%f",&x)!=1) break;
        if (n==cap) {
            cap*=2;
            v=(float *)realloc(v,(size_t)cap*sizeof(float));
            if (!v) { fprintf(stderr,"oom\n"); exit(1); }
        }
        v[n++]=x;
    }
    fclose(f);
    if (n==0) { fprintf(stderr,"%s has no numbers\n",path); exit(1); }
    *n_out=n;
    return v;
}

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

static void copy_exact(FILE *in, FILE *out, uint64_t len)
{
    char buf[1<<16];
    while (len) {
        size_t c = len > sizeof buf ? sizeof buf : (size_t)len;
        if (fread(buf,1,c,in)!=c) { fprintf(stderr,"short read\n"); exit(1); }
        if (fwrite(buf,1,c,out)!=c) { fprintf(stderr,"short write\n"); exit(1); }
        len -= c;
    }
}

static void write_ti(FILE *out, const char *name, uint64_t count, uint64_t offset)
{
    wr64(out,(uint64_t)strlen(name));
    fwrite(name,1,strlen(name),out);
    wr32(out,1);                 /* n_dims */
    wr64(out,count);             /* dims[0] */
    wr32(out,0);                 /* GGML_TYPE_F32 */
    wr64(out,offset);
}

int main(int argc, char **argv)
{
    FILE *in, *out;
    struct stat st;
    uint32_t magic, version;
    uint64_t n_tensors, n_kv, alignment = 32, i;
    uint64_t pos_kv_end, pos_ti_end, old_data_start, old_data_size;
    uint64_t new_data_start, off_short, off_long, add, cur;
    int nf = 0, nlong = 0, found = 0;
    float *short_f, *long_f;

    if (argc < 4) {
        fprintf(stderr,"usage: %s <in.gguf> <out.gguf> <short.txt> [long.txt]\n", argv[0]);
        return 2;
    }
    short_f = read_factors(argv[3], &nf);
    if (argc >= 5) long_f = read_factors(argv[4], &nlong);
    else { long_f = short_f; nlong = nf; }
    if (nlong != nf) { fprintf(stderr,"factor counts differ (%d vs %d)\n",nf,nlong); return 2; }

    in = fopen(argv[1],"rb");
    if (!in) { fprintf(stderr,"cannot open %s\n",argv[1]); return 1; }
    if (fstat(fileno(in),&st)!=0) { fprintf(stderr,"stat failed\n"); return 1; }

    magic = rd32(in); version = rd32(in); n_tensors = rd64(in); n_kv = rd64(in);
    if (magic != GGUF_MAGIC) { fprintf(stderr,"%s is not a GGUF file\n",argv[1]); return 1; }

    for (i = 0; i < n_kv; ++i) {
        char *k = read_string(in);
        uint32_t t = rd32(in);
        if (t == 4) {
            uint32_t v = rd32(in);
            if (strcmp(k,"general.alignment")==0) alignment = v;
        } else {
            skip_value(in,t);
        }
        free(k);
    }
    pos_kv_end = (uint64_t)ftell(in);

    for (i = 0; i < n_tensors; ++i) {
        char *name = read_string(in);
        uint32_t nd = rd32(in), d;
        if (nd > 8) { fprintf(stderr,"bad n_dims\n"); return 1; }
        for (d = 0; d < nd; ++d) (void)rd64(in);
        (void)rd32(in);
        (void)rd64(in);
        if (strcmp(name,"rope_factors_short.weight")==0 ||
            strcmp(name,"rope_factors_long.weight")==0) found = 1;
        free(name);
    }
    pos_ti_end = (uint64_t)ftell(in);
    if (found) {
        fprintf(stderr,"%s already carries rope factor tensors; nothing to do\n",argv[1]);
        return 0;
    }
    if (alignment < 4 || (alignment & (alignment-1)) != 0) {
        fprintf(stderr,"unsupported alignment %llu\n",(unsigned long long)alignment);
        return 1;
    }

    old_data_start = align_up(pos_ti_end, alignment);
    old_data_size  = (uint64_t)st.st_size - old_data_start;
    add  = 8 + strlen("rope_factors_short.weight") + 4 + 8 + 4 + 8;
    add += 8 + strlen("rope_factors_long.weight")  + 4 + 8 + 4 + 8;
    new_data_start = align_up(pos_ti_end + add, alignment);
    off_short = align_up(old_data_size, alignment);
    off_long  = align_up(off_short + (uint64_t)nf * 4u, alignment);

    out = fopen(argv[2],"wb");
    if (!out) { fprintf(stderr,"cannot create %s\n",argv[2]); return 1; }
    wr32(out,magic); wr32(out,version); wr64(out,n_tensors+2); wr64(out,n_kv);

    fseek(in,24,SEEK_SET);
    copy_exact(in,out,pos_kv_end-24);
    copy_exact(in,out,pos_ti_end-pos_kv_end);
    write_ti(out,"rope_factors_short.weight",(uint64_t)nf,off_short);
    write_ti(out,"rope_factors_long.weight",(uint64_t)nlong,off_long);

    cur = (uint64_t)ftell(out);
    while (cur < new_data_start) { fputc(0,out); ++cur; }

    fseek(in,(long)old_data_start,SEEK_SET);
    copy_exact(in,out,old_data_size);
    cur = new_data_start + old_data_size;
    while (cur < new_data_start + off_short) { fputc(0,out); ++cur; }
    if (fwrite(short_f,sizeof(float),(size_t)nf,out)!=(size_t)nf) { fprintf(stderr,"write failed\n"); return 1; }
    cur = new_data_start + off_short + (uint64_t)nf*4u;
    while (cur < new_data_start + off_long) { fputc(0,out); ++cur; }
    if (fwrite(long_f,sizeof(float),(size_t)nlong,out)!=(size_t)nlong) { fprintf(stderr,"write failed\n"); return 1; }

    fclose(out);
    fclose(in);
    fprintf(stderr,"%s -> %s: added rope_factors_short/long (%d factors)\n", argv[1], argv[2], nf);
    return 0;
}
