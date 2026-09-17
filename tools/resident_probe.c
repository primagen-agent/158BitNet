/* Export fresh C backbone features for offline parity diagnostics only.
 * Each input line is encoded independently; no prior context is reused. */
#include "metis/resident_identity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if(argc!=4)return 2;
    bitnet_model_t *model=bitnet_load_model(argv[1]);
    if(!model)return 3;
    FILE *input=fopen(argv[2],"rb"),*output=fopen(argv[3],"wb");
    if(!input||!output)return 4;
    char text[4096];int rc=0;
    while(fgets(text,sizeof text,input)) {
        size_t length=strlen(text);
        if(!length||text[length-1]!='\n'){rc=5;break;}
        text[length-1]='\0';size_t tokens=0;
        float *features=resident_encode(model,text,&tokens);uint32_t n=(uint32_t)tokens;
        if(!features||fwrite(&n,4,1,output)!=1||
           fwrite(features,2048*sizeof(float),tokens,output)!=tokens){free(features);rc=6;break;}
        free(features);
    }
    if(ferror(input))rc=7;
    fclose(input);if(fclose(output))rc=8;bitnet_free_model(model);return rc;
}
