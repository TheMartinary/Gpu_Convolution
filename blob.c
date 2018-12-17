#include <stdlib.h>
#include <string.h>
#include "blob.h"
#include <stdio.h>
#include "logging.h"


typedef struct {
    int d;
    int h;
    int w;
} bin_blob_hdr;


//generic blob allocation which can be called with malloc or calloc as allocation function
BLOB* __blob_alloc(void*(*alloc)(size_t size), int d, int h, int w){
    BLOB* b=(BLOB*)(*alloc)(sizeof(BLOB));
    b->w=w;
    b->h=h;
    b->d=d;
    b->data=(float*) (*alloc)(blob_bytes(b));
    return b;
};

//custom calloc which matches the "malloc" signature
void* my_calloc(size_t size){
    return calloc(1, size);
}

//allocate blob with given dimensions
BLOB* blob_alloc(int d, int h, int w){
    return __blob_alloc(&malloc,d,h,w);
}

//calloc blob with given dimensions (zero init)
BLOB* blob_calloc(int d, int h, int w){
    return __blob_alloc(&my_calloc,d,h,w);
}

//duplicate blob
BLOB* blob_duplicate(BLOB* b){
    BLOB* out = blob_alloc(b->d, b->h, b->w);
    memcpy(out->data, b->data, blob_bytes(b));
    return out;
}

//write blob to text file
void blob_write_txt(const char* fname, BLOB* b){

    //open file for writing
    FILE* fp = fopen(fname, "wt");
    if(fp==NULL)
        error("unable to open file %s for writing\n", fname);

    //write out dimensions
    fprintf(fp,"%d,%d,%d\n",b->d,b->h,b->w);

    //write out all data, one line per element
    for(int i=0;i<blob_size(b);i++)
        fprintf(fp,"%f\n",b->data[i]);

    //close the file
    fclose(fp);
}

//write blob to binary file
void blob_write_bin(const char* fname, BLOB* b){

    //open file for writing
    FILE* fp = fopen(fname, "wb");
    if(fp==NULL)
        error("unable to open file %s for writing\n", fname);

    //create blob header
    bin_blob_hdr hdr;
    hdr.d=b->d;
    hdr.h=b->h;
    hdr.w=b->w;

    //write out header
    fwrite(&hdr, sizeof(bin_blob_hdr), 1, fp);

    //write out raw float data
    if((int)fwrite(b->data, sizeof(float), blob_size(b), fp)!=blob_size(b))
        error("writing to file %s\n",fname);

    //close the file
    fclose(fp);
}

//load blob from text file
BLOB* blob_read_txt(const char* fname){

    //open file for reading
    FILE* fp = fopen(fname, "rt");
    if(fp==NULL)
        error("unable to open file %s for reading\n", fname);

    //get dimensions
    int d,h,w;
    if(fscanf(fp,"%d,%d,%d\n",&d,&h,&w)!=3)
        error("parsing header from file %s\n", fname);

    //allocate BLOB
    BLOB* b = blob_alloc(d,h,w);

    //fill BLOB with data
    for(int i=0;i<blob_size(b);i++)
        if(fscanf(fp,"%f\n",&(b->data[i]))!=1)
            error("reading float value from %s\n", fname);

    //close the file
    fclose(fp);

    //return the parsed blob
    return b;
}

//load blob from binary file
BLOB* blob_read_bin(const char* fname){

    //open file for reading
    FILE* fp = fopen(fname, "rb");
    if(fp==NULL)
        error("unable to open file %s for reading\n", fname);

    //create blob header
    bin_blob_hdr hdr;

    //read out header
    if(fread(&hdr, sizeof(bin_blob_hdr), 1, fp)!=1)
        error("reading header from file %s\n", fname);

    //allocate blob
    BLOB* b = blob_alloc(hdr.d, hdr.h, hdr.w);

    //read out raw float data
    if((int)fread(b->data, sizeof(float), blob_size(b), fp)!=blob_size(b))
        error("reading float data from %s\n",fname);

    //close the file
    fclose(fp);

    //return the blob
    return b;
}
