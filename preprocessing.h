#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include "blob.h"

//do processing on the cpu
void cpu_preprocess(BLOB* img);

//do processing on the gpu
void gpu_preprocess(BLOB* img);

#endif
