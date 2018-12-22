#ifndef CONVO_KERNELS
#define CONVO_KERNELS
#include "blob.h"
#include "convolution.h"

void convolve_gpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param);
void convolve_cpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param);

#endif