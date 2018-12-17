#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include "blob.h"
#include <stdbool.h>

typedef struct {

    //number of outputs
    int num_out;

    //kernels sizes
    int Ky;
    int Kx;

    //strides
    int Sy;
    int Sx;

    //padding
    int pad;

    //group
    int group;

    //weights
    const char* weights;

    //bias term
    const char* bias;

    //scaling terms
    const char* scale;
    const char* scale_bias;

    //batch norm
    const char* bn_mean;
    const char* bn_var;
    float bn_eps;

    //relu
    bool relu;

    //fully connected
    bool fc;

} conv_param_t;

//perform convolution
BLOB* convolution(BLOB* in, conv_param_t* p);

#endif
