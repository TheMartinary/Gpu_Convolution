#ifndef NETWORK_H
#define NETWORK_H
#include "blob.h"
#include "image_util.h"
#include "convolution.h"
#include "eltwise.h"
#include "pooling.h"

typedef union{
    conv_param_t conv;
    eltwise_param_t eltwise;
    pool_param_t pool;
} layer_param_t;

typedef enum{
    CONVOLUTION,
    ELTWISE,
    POOLING,
    NONE        //not a real layer, used to indicated last layer in list
}layer_type;

typedef struct{
    const char* name;
    layer_type type;
    int input;
    int input2;
    layer_param_t param;
}layer_t;

typedef struct{
    const char* name;
    layer_t* layers; //list network of layers
}Network;

//inference on network
BLOB* network(Network* net, BLOB* input);

#endif
