#ifndef POOLING_H
#define POOLING_H
#include "blob.h"
#include <stdbool.h>

typedef enum{
    POOL_MAX,
    POOL_AVG,
}pool_operation_t;

typedef struct{
    pool_operation_t type;
    bool global;
    int Kx;
    int Ky;
    int Sx;
    int Sy;
}pool_param_t;

//perform pooling
BLOB* pooling(BLOB* in, pool_param_t* p);


#endif
