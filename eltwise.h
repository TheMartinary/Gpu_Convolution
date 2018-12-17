#ifndef ELTWISE_H
#define ELTWISE_H
#include "blob.h"

typedef enum{
    EW_MAX,
    EW_SUM,
    EW_PROD
}eltwise_operation_t;

typedef struct{
    eltwise_operation_t type;
}eltwise_param_t;

//perform elementwise operation
BLOB* eltwise(BLOB** in_arr, eltwise_param_t* p);

//shorthand for two input elementwise operation
BLOB* eltwise2(BLOB* in1, BLOB* in2, eltwise_param_t* p);

#endif
