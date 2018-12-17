#include "eltwise.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

//perform elementwise operation
BLOB* eltwise(BLOB** in_arr, eltwise_param_t* p){

    //copy first input layer
    BLOB* out=blob_duplicate(in_arr[0]);

    //loop over output elements
    for(int i=0;i<blob_size(out);i++)
        //reduce all remaining blobs
        for(int j=1;in_arr[j]!=NULL;j++)
            switch(p->type){

                case EW_SUM:
                    out->data[i]+=in_arr[j]->data[i];
                break;

                case EW_PROD:
                    out->data[i]*=in_arr[j]->data[i];
                break;

                case EW_MAX:
                    out->data[i]=fmax(out->data[i],in_arr[j]->data[i]);
                break;
            }

    //return output
    return out;
}


BLOB* eltwise2(BLOB* in1, BLOB* in2, eltwise_param_t* p){
    BLOB* list[3]={in1, in2, NULL};
    return eltwise(list, p);
}
