#include "pooling.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "logging.h"

//perform  pooling
BLOB* pooling(BLOB* in, pool_param_t* p){

    //do some conversions of parameters in case of global pooling
    int Sy=(!p->global)?p->Sy:in->h;
    int Sx=(!p->global)?p->Sx:in->w;
    int Ky=(!p->global)?p->Ky:in->h;
    int Kx=(!p->global)?p->Kx:in->w;

    //perform pooling
    BLOB* out=blob_alloc(in->d, in->h/Sy, in->w/Sx );

    for(int o=0;o<out->d;o++)
        for(int m=0;m<out->h;m++)
            for(int n=0;n<out->w;n++){

                //init reduction result
                float r=(p->type==POOL_MAX)?blob_data(in,o,m*Sy,n*Sx):0.0f;
                for(int k=0;k<Ky;k++)
                    for(int l=0;l<Kx;l++)
                        switch(p->type){
                            case POOL_AVG:
                               r+=blob_data(in,o,m*Sy+k,n*Sx+l);
                            break;
                            case POOL_MAX:
                               r=fmax(r,blob_data(in,o,m*Sy+k,n*Sx+l));
                            break;
                        }

                //divide by window size for averaging
                if(p->type==POOL_AVG)
                    r/=(Ky*Kx);


                //store result in output blob
                blob_data(out,o,m,n)=r;
            }

    //return pooled result
    return out;
}
