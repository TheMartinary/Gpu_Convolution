#include "convolution.h"
#include "convolution_kernels.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "logging.h"

typedef struct {
    int d;
    int h;
    int w;
} blob_dims;

//add padding to blob
BLOB* pad(BLOB* in, int pad){

    //create output blob
    BLOB* out = blob_calloc(in->d, in->h+2*pad, in->w+pad*2);

    //copy non-padded input into output blob
    for(int z=0;z<in->d;z++)
       for(int y=0;y<in->h;y++)
          for(int x=0;x<in->w;x++)
              blob_data(out,z,y+pad,x+pad)= blob_data(in,z,y,x);

    //return pointer to padded blob
    return out;
}


BLOB* load_weights(BLOB* b, conv_param_t* conv_param){

    //open weights file for reading
    info(conv_param->weights);
    info("\n");
    
    FILE* fp = fopen(conv_param->weights, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\n",conv_param->weights);

    //for fully connected layers the kernel size is equal to the input size
    int Ky=(conv_param->fc)?b->h:conv_param->Ky;
    int Kx=(conv_param->fc)?b->w:conv_param->Kx;

    //allocate 3D blob, and emulate 4D in KxKy later
    BLOB* w = blob_alloc(conv_param->num_out, b->d/conv_param->group, Ky*Kx);

    //fill 4D weight structure
    for(int group_id=0;group_id<conv_param->group;group_id++)
        for(int out_depth=group_id*(conv_param->num_out/conv_param->group);out_depth<(group_id+1)*(conv_param->num_out/conv_param->group);out_depth++)
            for(int i=group_id*(b->d/conv_param->group);i<(group_id+1)*(b->d/conv_param->group);i++)
                //note: each output map has only  b->d/conv_param->group input maps. Hence the absolute index of i is subtracted when storing in w!
                if((int)fread( &(blob_data(w,out_depth,i-group_id*(b->d/conv_param->group),0)),sizeof(float),Ky*Kx, fp)!=Ky*Kx)
                    error("loading weights from file %s\n", conv_param->weights);

    //close file
    fclose(fp);

    //return weight blob
    return w;
}


float* load_1d(const char* fname, size_t num){

    //open file for reading
    FILE* fp = fopen(fname, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\n",fname);

    //read in array
    float* arr= (float*) malloc(sizeof(float)*num);
    if(fread(arr,sizeof(float), num, fp)!=num)
        error("loading data from file %s\n", fname);

    //close file
    fclose(fp);

    return arr;
}






BLOB* initialize_outputBlob(conv_param_t* conv_param,int height,int width)
{
    BLOB* out;
    if(conv_param->bias==NULL){
        //zero init
        out = blob_calloc(conv_param->num_out, height, width);
    }else{
        //not required to calloc
        out = blob_alloc(conv_param->num_out, height, width);

        //load bias values from file
        float* bias =load_1d(conv_param->bias, conv_param->num_out);

        //set bias or init with zeroes
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x)=bias[out_depth];

        //cleanup bias
        free(bias);
    }
    return out;

}

void printArrays(float * arr1,float * arr2)
{
    int i;
    for (i=0;i < sizeof(arr1) / sizeof(float);i++) {
        printf("%lf %lf\n",arr1[i],arr2[i]);
    }

}

void CompareBlobs(BLOB * blob,BLOB * blob2)
{
    for(int z = 0; z<blob->d;z++)
    {
        for(int y = 0; y<blob->h;y++)
        {
            for(int x = 0; x<blob->w;x++)
            {
                float delta = blob_data(blob,z,y,x) - blob_data(blob2,z,y,x) > 0.01;
                if(delta > 0.01 || delta < -0.01)
                {
                    printf("%lf Not Equal to %lf at x : %i , y : %i , z : %i \n"  ,blob_data(blob,z,y,x),blob_data(blob2,z,y,x),x,y,z);   
                    
                    x = y = z=100000;
                    break;
                
                }
            }   
        }   
    }

}


//convolution, NOTE: destructive of BLOB* in. duplicate if further required!
BLOB* convolution(BLOB* input, conv_param_t* conv_param){

    //use local pointer
    BLOB* in = input;

    //padding of input if required
    if(conv_param->pad!=0)
        in = pad(in, conv_param->pad);

    //if fully connected, the kernel size is set to the image size
    int Ky=(conv_param->fc)?in->h:conv_param->Ky;
    int Kx=(conv_param->fc)?in->w:conv_param->Kx;

    //create blob to hold output
    int height=(int)floor(((float)in->h - (float)Ky)/(float)conv_param->Sy)+1;
    int width =(int)floor(((float)in->w - (float)Kx)/(float)conv_param->Sx)+1;
    
    BLOB* out = initialize_outputBlob(conv_param,height,width);
    BLOB* out2 = initialize_outputBlob(conv_param,height,width);
    
    //load weightsint input_id = 
    BLOB* w = load_weights(in, conv_param);
    //convolve_gpu(in,out,w,Kx,Ky,conv_param);
    #ifdef CPU_ONLY
    convolve_cpu(in,out,w,Kx,Ky,conv_param);
    #else
    #ifdef DEBUG
    convolve_cpu(in,out2,w,Kx,Ky,conv_param);
    convolve_gpu(in,out,w,Kx,Ky,conv_param);
    CompareBlobs(out,out2);    
    #else
    convolve_gpu(in,out,w,Kx,Ky,conv_param);
    #endif
    #endif

    //free weights
    blob_free(w);
    //done with padded blob, free
    if(conv_param->pad!=0)
        blob_free(in);
    //perform batchnorm if needed
    if(conv_param->bn_mean!=NULL){
        //load batchnorm mean and variance
        float* mean = load_1d(conv_param->bn_mean, out->d);
        float* var  = load_1d(conv_param->bn_var, out->d);

        //batchnorm
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x)= (blob_data(out,out_depth,out_y,out_x) - mean[out_depth])/sqrtf(var[out_depth]+conv_param->bn_eps);

        //free mean and variance
        free(mean);
        free(var);
    }

    //perform scale if needed
    if(conv_param->scale!=NULL){
        //load scale parameters
        float* scale = load_1d(conv_param->scale, out->d);
        float* scale_bias = load_1d(conv_param->scale_bias, out->d);

        //scale
        for(int out_depth=0;out_depth<out->d;out_depth++)
            for(int out_y=0;out_y<out->h;out_y++)
                for(int out_x=0;out_x<out->w;out_x++)
                    blob_data(out,out_depth,out_y,out_x) = blob_data(out,out_depth,out_y,out_x)*scale[out_depth] + scale_bias[out_depth];

        //free parameters
        free(scale);
        free(scale_bias);
    }

    //perform relu
    if(conv_param->relu==true)
        for(int i=0;i<blob_size(out); i++)
            out->data[i] =  fmax(0.0f, out->data[i]);

    //return output
    return out;
}
