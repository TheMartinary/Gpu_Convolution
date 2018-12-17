#include "convolution.h"
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
int calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}
int calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}


__global__ void gpu_device_convolve(float* data_in,float * data_weight, float* data_out,blob_dims* dims,float* depths,int Sx,int Sy,int delta){

    unsigned int out_x = blockIdx.x*blockDim.x + threadIdx.x;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
    
    unsigned int img_width  = gridDim.x*blockDim.x;
    unsigned int img_height = gridDim.y*blockDim.y;

    for(int ky=0;ky<Ky;ky++)
    {
        for(int kx=0;kx<Kx;kx++)
        {
            int in_y = out_y*conv_param->Sy+ky;
            int in_x = out_x*conv_param->Sx+kx;

            int weigth_y = in_depth-(group_id*(in-delta));
            int weight_x = ky*Kx + kx;
            
            int out_id = calc_blob_id(depths[0],out_y,out_x,dims[0].h,dims[0].w);
            int weight_id = calc_blob_id(depths[1],weigth_y,weight_x,dims[1].h,dims[1].w);
            int in_id = calc_blob_id(depths[2],in_y,in_x,dims[2].h,dims[2].w);
              
            data_out[out_id] = data_in[weight_id] * data_weight[weight_id]; 
        }
    }

}
void convolve_gpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
    int numBlocksX=16;
    int numBlocksY=16;

    int threadsPerBlockX=img->w/numBlocksX;  //NOTE: this should have remainder==0 for this code!!
    int threadsPerBlockY=img->h/numBlocksY;  //NOTE: this should have remainder==0 for this code!!

    float* in_data;
    float* out_data;
    float* w_data;
    blob2gpu(in_data, in);
    blob2gpu(out_data, out);
    blob2gpu(w_data, w);


    dim3 grid( 1, 1, 1 );             // numBlocksX x numBlocksY ( x 1)
    dim3 block(threadsPerBlockX, threadsPerBlockY, 3);  // threadsPerBlockX x threadsPerBlockY x 3
    

    for(int group_id=0;group_id<conv_param->group;group_id++)
    {
    int delta = (out->d/conv_param->group);//Depth of output divided by number of groups. 
    int output_starting_depth = group_id*delta;
    for(int out_depth=output_starting_depth;out_depth< output_starting_depth + delta;out_depth++)
        {
        int delta = (in->d/conv_param->group);//Depth of input divided by number of groups. 
        int input_starting_depth = group_id*delta;
        for(int in_depth=input_starting_depth;in_depth<input_starting_depth+delta;in_depth++)
            {
            }
        }
    }          


    gpu_device_convolve<<< grid, block >>>(device_data, device_out);
    
    gpu2blob(in,in_data);
    gpu2blob(out,out_data);
    gpu2blob(w,w_data);


}



void convolve_cpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
    for(int group_id=0;group_id<conv_param->group;group_id++)
            {
            int delta = (out->d/conv_param->group);//Depth of output divided by number of groups. 
            int output_starting_depth = group_id*delta;
            for(int out_depth=output_starting_depth;out_depth< output_starting_depth + delta;out_depth++)
                {
                int delta = (in->d/conv_param->group);//Depth of input divided by number of groups. 
                int input_starting_depth = group_id*delta;
                for(int in_depth=input_starting_depth;in_depth<input_starting_depth+delta;in_depth++)
                    {
                    for(int out_y=0;out_y<out->h;out_y++)
                        for(int out_x=0;out_x<out->w;out_x++)
                            for(int ky=0;ky<Ky;ky++)
                                for(int kx=0;kx<Kx;kx++)
                                {
                                    int in_y = out_y*conv_param->Sy+ky;
                                    int in_x = out_x*conv_param->Sx+kx;

                                    int out_y_ = in_depth-(group_id*(in->d/conv_param->group));
                                    int out_x_ = ky*Kx + kx;

                                    float input = blob_data(in, in_depth, in_y,in_x);
                                    float weight = blob_data(w, out_depth, out_y_, out_x_);
                                    blob_data(out,out_depth,out_y,out_x)+= input*weight; 
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
    BLOB* out;

    //load bias if required
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

    //load weightsint input_id = 
    BLOB* w = load_weights(in, conv_param);
    //float input = blob_data(in, 0, 0, 0);
           
    //perform convolution
    // for(int group_id=0;group_id<conv_param->group;group_id++)
    //     {
    //     int delta = (out->d/conv_param->group);//Depth of output divided by number of groups. 
    //     int output_starting_depth = group_id*delta;
    //     for(int out_depth=output_starting_depth;out_depth< output_starting_depth + delta;out_depth++)
    //         {
    //         int delta = (in->d/conv_param->group);//Depth of input divided by number of groups. 
    //         int input_starting_depth = group_id*delta;
    //         for(int in_depth=input_starting_depth;in_depth<input_starting_depth+delta;in_depth++)
    //             {
    //             for(int out_y=0;out_y<out->h;out_y++)
    //                 for(int out_x=0;out_x<out->w;out_x++)
    //                     for(int ky=0;ky<Ky;ky++)
    //                         for(int kx=0;kx<Kx;kx++)
    //                         {
    //                             int in_y = out_y*conv_param->Sy+ky;
    //                             int in_x = out_x*conv_param->Sx+kx;

    //                             int out_y_ = in_depth-(group_id*(in->d/conv_param->group));
    //                             int out_x_ = ky*Kx + kx;

    //                             float input = blob_data(in, in_depth, in_y,in_x);
    //                             float weight = blob_data(w, out_depth, out_y_, out_x_);
    //                             blob_data(out,out_depth,out_y,out_x)+= input*weight; 
    //                         }
    //             }
    //         }
    //     }          
    convolve_cpu(in,out,w,Kx,Ky,conv_param);


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
