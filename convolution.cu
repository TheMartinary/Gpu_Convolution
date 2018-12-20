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
blob_dims* get_gpu_blob_dims(BLOB* out,BLOB* weight,BLOB*in)
{
    int numblob_dims    = 3,
        blob_dim_size    = sizeof(blob_dims),
        numBytes     = numblob_dims * blob_dim_size;
        // allocate memory
        blob_dims *cpu_blob_dim_arr,*gpu_blob_dim_arr;
        cpu_blob_dim_arr = (blob_dims*)malloc(numBytes);
    
    cudaMalloc((void**)&gpu_blob_dim_arr, numBytes);
    // 0 = out
    // 1 = weight
    // 2 = in
    cpu_blob_dim_arr[0].d = out->d;
    cpu_blob_dim_arr[0].w = out->w;
    cpu_blob_dim_arr[0].h = out->h;

    cpu_blob_dim_arr[1].d = weight->d;
    cpu_blob_dim_arr[1].w = weight->w;
    cpu_blob_dim_arr[1].h = weight->h;

    cpu_blob_dim_arr[2].d = in->d;
    cpu_blob_dim_arr[2].w = in->w;
    cpu_blob_dim_arr[2].h = in->h;
    cudaMemcpy(gpu_blob_dim_arr,cpu_blob_dim_arr,numBytes,cudaMemcpyHostToDevice);
    free(cpu_blob_dim_arr);
    return gpu_blob_dim_arr;
}

__device__ int calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}

int cpu_calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}


__global__ void gpu_device_convolve
    (float* data_in,float * data_weight, float* data_out // Data
    ,int Sx,int Sy // Sizes ...
    ,int in_w,int in_h,int in_d, // input blob dimensions
    int w_w,int w_h, // weigth height and depth
    int out_w,int out_h, // 
    int Ky,int Kx
    ,int in_depth,int out_depth,
    int group_id,int group){

    unsigned int out_x = blockIdx.x*blockDim.x + threadIdx.x;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
    if(out_x < out_w && out_y < out_h)//dims[0].w && out_y<dims[0].h)
    {
        int out_id = calc_blob_id(out_depth,out_y,out_x,out_h,out_w);
        for(int ky=0;ky<Ky;ky++)
        {
            for(int kx=0;kx<Kx;kx++)
            {
                int in_y = out_y*Sy+ky;
                int in_x = out_x*Sx+kx;

                int weigth_y = in_depth-(group_id*(in_d/group));
                int weight_x = ky*Kx + kx;
                
                int weight_id = calc_blob_id(out_depth,weigth_y,weight_x,w_h,w_w);
                int in_id = calc_blob_id(in_depth,in_y,in_x,in_h,in_w);
   
                data_out[out_id] += data_weight[weight_id] * data_in[in_id]; 
            }
        }
    }
}

__global__ void out_test(float* data_out,int out_depth,blob_dims* dims)
{
    unsigned int out_x = blockIdx.x*blockDim.x + threadIdx.x;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int width = gridDim.x*blockDim.x;
    unsigned int height = gridDim.y*blockDim.y;

    // if(out_x < width && out_y<height)
    // {
        int out_id = calc_blob_id(out_depth,out_y,out_x,height,width);
        data_out[out_id] = 5;    
    //}
}



void convolve_gpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
    int numBlocksX=4;
    int numBlocksY=4;

    int threadsPerBlockX=out->w/numBlocksX+1;///numBlocksX;  //NOTE: this should have remainder==0 for this code!!
    int threadsPerBlockY=out->h/numBlocksY+1;//numBlocksY;  //NOTE: this should have remainder==0 for this code!!

    printf("Width : %i ",out->w);
    printf(", Height : %i \n",out->h);

    float* in_data;
    float* out_data;
    
    float* w_data;
    
    blob2gpu(in_data, in);
    int last_id = out->w*out->h*out->d-1;
    blob2gpu(out_data, out);
    blob2gpu(w_data, w);
    blob_dims *gpu_blob_dim_arr = get_gpu_blob_dims(in,w,out);
    

    dim3 grid( numBlocksX, numBlocksY, 1 );             // numBlocksX x numBlocksY ( x 1)
    dim3 block(threadsPerBlockX, threadsPerBlockY, 1);  // threadsPerBlockX x threadsPerBlockY x 3
    

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
//                out_test<<<grid,block>>>(out_data,out_depth,gpu_blob_dim_arr);

                //printf("OutDepth : %i \n",out_depth );
                gpu_device_convolve<<<grid,block>>>(
                      in_data,w_data,out_data
                     ,conv_param->Sx,conv_param->Sy
                     ,in->w,in->h,in->d
                     ,w->w,w->h
                     ,out->w,out->h
                     ,Ky,Kx
                     ,in_depth,out_depth
                     ,group_id,conv_param->group);
            }
        }
    }          

    cudaFree(gpu_blob_dim_arr);
     
 //   gpu2blob(in,in_data);
    gpu2blob(out,out_data);
    // gpu2blob(w,w_data);
    // cudaFree(in_data);
    cudaFree(out_data);
    // cudaFree(w_data);
//    printf("test_first : %f \n",out->data[0]);
//    printf("test_last : %f \n",out->data[last_id]);
    // cudaCheckError();
    // cudaCheckError(cudaFree(out_data));
    // cudaCheckError(cudaFree(w_data));
    


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

                                    int weigth_y = in_depth-(group_id*(in->d/conv_param->group));
                                    int weigth_x = ky*Kx + kx;

                                    float input = blob_data(in, in_depth, in_y,in_x);
                                    float weight = blob_data(w, out_depth, weigth_y, weigth_x);
                                    out->data[cpu_calc_blob_id(out_depth,out_y,out_x, out->h,out->w)] += input*weight;
                                    
//                                    blob_data(out,out_depth,out_y,out_x)+= input*weight; 
                                }
                    }
                }
            }          


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
                
                if(blob_data(blob,z,y,x) != blob_data(blob2,z,y,x))
                {
                    printf("%lf Not Equal to %lf at x : %i , y : %i , z : %i \n"  ,blob_data(blob,z,y,x),blob_data(blob2,z,y,x),x,y,z);   
                    x = y = z=10000000000;
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
  //  BLOB* out2 = initialize_outputBlob(conv_param,height,width);
    
    //load weightsint input_id = 
    BLOB* w = load_weights(in, conv_param);
    //convolve_gpu(in,out,w,Kx,Ky,conv_param);

    //convolve_cpu(in,out,w,Kx,Ky,conv_param);
    convolve_gpu(in,out,w,Kx,Ky,conv_param);
    //CompareBlobs(out,out2);

//    printArrays(out->data,out2->data);

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
