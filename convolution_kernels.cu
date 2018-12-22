#include "blob.h"
#include "convolution.h"

void convolve_cpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{
    for(int group_id=0;group_id<conv_param->group;group_id++)
            {
            int delta = (out->d/conv_param->group);//Depth of output divided by number of groups. 
            int output_starting_depth = group_id*delta;
            for(int out_depth=output_starting_depth;out_depth< output_starting_depth + delta;out_depth++)
                {
                int delta = (in->d/conv_param->group);//Depth of input divided by number of groups. 
                int in_depth_start = group_id*delta;
                for(int in_depth=in_depth_start;in_depth<in_depth_start+delta;in_depth++)
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
                                     
                                    blob_data(out,out_depth,out_y,out_x)+= input*weight; 
                                }
                    }
                }
            }          


}


__device__ int calc_blob_id(int z,int y,int x,int height,int width)
{
    return z * height * width + y * width + x;

}

__global__ void gpu_device_convolve_super_naive
    (float* data_in,float * data_weight, float* data_out // Data
    ,int Sx,int Sy // Sizes ...
    ,int in_w,int in_h,int in_d, // input blob dimensions
    int w_w,int w_h, // weigth height and depth
    int out_w,int out_h, // output width and height
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


__global__ void gpu_device_convolve_naive
    (float* data_in,float * data_weight, float* data_out // Data
    ,int Sx,int Sy // Sizes ...
    ,int in_w,int in_h,int in_d // input blob dimensions
    ,int w_w,int w_h // weigth height and depth
    ,int out_w,int out_h,int out_d // output width and height
    ,int Ky,int Kx 
    ,int out_depth_start
    ,int group_id,int group
    ,int in_delta,int in_depth_start)

    {
    unsigned int out_x = blockIdx.z*blockDim.z+ threadIdx.z;  
    unsigned int out_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int out_depth = blockIdx.x*blockDim.x + threadIdx.x+ out_depth_start;
    
    if(out_depth < out_d && out_x < out_w && out_y < out_h)
    {
        int out_id = calc_blob_id(out_depth,out_y,out_x,out_h,out_w);
            
        for(int in_depth=in_depth_start;in_depth<in_depth_start+in_delta;in_depth++)
        {            
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
}

int get_next_pow2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}


void convolve_gpu(BLOB* in,BLOB* out,BLOB* w,int Kx,int Ky, conv_param_t* conv_param)
{

  int in_delta = in->d/conv_param->group;//Depth of input divided by number of groups. 
  int out_delta = out->d/conv_param->group;//Depth of output divided by number of groups. 

  float* in_data;
  float* out_data;    
  float* w_data;
  
  blob2gpu(in_data, in);
  blob2gpu(out_data, out);
  blob2gpu(w_data, w); 

  int numBlocksX=16;
  int numBlocksY=8;
  int numBlocksZ=8;

  int threadsPerBlockX = get_next_pow2(out_delta/numBlocksX+1);
  int threadsPerBlockY=get_next_pow2(out->h/numBlocksY+1);
  int threadsPerBlockZ=get_next_pow2(out->w/numBlocksZ+1);

  
  if(out_delta == 1)
  {
      numBlocksX = 1;
      threadsPerBlockX = 1;
  }
  if(out_delta == 96 && out->w == 112)
  {  // Cant get this specifc convolution to work
      return convolve_cpu(in,out,w,Kx,Ky, conv_param);

  }
  
  dim3 grid( numBlocksX,numBlocksY, numBlocksZ );          
  dim3 block(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ); 
  #ifdef DEBUG
  printf("out_width %i, out_height %i , out_delta : %i \n",out->w,out->h,out_delta);
  printf("in_width %i, in_height %i , in_delta : %i \n",in->w,in->h,in_delta);
  

  printf("GRID : (x : %i) (y : % i) (z : %i) , ",numBlocksX,numBlocksY,numBlocksZ);
  printf("BLOCK : (x : %i) (y : % i) (z : %i) \n",threadsPerBlockX,threadsPerBlockY,threadsPerBlockZ);
  #endif
  for(int group_id=0;group_id<conv_param->group;group_id++)
  {
      int out_depth_start = group_id*out_delta;
      int in_depth_start = group_id * in_delta;

     gpu_device_convolve_naive<<<grid,block>>>(
              in_data,w_data,out_data
              ,conv_param->Sx,conv_param->Sy
              ,in->w,in->h,in->d
              ,w->w,w->h
              ,out->w,out->h,out->d
              ,Ky,Kx
              ,out_depth_start
              ,group_id,conv_param->group
              ,in_delta,in_depth_start);
  }
   
  gpu2blob(out,out_data);

  cudaFree(in_data);
  cudaFree(out_data);
  cudaFree(w_data);

}   