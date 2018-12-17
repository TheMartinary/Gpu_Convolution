#include "preprocessing.h"
#include "logging.h"

void cpu_preprocess(BLOB* img){
    //Subtract mean RGB values, scale with 0.017, and swap RGB->BGR
    for(int y=0;y<img->h;y++)
        for(int x=0;x<img->w;x++){
            float R              =  (blob_data(img,0,y,x)-123.680f)*0.017f; //R
            blob_data(img, 1,y,x) = (blob_data(img,1,y,x)-116.779f)*0.017f; //G
            blob_data(img, 0,y,x) = (blob_data(img,2,y,x)-103.939f)*0.017f; //B
            blob_data(img, 2,y,x) = R;
        }
}


__global__ void gpu_device_preprocess(float* data_in, float* data_out){

    unsigned int global_x = blockIdx.x*blockDim.x + threadIdx.x;  
    unsigned int global_y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int global_z = blockIdx.z*blockDim.z + threadIdx.z; //NOTE: gridDim.z==1 and thus blockIdx.z==0 in this example!

    unsigned int img_width  = gridDim.x*blockDim.x;
    unsigned int img_height = gridDim.y*blockDim.y;

       float value = data_in[ global_z*img_height*img_width + global_y*img_width + global_x];
    //each channel (Z) needs to correct with a different mean value
    float mean[3]={
        123.680f,
        116.779f,
        103.939f
    };

    //correct by subtracting the correct mean for this channel and scaling by a factor 0.017
    value= (value-mean[global_z]) * 0.017f;
    // swap rgb
    data_out[(2-global_z)*img_width*img_height + global_y*img_width + global_x]=value;
}

//GPU host code (called from the CPU, copies data back and forth and launched the GPU thread)
void gpu_preprocess(BLOB* img){

   
    //let's first divide the X and Y dimensions of the image into a number of blocks here
    int numBlocksX=16;
    int numBlocksY=16;

    int threadsPerBlockX=img->w/numBlocksX;  //NOTE: this should have remainder==0 for this code!!
    int threadsPerBlockY=img->h/numBlocksY;  //NOTE: this should have remainder==0 for this code!!

    dim3 grid( numBlocksX, numBlocksY, 1 );             // numBlocksX x numBlocksY ( x 1)
    dim3 block(threadsPerBlockX, threadsPerBlockY, 3);  // threadsPerBlockX x threadsPerBlockY x 3

    //Now that we have decided on the grid and block dimensions, it's time to copy our
    //image data over from the CPU to the GPUs global memory

    //First create a pointer to data on the GPU
    float* device_data;
#ifndef SHORTHANDS
    //This variable holds return values of cuda functions, which can be very useful for error checking
    cudaError_t err;

    //malloc space on the on the GPU
    err=cudaMalloc(&device_data, blob_bytes(img));

    //check for errors (NOTE: this is not a standard cuda function. Check logging.h)
    cudaCheckError(err)

    //copy the image data over to the GPU
    cudaCheckError(cudaMemcpy(device_data, img->data, blob_bytes(img), cudaMemcpyHostToDevice));
#else
    //For your convenience a helper function is defined in blob.h which can take care of the allocation and memcpy of blobs
    blob2gpu(device_data, img);
#endif

    //next we also allocate a buffer that will hold the output
    float* device_out;
    cudaCheckError(cudaMalloc(&device_out, blob_bytes(img)));

    //Perform the preprocessing on the GPU
    info("Preprocessing on GPU...\n");
    gpu_device_preprocess<<< grid, block >>>(device_data, device_out);

    //We use "peekatlasterror" since a kernel launch does not return a cudaError_t to check for errors
    cudaCheckError(cudaPeekAtLastError());

#ifndef SHORTHANDS
    //copy the processed image data back from GPU global memory to CPU memory
    cudaCheckError(cudaMemcpy(img->data, device_out, blob_bytes(img), cudaMemcpyDeviceToHost));

    //free the allocated GPU memory that holds the output
    cudaCheckError(cudaFree(device_out));
#else
    //again a simple shorthand to transfer a blob back from the gpu to the cpu and free the allocated memory
    gpu2blob(img, device_data);
#endif

    //finally we also need to release the space that holds the input on the GPU
    cudaCheckError(cudaFree(device_data));

}
