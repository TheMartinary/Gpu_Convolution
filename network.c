#include "network.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#include "timer.h"

#ifndef SILENT
//mapping of layer_type enum to strings
static const char* layer_type_str[4] ={
    "convolution",
    "element wise",
    "pooling",
    "none",
};
#endif //SILENT

BLOB* network(Network* net, BLOB* input){

    //get number of layers in the network
    int num_layers=0;
    while(net->layers[num_layers].type!=NONE) num_layers++;

    //create placeholder for all layer blobs
    BLOB** layer_blobs=(BLOB**)malloc(sizeof(BLOB*)*(num_layers+1));

    //insert input layer
    layer_blobs[0]=input;

    //iterate through the layers in the network
    for(int l=0;l<num_layers;l++){

        //select layer "l" from the network
        layer_t layer=net->layers[l];

        //blobl pointer to hold output of this layer
        BLOB* out=NULL;

        //wrapping the whole layer evaluation in a named timer object
        timeit_named(layer.name,

            
            //depending on type the actions differ
            switch(layer.type){
                case CONVOLUTION:
                    out = convolution(
                        layer_blobs[layer.input],
                        &(layer.param.conv)
                    );
                    
                break;

                case ELTWISE:
                    out = eltwise2(
                        layer_blobs[layer.input],
                        layer_blobs[layer.input2],
                        &(layer.param.eltwise)
                    );

                break;

                case POOLING:
                    out = pooling(
                        layer_blobs[layer.input],
                        &(layer.param.pool)
                    );
                break;

                case NONE:
                    //last layer in the list - Note: should never reach here
                    out=NULL;
                break;
            }

        )
        
        //end of timer wrapper

        //store out in the blob holding structure to be used by future layers
        //NOTE: currently all blobs are stored until the network is fully evaluated,
        //with smarter management this storage could be reduced significantly
        if(out!=NULL)
            layer_blobs[l+1]=out;
        //break;
        #ifdef DEBUG
        if(out){
            // Some example on how to use the blob IO routines
            // Can be very useful to debug problems in the output of your network

            //create a name for the output file
            char* fname = (char*) malloc(sizeof(char)*strlen(layer.name)+5);
            sprintf(fname, "%s.txt", layer.name);

            //write to file
            blob_write_txt(fname, out);                 //text version
            //blob_write_bin(fname, layer_blobs[l+1]);  //binary version

            //read from file
            //out=blob_read_txt(fname);  //text version
            //out=blob_read_bin(fname);  //binary version

            //free the file name
            free(fname);
        }
        #endif //DEBUG

    }
    //done evaluating the network

    //extract output blob pointer
    BLOB* out = layer_blobs[num_layers];

    //clean up all intermediate blobs
    for(int i=1;i<num_layers;i++)
        blob_free(layer_blobs[i]);

    //clean up layer blob structure (note: only structure, data is managed by caller)
    free(layer_blobs);

    //return the output blob
    return out;
}
