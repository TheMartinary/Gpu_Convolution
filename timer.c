#include <sys/time.h>
#include "timer.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct{
    struct timeval start;
    long int us;
} timer_stack_item_t;

//global timer stack and stack pointer
timer_stack_item_t* stack=NULL;
int sp=0;

FILE * fp;
int i;

void create_file()
{
    fp = fopen ("Performance.txt","w");
}

void writeToFile(const char name[],double time)
{
   //char line[] = ;
   fprintf(fp,"%s %lf\n",name,time,i+1);
   //info ( );
   i++;
   /* close the file*/  
   
}
void close_file()
{
   fclose (fp);
   // printf() displays the string inside quotation
}

void timer_start(){
    //create space on our timer stack
    sp++;
    stack=(timer_stack_item_t*)realloc(stack, sizeof(timer_stack_item_t)*sp);

    //start timing
    if(gettimeofday(&(stack[sp-1].start), NULL))
        error("starting timer\n");
    stack[sp-1].us=-1;
}

long int timer_get(){
    //if the timer has been stopped already, return the stopped time
    if(stack[sp-1].us!=-1)
        return stack[sp-1].us;

    //else get current timing
    struct timeval stop;
    if(gettimeofday(&stop, NULL))
        error("getting current time\n");

    //return us
    return (long int) ((stop.tv_sec- stack[sp-1].start.tv_sec)*1000000)+stop.tv_usec - stack[sp-1].start.tv_usec;
}

//stop timing and return the measured us
long int timer_stop(){
    //stop the timer by storing the current time
    stack[sp-1].us=timer_get();

    //return the measured us
    return stack[sp-1].us;
}


//destroy the timer
void timer_destroy(){
    //sanity check, cannot free more timers than what is allocated on the stack
    if(sp==0)
        return;

    //release space on our timer stack
    sp--;

    //just shrink the stack if there is still something left. O.w. a free is needed (realloc with size==0 is not guaranteed to have the same effect)
    if (sp!=0)
        stack=(timer_stack_item_t*)realloc(stack, sizeof(timer_stack_item_t)*sp);
    else if(stack!=NULL){
        //free the stack
        free(stack);
        stack=NULL;
    }

}
