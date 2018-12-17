
#define TIMER_H
#include "logging.h"

//function to start a new timer
//note that timers can be nested!
void timer_start();
void create_file();
void close_file();

//get current time passed in us
long int timer_get();

//stop timer and get passed time in us
long int timer_stop();

//destroy the current timer object (if nested the previous one will become active)
void timer_destroy();

void writeToFile(const char name[],double time);
//timeit macro to be wrapped around what should be timed
#define timeit_named(name, func) \
    timer_start();\
    func;\
    writeToFile(name,(double)timer_stop());\
    timer_destroy();

    //info("----------%s\nTime taken: %ld us (%lf seconds)\n",name, timer_stop(), (double)timer_stop()/1000000.0 );\
    
//time with empty name
#define timeit(func) timeit_named("", func);




///////////////////////////////////////
//example usage of the 'timeit' macro

//#include <unistd.h> //include for sleep
//timeit(
//    info("Toplevel timer\n");
//    timeit(
//        info("inner timer\n");
//        sleep(1);
//    );
//    sleep(1);
//);

