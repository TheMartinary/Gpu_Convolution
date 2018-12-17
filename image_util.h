#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H
#include <stdio.h>
#include <stdbool.h>
#include "blob.h"


BLOB* read_png(const char *filename);
bool write_png(const char *filename, BLOB* img);

#endif
