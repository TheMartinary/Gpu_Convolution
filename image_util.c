/*
 * Modified by Luc Waeijen - 2018
 *
 * Modified by Yoshimasa Niwa to make it much simpler
 * and support all defined color_type.
 *
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "image_util.h"
#include <stdlib.h>
#include <png.h>
#include "logging.h"

BLOB* read_png(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if(fp==NULL) error("Could not open %s for reading\n", filename);

  //create png struct
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) error("Could not read png file %s\n",filename);

  //create png info
  png_infop info = png_create_info_struct(png);
  if(!info) error("Problem creating png info structure\n");

  if(setjmp(png_jmpbuf(png))) error("Problem with 'setjmp'");

  //read the png data
  png_init_io(png, fp);

  //read the info
  png_read_info(png, info);


  //Image struct
  BLOB* img = (BLOB*) malloc(sizeof(BLOB));

  //set width and height
  img->w   = png_get_image_width(png, info);
  img->h   = png_get_image_height(png, info);
  img->d   = 3; //fixed for this function to 3

  // Read any color_type into 8bit depth, RGBA format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt
  png_byte color_type = png_get_color_type(png, info);
  png_byte bit_depth  = png_get_bit_depth(png, info);

  if(bit_depth == 16)
    png_set_strip_16(png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  // These color_type don't have an alpha channel then fill it with 0xff.
  if(color_type == PNG_COLOR_TYPE_RGB ||
     color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

  if(color_type == PNG_COLOR_TYPE_GRAY ||
     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png);

  png_read_update_info(png, info);

  //allocate space for png bytes
  png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->h);
  for(int y = 0; y < img->h; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }

  //finally read png data
  png_read_image(png, row_pointers);

  //close image
  fclose(fp);

  //allocate space to hold data (now that the dimensions are initialized blob_bytes will work)
  img->data=(float*)malloc(blob_bytes(img));

  //convert to elements to floats [0f-255f], and reorder to [chan][rows][col] dropping the alpha channel
  for(int c=0; c<img->d;c++)
    for(int y=0; y<img->h;y++)
      for(int x=0;x<img->w;x++)
        blob_data(img,c,y,x)=(float)row_pointers[y][x*4+c];


  //clean up original data structure
  for(int y=0;y<img->h;y++)
     free(row_pointers[y]);
  free(row_pointers);


  //clean up png read structure
  png_destroy_read_struct(&png, &info, NULL);

  return img;
}

bool write_png(const char *filename, BLOB* img) {

  FILE *fp = fopen(filename, "wb");
  if(!fp) error("Error opening file %s for writing\n", filename);

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) error("Error creating png write structure\n");

  png_infop info = png_create_info_struct(png);
  if (!info) error("Error creating png info structure\n");

  if (setjmp(png_jmpbuf(png))) error("Error with setjmp function\n");

  //init the io
  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    img->w, img->h,
    8,
    PNG_COLOR_TYPE_RGBA,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  //write info
  png_write_info(png, info);

  // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
  // Use png_set_filler().
  //png_set_filler(png, 0, PNG_FILLER_AFTER);


  //allocate space for png bytes
  png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->h);
  for(int y = 0; y < img->h; y++) {
    row_pointers[y] = (png_byte*)malloc(sizeof(png_byte)*4*img->w);
  }

  //convert and reorder image float data
  for(int y=0; y<img->h;y++){
    for(int x=0;x<img->w;x++){
        for(int c=0;c<3;c++){
            //copy over RGB channels
            row_pointers[y][x*4+c]=(png_byte) blob_data(img,c,y,x);
        }
        //add alpha channel back in there
        row_pointers[y][x*4+3]=(png_byte)0xFF;
    }
  }

  //write data to file
  png_write_image(png, row_pointers);
  png_write_end(png, NULL);

  //close the file
  fclose(fp);

  //clean up byte data structure
  for(int y = 0; y < img->h; y++) {
    free(row_pointers[y]);
  }
  free(row_pointers);

  //clean up png write structures
  png_destroy_write_struct(&png, &info);

  return true;
}
