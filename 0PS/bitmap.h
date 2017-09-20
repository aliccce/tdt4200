#ifndef BITMAP_H
#define BITMAP_H

#include "julia.h"

typedef unsigned char uchar;
void savebmp(char *name,uchar *buffer,int x,int y);
void fancycolour(uchar *p,int iter);

#endif
