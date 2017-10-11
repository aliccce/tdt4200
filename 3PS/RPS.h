#ifndef RPS_H
#define RPS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
  int color;
  int strength;
} cell;

#include "CA.h"
#include "bitmap.h"

#define TRANS(x, y) ((x)+(y)*IMG_X)

#define ITERATIONS 3000
#define BORDER_SIZE 1

#define WHITE   0
#define ROCK    1
#define PAPER   2
#define SCISSOR 3

#define IMG_X 1024
#define IMG_Y 1024

#endif
