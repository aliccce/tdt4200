#ifndef CA_H
#define CA_H

#include "RPS.h"


void free_img(cell** image);
bool neighborhood_contains(int x, int y, cell** image, int color);
int next_color(int x, int y, cell** image);
void iterate_image(cell* old_image, cell* next_image);
void iterate_image_pth(cell* old_image, cell* next_image, int y_start, int y_end);

cell** alloc_img();
cell next_cell(int x, int y, cell* image, int chosen);

#endif
