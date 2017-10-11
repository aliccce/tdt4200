#include "CA.h"

#define TRANS(x, y) ((x)+(y)*IMG_X)

cell pick_neighbor(int x, int y, cell* image, int chosen);

cell** alloc_img(cell* buffer, int index){
  cell** image = malloc(IMG_X*sizeof(cell*));

  for(int ii = 0; ii < IMG_X; ii++){
    image[ii] = &buffer[(IMG_X*IMG_Y*index) + IMG_X*ii];
  }

  return image;
}

void free_img(cell** image){
  for (int ii = 0; ii < IMG_X; ii++){
    free(image[ii]);
  }
  free(image);
}


bool beats(cell me, cell other){
  return
    (((me.color == SCISSOR) && (other.color == PAPER)) ||
     ((me.color == PAPER) && (other.color == ROCK))    ||
     ((me.color == ROCK) && (other.color == SCISSOR))  ||
     (me.color == other.color)) && ((me.color == SCISSOR) || (me.color == PAPER) || (me.color == ROCK));
}

cell next_cell(int x, int y, cell* image, int chosen){

  cell neighbor_cell = pick_neighbor(x, y, image, chosen);
  cell my_cell = image[TRANS(x,y)];
  if(neighbor_cell.color == WHITE){
    return my_cell;
  }
  cell next_cell = my_cell;

  if(my_cell.color == WHITE){
    next_cell.strength = 1;
    next_cell.color = neighbor_cell.color;
    return next_cell;
  }
  else {
    if(beats(my_cell, neighbor_cell)){
      next_cell.strength++;
    }
    else{
      next_cell.strength--;
    }
  }

  if(next_cell.strength == 0){
    next_cell.color = neighbor_cell.color;
    next_cell.strength = 1;
  }

  if(next_cell.strength > 4){
    next_cell.strength = 4;
  }

  return next_cell;
}


cell pick_neighbor(int x, int y, cell* image, int chosen){

  if(chosen >= 4){ chosen++; } // a cell cant choose itself
  int c_x = chosen % 3;
  int c_y = chosen / 3;

  return image[TRANS(x + c_x - 1,y + c_y - 1)];
}

void iterate_image(cell* old_image, cell* next_image){

  int seed = rand();
  seed = (seed * 0x5DEECE66DL + 0xBL) & 0xFFFFFFFFFFFFL;

  for(int xx = 1; xx < IMG_X - 2; xx++){
    for(int yy = 1; yy < IMG_Y - 2; yy++){
      next_image[TRANS(xx,yy)] = next_cell(xx, yy, old_image, (seed % 8) + 8*(seed < 8));
    }
  }
}
