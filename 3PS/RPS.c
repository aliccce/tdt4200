#include "RPS.h"
#include <time.h>
#include <omp.h>
#include <stdlib.h>

void swap_petris();

cell* petri_A;
cell* petri_B;

#define ARGUMENTS(x) (x > -1)

int main(int argc, char** argv){

  NUM_OF_THREADS = -1;
  if (argc > 1) {
      NUM_OF_THREADS = strtol( argv[1], NULL, 10 );
  }
  else {
      printf("Argument missing: This program takes one argument. Please specify a number of threads.\n");
      exit(0);
  }
  printf("running %d iterations with %d threads. \n",ITERATIONS, NUM_OF_THREADS);


  srand(time(NULL));
  petri_A = calloc(IMG_X*IMG_Y, sizeof(cell));
  petri_B = calloc(IMG_X*IMG_Y, sizeof(cell));

  int seed = rand();

  // Seed some CAs
  for(int ii = 0; ii < 100; ii++){
    int rx = rand() % (IMG_X - 1);
    int ry = rand() % (IMG_Y - 1);
    int rt = rand() % 4;

    petri_A[TRANS(rx,ry)].color = rt;
    petri_A[TRANS(rx,ry)].strength = 1;
  }

  time_t a;
  time_t b;
  time(&a);

  for(int ii = 0; ii < ITERATIONS; ii++){
    iterate_image(petri_A, petri_B);
    swap_petris();
  }

  time(&b);

  printf("This took %d seconds..\n", (int)(b - a));

  char* filename = "RPS_omp.bmp";
  make_bmp(petri_A, 0, filename);

}


void swap_petris(){
  cell* tmp1 = petri_A;
  petri_A = petri_B;
  petri_B = tmp1;
}

void free_stuff() {
    free(petri_B);
    free(petri_A);
}
