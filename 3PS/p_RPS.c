#define GNU
#include "RPS.h"
#include <time.h>
#include <pthread.h>
#include <stdlib.h>

void swap_petris();
void *p_thread_rps( void *pthread_id );
void rps( int id );
void free_stuff();

cell* petri_A;
cell* petri_B;

pthread_t* threads;
int* thread_ids;
pthread_barrier_t barrier;



#define ARGUMENTS(x) (x > -1)

int main(int argc, char** argv){
    /**************** Parsing input ********************/
  NUM_OF_THREADS = -1;
  if (argc > 1) {
      NUM_OF_THREADS = strtol( argv[1], NULL, 10 );
      NUM_OF_THREADS--; // main thread + extra threads.
      if ( !NUM_OF_THREADS ) {
          printf("There should be at least two threads! Setting number of threads to two.");
          NUM_OF_THREADS = 1;
      }
  } else {
      printf("Argument missing: This program takes one argument. Please specify a number of threads.\n");
      exit(0);
  }

  printf("running %d iterations with %d threads.\n",ITERATIONS, NUM_OF_THREADS + 1);

  threads = calloc( NUM_OF_THREADS, sizeof(pthread_t) );
  thread_ids = calloc( NUM_OF_THREADS, sizeof(int));
  petri_A = calloc(IMG_X*IMG_Y, sizeof(cell));
  petri_B = calloc(IMG_X*IMG_Y, sizeof(cell));

  pthread_barrier_init(&barrier, NULL, NUM_OF_THREADS + 1);

  srand(time(NULL));
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

  for (int i = 0; i < NUM_OF_THREADS; i++ ){
      thread_ids[i] = i + 1;
      pthread_create( &threads[i], NULL, p_thread_rps, (void*) &thread_ids[i] );
  }
  rps( 0 );

  for (int i = 0; i < NUM_OF_THREADS; i++ ){
      pthread_join( threads[i], NULL );
  }

  time(&b);

  printf("This took %d seconds..\n", (int)(b - a));

  char* filename = "RPS_pthread.bmp";
  make_bmp(petri_A, 0, filename);

}


void *p_thread_rps( void *pthread_id ) {
    int* id = (int*) pthread_id;
    rps( *id );
}

void rps( int id ) {
    int rows_per_thread = IMG_Y / (NUM_OF_THREADS + 1);


    int my_y_start = id > 0 ?
        id * rows_per_thread
        :  1;

    int my_y_end = id < NUM_OF_THREADS  ?
        (id + 1) * rows_per_thread
        :  IMG_Y - 2;

    for(int ii = 0; ii < ITERATIONS; ii++){
          iterate_image_pth(petri_A, petri_B, my_y_start, my_y_end);
          pthread_barrier_wait(&barrier);
          swap_petris();
      }
}


void swap_petris(){
  cell* tmp1 = petri_A;
  petri_A = petri_B;
  petri_B = tmp1;
}

void free_stuff() {
    free(petri_B);
    free(petri_A);
    free(threads);
    free(thread_ids);
}
