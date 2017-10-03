#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#include "RPS_MPI.h"

#define pixel(i, j) ((i + BORDER_SIZE) * (p_local_petri_cols + 2*BORDER_SIZE) + j + BORDER_SIZE)


void initialize();
void initialize_petri();
void exchange_borders();
void iterate_CA();
void gather_petri();
void create_types();
void free_stuff();

cell next_cell(int x, int y, cell* image);
cell pick_neighbor(int x, int y, cell* image);
bool beats(cell me, cell other);
void alloc_img();
void free_img();




int rank;
int size;
cell** image;

// I denote mpi process specific values with hungarian notation, adding a p

// The dimensions of the processor grid. Same for every process
int p_j_dims;
int p_i_dims;

// The location of a process in the process grid. Unique for every process
int p_my_j_dim;
int p_my_i_dim;

int p_north, p_south, p_east, p_west;
int p_border_top, p_border_bottom, p_border_left, p_border_right;
int p_top, p_bottom, p_left, p_right;

// The dimensions for the process local petri
int p_local_petri_cols;
int p_local_petri_rows;
int p_local_petri_offset;
int p_local_petri_size;

MPI_Comm cart_comm;

// some datatypes, useful for sending data with somewhat less primitive semantics
MPI_Datatype mpi_cell_t;
MPI_Datatype mpi_row_t;
MPI_Datatype mpi_col_t;

// Each process is responsible for one part of the petri dish.
// Since we can't update the petri-dish in place each process actually
// gets two petri-dishes which they update in a lockstep fashion.
// dish A is updated by writing to dish B, then next step dish B updates dish A.
// (or you can just swap them inbetween iterations)
cell* local_petri_A;
cell* local_petri_B;
cell* local_petri;
cell* local_petri_next;

int main(int argc, char** argv){


  // Ask MPI what size (number of processors) and rank (which process we are)
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand( 4321 * rank );

  ////////////////////////////////
  // Create cartesian communicator
  int dims[2];
  dims[0] = p_j_dims;
  dims[1] = p_i_dims;

  int periods[2]; // we set these to 0 because we are not interested in wrap-around
  periods[0] = 0;
  periods[1] = 0;

  MPI_Dims_create(size, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  MPI_Cart_shift(cart_comm, 0, 1, &p_north, &p_south);
  MPI_Cart_shift(cart_comm, 1, 1, &p_west, &p_east);

  p_i_dims = dims[0];
  p_j_dims = dims[1];

  /*
  printf("Process: %d, dims (i, j) = (%d, %d),    %d    %d \n", rank, p_my_i_dim, p_my_j_dim, IMG_X % p_j_dims, IMG_Y % p_i_dims);
  printf("\t%d\n", p_north);
  printf("%d\t%d\t%d\n", p_west, rank, p_east);
  printf("\t%d\n\n", p_south);*/

  ////////////////////////////////
  ////////////////////////////////

  initialize();

  create_types();

  for ( int iter = 0; iter < 1000; iter++ ) {
      local_petri = iter % 2 == 0 ? local_petri_A : local_petri_B;
      local_petri_next = iter % 2 == 0 ? local_petri_B : local_petri_A;

      iterate_CA();
      exchange_borders();

  }

  gather_petri();

  MPI_Finalize();

  if(rank==0){
      make_bmp(image, 0);
  }

  // You should probably make sure to free your memory here
  // We will dock points for memory leaks, don't let your hard work go to waste!
  //free_stuff();

  exit(0);
}


void create_types(){

    // cell type
    const int    nitems=2;
    int          blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(cell, color);
    offsets[1] = offsetof(cell, strength);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_cell_t);
    MPI_Type_commit(&mpi_cell_t);

    // Row type
    MPI_Type_vector( p_local_petri_cols, 1, 1, mpi_cell_t, &mpi_row_t );
    MPI_Type_commit( &mpi_row_t);

    // Column type
    MPI_Type_vector( p_local_petri_rows, 1, p_local_petri_rows + 2*BORDER_SIZE, mpi_cell_t, &mpi_col_t );
    MPI_Type_commit( &mpi_col_t);

}


void initialize(){
  // Getting rows and columns in petri_dish
  p_local_petri_rows = IMG_Y / p_i_dims;
  p_local_petri_cols = IMG_X / p_j_dims;

  // Allocate petri_dish with padding
  p_local_petri_size = (2*BORDER_SIZE + p_local_petri_cols) * (2*BORDER_SIZE + p_local_petri_rows);
  local_petri_A = malloc( p_local_petri_size * sizeof(cell) );
  local_petri_B = malloc( p_local_petri_size * sizeof(cell) );

  // Make the petri_dish aaaall white
    if (rank == 0) {

      for( int i = -BORDER_SIZE; i < p_local_petri_rows + 2*BORDER_SIZE; i++ ) {
          for( int j = -BORDER_SIZE; j < p_local_petri_cols + 2*BORDER_SIZE; j++ ) {
              local_petri_A[ pixel(i, j) ].color = 0;
              local_petri_A[ pixel(i, j) ].strength = 1;
          }
      }
    }

  // and then add some colors ...
  int color, i, j;

  for (int iter = 0; iter < p_local_petri_rows; iter++ ) {
      i = rand() % p_local_petri_rows;
      j = rand() % p_local_petri_cols;
      color = rand() % 4;
      local_petri_A[ pixel(i, j) ].color = color;
      local_petri_A[ pixel(i, j) ].strength = 1;
  }

  // Calculate border offsets (for receiving borders)
  p_border_top = pixel(-BORDER_SIZE, 0);
  p_border_bottom = pixel(p_local_petri_rows, 0);
  p_border_left = pixel(0, -BORDER_SIZE);
  p_border_right = pixel(0, p_local_petri_cols);

  // And calculate inline border offsets (for sending).
  p_top = pixel(0, 0);
  p_bottom = pixel(p_local_petri_rows - 1, 0);
  p_left = pixel(0, 0);
  p_right = pixel(0, p_local_petri_cols - 1);

}


void exchange_borders(){

    /* Exchanges borders between processes, by doing non-blocking sending and
    receiving. Will wait for all incoming data (receiving) to complete before
    continuing on to the next iteration.
    */

    MPI_Request send[4];
    MPI_Request recv[4];
    int r_length = 4;
    int r_index = 0;

    if ( p_west != -1 ) {
        MPI_Isend( local_petri + p_left, 1, mpi_col_t, p_west, 1, cart_comm, send );
        MPI_Irecv( local_petri + p_border_left, 1, mpi_col_t, p_west, 1, cart_comm, recv + r_index++ );
    } else { r_length--; }

    if ( p_east != -1 ) {
        MPI_Isend( local_petri + p_right, 1, mpi_col_t, p_east, 1, cart_comm, send + 1 );
        MPI_Irecv( local_petri + p_border_right, 1, mpi_col_t, p_east, 1, cart_comm, recv + r_index++ );
    } else { r_length--; }

    if ( p_north != -1 ) {
        MPI_Isend( local_petri + p_top, 1, mpi_row_t, p_north, 1, cart_comm, send + 2 );
        MPI_Irecv( local_petri + p_border_top, 1, mpi_row_t, p_north, 1, cart_comm, recv + r_index++ );
    } else { r_length--; }

    if ( p_south != -1 ) {
        MPI_Isend( local_petri + p_bottom, 1, mpi_row_t, p_south, 1, cart_comm, send + 3 );
        MPI_Irecv( local_petri + p_border_bottom, 1, mpi_row_t, p_south, 1, cart_comm, recv + r_index++ );
    } else { r_length--; }

    MPI_Status stats[r_length];
    MPI_Waitall( r_length, recv, stats );
}



void iterate_CA(){
  // Iterate the cellular automata one step
  for( int i = 0; i < p_local_petri_rows; i++ ) {
      for ( int j = 0; j < p_local_petri_cols; j++ ) {
          local_petri_next[ pixel(i, j) ] = next_cell( i, j, local_petri );
      }
  }
  local_petri = local_petri_next;
}


void gather_petri(){
  // Gather the final petri for process rank 0

    if (rank == 0) {
        alloc_img();

        // Start with rank 0
        for ( int i = 0; i < p_local_petri_rows; i++ ) {
            memcpy( image[i], local_petri + pixel( i, 0 ), p_local_petri_cols * sizeof(cell) );
        }

        // Then moving on to the other ranks
        for (int iter = 1; iter < size; iter ++ ) {

            MPI_Status status;
            MPI_Probe( MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status );

            /* Get data size and allocate receiving buffer */
            int count;
            MPI_Get_count( &status, mpi_cell_t, &count );
            cell* buf = malloc( count * sizeof(cell) );

            MPI_Recv( buf, count, mpi_cell_t, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            // Get the coordinates of the process
            int coords[2] ;
            MPI_Cart_coords(cart_comm, status.MPI_SOURCE, 2, coords);
            int proc_i_dim = coords[0];
            int proc_j_dim = coords[1];

            int i_offset = p_local_petri_rows * proc_i_dim;
            int j_offset = p_local_petri_cols * proc_j_dim;

            for ( int i = 0; i < p_local_petri_rows; i++ ) {
                memcpy( image[i_offset + i] + j_offset, buf + pixel( i, 0 ), p_local_petri_cols * sizeof(cell) );
            }

            free(buf);
        }
    }
  else {
      MPI_Send( local_petri, p_local_petri_size, mpi_cell_t, 0, 0, MPI_COMM_WORLD );
  }

}

void free_stuff() {
    free(local_petri_A);
    free(local_petri_B);
    free_img();
}





/////////////////////////////////////////////////
/////////////////////////////////////////////////
//  Image allocation

void alloc_img(){
    image = malloc( IMG_Y * sizeof(cell*) );
    for (int i = 0; i < IMG_X; i++ ){
        image[i] = (cell*)malloc( IMG_X * sizeof(cell) );
    }
}

void free_img(){
    for (int i = 0; i < IMG_X; i++ ){
        free(image[i]);
    }
    free(image);
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
//  Rock-paper-scizzor logic


bool beats(cell me, cell other){
  return
    (((me.color == SCISSOR) && (other.color == PAPER)) ||
     ((me.color == PAPER) && (other.color == ROCK))    ||
     ((me.color == ROCK) && (other.color == SCISSOR))  ||
     (me.color == other.color));
}


cell next_cell(int i, int j, cell* petri_dish){

  cell neighbor_cell = pick_neighbor(i, j, petri_dish);
  cell my_cell = petri_dish[ pixel(i, j) ];
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

  if(next_cell.strength > 6){
    next_cell.strength = 6;
  }

  return next_cell;
}


cell pick_neighbor(int i, int j, cell* image){
  int chosen = rand() % 9;
  chosen = chosen == 4? 5 : chosen;
  int c_i = chosen / 3;
  int c_j = chosen % 3;

  return image[  pixel( i + c_i - 1, j + c_j - 1) ];
}
