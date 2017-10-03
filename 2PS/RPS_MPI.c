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

void get_column( int col, cell* buffer, cell* local_petri );
void fill_column( int col, cell* source, cell* local_petri );
void fill_left();
void fill_right();

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
int p_border_top, p_border_bottom;
int p_top, p_bottom;

// The dimensions for the process local petri
int p_local_petri_cols;
int p_local_petri_rows;
int p_local_petri_offset;
int p_local_petri_size;

MPI_Comm cart_comm;

// some datatypes, useful for sending data with somewhat less primitive semantics
MPI_Datatype border_row_t;  // TODO: Implement this
MPI_Datatype border_col_t;  // TODO: Implement this
MPI_Datatype local_petri_t; // Already implemented
MPI_Datatype mpi_cell_t;    // Already implemented

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

cell* p_send_west;
cell* p_from_west;

cell* p_send_east;
cell* p_from_east;


int main(int argc, char** argv){


  // Ask MPI what size (number of processors) and rank (which process we are)
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  srand( 1234 * rank );


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
      local_petri      = iter % 2 == 0 ? local_petri_A : local_petri_B;
      local_petri_next = iter % 2 == 0 ? local_petri_B : local_petri_A;
      iterate_CA();
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

    ////////////////////////////////
    ////////////////////////////////
    // cell type
    const int    nitems=2;
    int          blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Aint     offsets[2];

    offsets[0] = offsetof(cell, color);
    offsets[1] = offsetof(cell, strength);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_cell_t);
    MPI_Type_commit(&mpi_cell_t);
    ////////////////////////////////
    ////////////////////////////////



    ////////////////////////////////
    ////////////////////////////////
    // A message for a local petri-dish
    MPI_Type_contiguous(p_local_petri_size,
                      mpi_cell_t,
                      &local_petri_t);
    MPI_Type_commit(&local_petri_t);
    ////////////////////////////////
    ////////////////////////////////


    //TODO: Create MPI types for border exchange

    MPI_Type_vector( p_local_petri_cols, 1, 1, mpi_cell_t, &mpi_row_t );
    MPI_Type_commit( &mpi_row_t);

    MPI_Type_vector( p_local_petri_rows, 1, p_local_petri_rows + 2*BORDER_SIZE, mpi_cell_t, &mpi_col_t );
    MPI_Type_commit( &mpi_col_t);




}


void initialize(){
  //TODO: assign the following to something more useful than 0
  p_local_petri_cols = IMG_X / p_j_dims;
  p_local_petri_rows = IMG_Y / p_i_dims;

  printf("rows, cols:  %d, %d\n", p_local_petri_rows, p_local_petri_cols);

  // TODO: When allocating these buffers, keep in mind that you might need to allocate a little more
  // than just your piece of the petri.
  p_local_petri_size = (2*BORDER_SIZE + p_local_petri_cols) * (2*BORDER_SIZE + p_local_petri_rows);
  local_petri_A = malloc( p_local_petri_size * sizeof(cell) );
  local_petri_B = malloc( p_local_petri_size * sizeof(cell) );

  // TODO: Randomly perturb the local dish. Only perturb cells that belong to your process,
  // leave border pixels white.


  // Make it aaaall white
  if (rank == 0) {

  for( int i = -BORDER_SIZE; i < p_local_petri_rows + 2*BORDER_SIZE; i++ ) {
      for( int j = -BORDER_SIZE; j < p_local_petri_cols + 2*BORDER_SIZE; j++ ) {
          local_petri_A[ pixel(i, j) ].color = 0;
          local_petri_A[ pixel(i, j) ].strength = 1;
      }
  }
}

/*
    if (rank == 1) {

    for( int i = -BORDER_SIZE; i < p_local_petri_rows + 2*BORDER_SIZE; i++ ) {
        for( int j = -BORDER_SIZE; j < p_local_petri_cols + 2*BORDER_SIZE; j++ ) {
            local_petri_A[ pixel(i, j) ].color = 1;
            local_petri_A[ pixel(i, j) ].strength = 4;
        }
    }
    }

    if (rank == 2) {

    for( int i = -BORDER_SIZE; i < p_local_petri_rows + 2*BORDER_SIZE; i++ ) {
        for( int j = -BORDER_SIZE; j < p_local_petri_cols + 2*BORDER_SIZE; j++ ) {
            local_petri_A[ pixel(i, j) ].color = 2;
            local_petri_A[ pixel(i, j) ].strength = 2;
        }
    }
  }
      if (rank == 3) {

      for( int i = -BORDER_SIZE; i < p_local_petri_rows + 2*BORDER_SIZE; i++ ) {
          for( int j = -BORDER_SIZE; j < p_local_petri_cols + 2*BORDER_SIZE; j++ ) {
              local_petri_A[ pixel(i, j) ].color = 3;
              local_petri_A[ pixel(i, j) ].strength = 2;
          }
      }
      }

*/




  // and then add some seeds ...
  int color, i, j;

  for (int iter = 0; iter < p_local_petri_rows; iter++ ) {
      i = rand() % p_local_petri_rows;
      j = rand() % p_local_petri_cols;
      color = rand() % 4;
      local_petri_A[ pixel(i, j) ].color = color;
      local_petri_A[ pixel(i, j) ].strength = 1;
  }


  // Calculate border offsets
  p_border_top = pixel(-BORDER_SIZE, 0);
  p_border_bottom = pixel(p_local_petri_rows, 0);

  // And calculate inline border offsets (for sending - these might not be needed).
  p_top = pixel(0, 0);
  p_bottom = pixel(p_local_petri_rows - 1, 0);

  // Allocate sending column buffers. TODO: Remember to free these!!
  p_send_west = malloc( p_local_petri_rows * sizeof( cell ) );
  p_from_west = malloc( p_local_petri_rows * sizeof( cell ) );

  p_send_east = malloc( p_local_petri_rows * sizeof( cell ) );
  p_from_east = malloc( p_local_petri_rows * sizeof( cell ) );

}


void exchange_borders(){
    //TODO: Exchange borders inbetween each step

    // Just sending and receiving as a lunatic here. Only ones that need some
    // post-processing is when sending west or east.
    // Trying a non-blocking approach ...

    MPI_Request cares[2];     // 0 = receive from west, 1 = receive from east
    MPI_Request dont_cares[6];
    /* A bit of a weird distinction, but the two arrays are used for waitall
    and waitany calls later. Basically, we don't need to do any post-processing
    after sending/receiving to/from the processes in the dont_cares array.
    It is needed, however, in the cares array (filling columns from west and east). */

    int dc_offset = 0;
    int dc_length = 6;
    int c_offset = 0;
    int c_length = 2;

    if ( p_west != -1 ) {
        // printf("Rank %d sends to west.\n", rank);
        get_column( 0, p_send_west, local_petri );
        MPI_Isend( local_petri + pixel(0, 0), 1, mpi_col_t, p_west, 1, cart_comm, dont_cares + dc_offset++);
        MPI_Irecv( local_petri + pixel(0, -BORDER_SIZE), 1, mpi_col_t, p_west, 1, cart_comm, cares + c_offset++ );
    } else { dc_length--; c_length--; }

    if ( p_east != -1 ) {
        // printf("Rank %d sends to east.\n", rank);
        get_column( p_local_petri_cols - 1, p_send_east, local_petri );
        MPI_Isend( local_petri + pixel(0, p_local_petri_cols - 1), 1, mpi_col_t, p_east, 1, cart_comm, dont_cares + dc_offset++ );
        MPI_Irecv( local_petri + pixel(0, p_local_petri_cols), 1, mpi_col_t, p_east, 1, cart_comm, cares + c_offset++ );
    } else { dc_length--; c_length--; }

    if ( p_north != -1 ) {
        // printf("Rank %d sends to north.\n", rank);
        MPI_Isend( local_petri + pixel( 0, 0 ), 1, mpi_row_t, p_north, 1, cart_comm, dont_cares + dc_offset++ );
        MPI_Irecv( local_petri + pixel( -1, 0 ), 1, mpi_row_t, p_north, 1, cart_comm, dont_cares + dc_offset++ );
    } else { dc_length = dc_length - 2; }

    if ( p_south != -1 ) {
        // printf("Rank %d sends to south.\n", rank);
        MPI_Isend( local_petri + pixel( p_local_petri_rows - 1, 0 ), 1, mpi_row_t, p_south, 1, cart_comm, dont_cares + dc_offset++ );
        MPI_Irecv( local_petri + pixel( p_local_petri_rows, 0 ), 1, mpi_row_t, p_south, 1, cart_comm, dont_cares + dc_offset++ );
    } else { dc_length = dc_length - 2; }

    while ( c_length > 0 ) {
        MPI_Status stat;
        int index;
        int length = c_length;
        MPI_Waitany( length, cares, &index, &stat );

        if ( stat.MPI_SOURCE == p_west ) { /*fill_left(); /*printf("Rank %d received from left.\n", rank); */}
        else if ( stat.MPI_SOURCE == p_east ) { /*fill_right(); /*printf("Rank %d received from right.\n", rank);*/ }
        c_length --;
    }

    MPI_Status stats[dc_length];
    MPI_Waitall( dc_length, dont_cares, stats );

    for (int i = 0; i < dc_length; i++ ){
        // printf("Rank %d has sent/received from %d. Error code: %d\n", rank, stats[i].MPI_SOURCE, stats[i].MPI_ERROR);
    }

}

void get_column( int col, cell* buffer, cell* local_petri ) {
    // Copies a column from local_petri into the buffer. It is expected that
    // the size of the buffer is equal to the size of a column in local_petri.
    for( int i = 0; i < p_local_petri_rows; i++ ) {
        buffer[i] = local_petri[ pixel(i, col) ];
    }
}

void fill_column( int col, cell* source, cell* local_petri ) {
    // Copies the contents of source into local_petri.
    for( int i = 0; i < p_local_petri_rows; i++ ) {
        local_petri[ pixel(i, col) ] = source[i];
    }
}

void fill_left() {
    fill_column( -1, p_from_west, local_petri );
}

void fill_right() {
    fill_column( p_local_petri_rows, p_from_east, local_petri );
}

void iterate_CA(){
  //TODO: Iterate the cellular automata one step

  for( int i = 0; i < p_local_petri_rows; i++ ) {
      for ( int j = 0; j < p_local_petri_cols; j++ ) {
          local_petri_next[ pixel(i, j) ] = next_cell( i, j, local_petri );
      }
  }

  local_petri = local_petri_next;
  exchange_borders();
}

void gather_petri(){
  //TODO: Gather the final petri for process rank 0

    if (rank == 0) {
        alloc_img();

        // Start with rank 0
        for ( int i = 0; i < p_local_petri_rows; i++ ) {
            memcpy( image[i], local_petri + pixel( i, 0 ), p_local_petri_cols * sizeof(cell) );
        }

        // Then moving on to the other ranks
        for (int proc = 1; proc < size; proc ++ ) {

            MPI_Status status;
            MPI_Probe( MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status );

            /* Get data size and allocate receiving buffer */
            int count;
            MPI_Get_count( &status, mpi_cell_t, &count );
            cell* buf = malloc( count * sizeof(cell) );

            MPI_Recv( buf, count, mpi_cell_t, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            int coords[2] ;
            MPI_Cart_coords(cart_comm, status.MPI_SOURCE, 2, coords);
            int proc_i_dim = coords[0];
            int proc_j_dim = coords[1];

            int i_offset = p_local_petri_rows * proc_i_dim;
            int j_offset = p_local_petri_cols * proc_j_dim;


            printf("(i, j) offset is %d, %d\n", i_offset, j_offset);

            printf("WHAT DO I GET   %d %d\n", coords[0], coords[1]);



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

    free(p_send_west);
    free(p_from_west);

    free(p_send_east);
    free(p_from_east);

    free_img();
}





// BAD GIRL BAD GIRL

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
    next_cell.strength = 4;
  }

  return next_cell;
}


cell pick_neighbor(int i, int j, cell* image){
  int chosen = 2 * (rand() % 4) + 1;  // Will choose 1, 3, 5 or 7

  int c_i = chosen / 3;
  int c_j = chosen % 3;

  return image[  pixel( i + c_i - 1, j + c_j - 1) ];
}
