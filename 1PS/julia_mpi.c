#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "julia.h"
#include "bitmap.h"
#include "mpi.h"

double x_start=-2.01;
double x_end=1;
double yupper;
double ylower;

double ycenter=1e-6;
double step;
double start;

int pixel[ XSIZE*YSIZE ];


// I suggest you implement these, however you can do fine without them if you'd rather operate
// on your complex number directly.
complex_t square_complex(complex_t c){
	complex_t d;
	d.real = pow( c.real, 2 ) - pow( c.imag, 2 );
	d.imag = 2 * c.real * c.imag;
	return d;
}

complex_t add_complex(complex_t a, complex_t b){
	complex_t c;
	c.real = a.real + b.real;
	c.imag = a.imag + b.imag;
	return c;
}

complex_t add_real(complex_t a, int b){
	complex_t c;
	c.real = a.real + b;
	c.imag = a.imag;
	return c;
}



// add julia_c input arg here?
void calculate( complex_t julia_C, int min_y, int max_y ) {
	for( int i = 0; i < XSIZE; i++ ) {
		for( int j = min_y; j < max_y; j++ ) {

			/* Calculate the number of iterations until divergence for each pixel.
			   If divergence never happens, return MAXITER */
			complex_t c;
			complex_t z;
			int iter=0;

			// find our starting complex number c
			c.real = ( x_start + step*i );
			c.imag = ( ylower + step*j );

			// our starting z is c
			z = c;

			// iterate until we escape
			while( z.real*z.real + z.imag*z.imag < 4 ) {
				// Each pixel in a julia set is calculated using z_n = (z_n-1)² + C
				// C is provided as user input, so we need to square z and add C until we
				// escape, or until we've reached MAXITER
				z = add_complex( square_complex( z ), julia_C );
				// z = z squared + C

				if( ++iter == MAXITER ) break;
			}
			pixel[ PIXEL(i,j) ] = iter;
		}
	}
}


int main(int argc, char **argv) {
	if( argc == 1 ) {
		puts("Usage: JULIA\n");
		puts("Input real and imaginary part. ex: ./julia 0.0 -0.8");
		return 0;
	}

	MPI_Init( NULL, NULL );

	int world_size;
	MPI_Comm_size( MPI_COMM_WORLD, &world_size );

	int world_rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

	if (world_rank == 0)
	{
		start = MPI_Wtime();
	}

	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step = ( x_end - x_start ) / XSIZE;
	yupper = ycenter + ( step*YSIZE ) / 2;
	ylower = (ycenter - ( step*YSIZE ) / 2);

	/* Calculate problem area for each process */
	int num_of_rows = YSIZE / world_size;
	int min_y = world_rank * num_of_rows;
	int max_y = (world_rank == world_size - 1) ? YSIZE : (world_rank + 1) * num_of_rows;

	//printf("Rank %d:   %d, %d\n", world_rank, min_y, max_y);

	// Unlike the mandelbrot set where C is the coordinate being iterated, the
	// julia C is the same for all points and can be chosen arbitrarily
	complex_t julia_C;

	// Get the command line args
	julia_C.real = strtod( argv[1], NULL );
	julia_C.imag = strtod( argv[2], NULL );

	calculate( julia_C, min_y, max_y );

	if ( world_rank != 0 )
	{
		MPI_Send( pixel + PIXEL(0, min_y), (max_y - min_y)*XSIZE, MPI_INT, 0, 0, MPI_COMM_WORLD );
	}
	else
	{
		for (int i = 1; i < world_size; i++)
		{
			/* Probe the sender, to check data size */
			MPI_Status status;
			MPI_Probe( MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status );

			/* Get data size */
			int source = status.MPI_SOURCE;
			int count;
			MPI_Get_count( &status, MPI_INT, &count );

			/* Get position in pixel array */
			min_y = status.MPI_SOURCE * num_of_rows;
			int *buf = (int*) (pixel + PIXEL( 0, min_y ));

			/* Actually receive data */
			MPI_Recv( buf, count, MPI_INT, source, 0, MPI_COMM_WORLD, &status );
			//printf("Recieved from %d\n", source);
		}

		/* create nice image from iteration counts. take care to create it upside
		 down (bmp format) */

		unsigned char *buffer = calloc( XSIZE*YSIZE*3, 1 );
		for( int i=0; i < XSIZE; i++) {
			for( int j=0; j < YSIZE; j++) {
				int p = ( (YSIZE - j - 1)*XSIZE + i ) * 3;
	  			fancycolour( buffer + p, pixel[ PIXEL(i,j) ] );
			}
		}

		/* write image to disk */

		savebmp( "julia.bmp", buffer, XSIZE, YSIZE );
		float t = MPI_Wtime() - start;
		printf( "\nProcesses:  %d\tProblem size:  (%d, %d)\tTime used: %f\n\n", world_size, XSIZE, YSIZE, t );
	}

	MPI_Finalize();
	return 0;

}
