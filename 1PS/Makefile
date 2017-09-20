CFLAGS = -Wall -std=c99 -I.
CC = gcc

julia_mpi: julia_mpi.o bitmap.o
	mpicc julia_mpi.o bitmap.o -lm -o julia_mpi

julia_mpi.o: julia_mpi.c julia.h bitmap.h
	mpicc -c julia_mpi.c $(CFLAGS)

bitmap.o: bitmap.c bitmap.h julia.h
	$(CC) -c bitmap.c $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o *.gch
