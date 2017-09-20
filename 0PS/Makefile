CFLAGS = -Wall -std=c99 -I.
CC = gcc

julia: julia.o bitmap.o
	$(CC) julia.o bitmap.o -lm -o julia

julia.o: julia.c julia.h bitmap.h
	$(CC) -c julia.c $(CFLAGS)

bitmap.o: bitmap.c bitmap.h julia.h
	$(CC) -c bitmap.c $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o *.gch
