CFLAGS+= -std=c99 -g -O3

# by uncommenting this line the preprocessor will see #ifdef DEBUG as true
# CFLAGS+= -DDEBUG
objects = CA.o bitmap.o RPS.o utils.o

SHELL := /bin/bash

serial :
	gcc -std=c99 -g -O3 CA.h CA.c utils.h utils.c bitmap.h bitmap.c RPS_serial.c RPS.h -o serialRPS.out

omp :
	gcc -std=c99 -g -O3 -fopenmp CA.h CA.c utils.h utils.c bitmap.h bitmap.c RPS.c RPS.h -o ompRPS.out

pth :
	gcc -std=c99 -D_GNU_SOURCE -g -O3 -pthread CA.h CA.c utils.h utils.c bitmap.h bitmap.c p_RPS.c RPS.h -o pthreadRPS.out

remake : clean all

# We add .PHONY when a target doesn't actually create any output. In this case we just run a shell
# command, removing all object files, i.e files ending on .o
# the * syntax means [anything].o
.PHONY : clean
clean :
	rm -f *.o && rm -f *.gch && rm -f data/*.bmp && rm -f *.mp4

# Finally, the test target. Builds the 'all' target, then runs the test script on the output
.PHONY : video
video : clean all
	./myProgram && ffmpeg -framerate 60 -i data/CA-%000d.bmp -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

.PHONY : run
run : clean omp
	./meme.out
