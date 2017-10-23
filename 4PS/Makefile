CCFLAGS = -Wall -Wshadow -O2 -g
LFLAGS = -lm


all: test example1 example4


test: test.o genann_blas.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
	./$@


example1: example1.o genann_blas.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

example4: example4.o genann_blas.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

.c.o:
	$(CC) -c $(CCFLAGS) $< -o $@


clean:
	rm -f *.o
	rm -f persist.txt
	rm -f test example1 example4
	rm -f *.exe
