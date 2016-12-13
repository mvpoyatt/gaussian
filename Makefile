gaussian:
	gcc -fopenmp -Wall -o gaussian gaussian.c

clean:
	rm -f gaussian
