/*
 * File: gaussian.c
 * Author: Katie Levy and Michael Poyatt
 * Date: 11/14/16
 * Description: Parallel execution of matrix-vector 
 * multiplication via gaussian elimination
 */

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#ifdef _OPENMP
#include <omp.h>
#endif


/* Function declarations */
int eliminate(float **matrix, int m, int n, float *b, int threadCount);
int backwardsSub(float **matrix, int m, int n, float *x, \
	float *b, int threadCount);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]){
    // Check for 2 command line args
    if(argc != 3){
       printf("Enter the number of threads to launch and input file\n");
       exit(0);
    }
   
    // Parse command line args
    int threadCount = (int) strtol(argv[1], NULL, 10);

    char* file = argv[2];
    FILE *input;
    int m, n;
    input = fopen(file, "r");
    fscanf(input, "%d", &m);
    fscanf(input, "%d", &n);
    if(n > m){ 
        printf("Cannot solve uniquely.\n");
        exit(0);
    }
    float *matrix[m];
    float *b = malloc(sizeof(float)*m);
    float *X = malloc(sizeof(float)*n);

    int i, j, temp;
    for(i = 0; i < m; i++){
        matrix[i] = (float *)malloc(sizeof(float)*n);
    }
    // Parse vector A into matrix and b into b
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            fscanf(input, "%d", &temp);
            matrix[i][j] = (float)temp;
        }
    }
    for(i = 0; i < m; i++){
        fscanf(input, "%d", &temp);
        b[i] = (float)temp;
    }

    // Perform the Gaussian Elimination
    eliminate(matrix, m, n, b, threadCount);

	// Perform the backwards substitution in parallel
	backwardsSub(matrix, m, n, X, b, threadCount);
	for(i = 0; i < n; i++){
            printf("x%d = %f\t", i, X[i]);
    }
	printf("\n");
    exit(0);
}

// Performs Gaussian elimination on matrix 2d array
// This produces the upper triangular form of the system of equations
// matrix is the 2d array to represent A
// m is the rows of matrix and n the columns
// b is the vector representing b
int eliminate(float **matrix, int m, int n, float *b, int threadCount){
    int r, c, k, i, index;
    float ratio;
    for(c = 0; c < n - 1; c++){
        index = c;
        #pragma omp parallel for num_threads(threadCount)
        for(r = (c + 1); r < m; r++){
            // Find a non-zero entry to divide by
            if((matrix[index][c] == 0) || (index == r)){
                for(i = (m-1); i >= 0; i--){
                    if((matrix[i][c] != 0) && (i != r)){
                        index = i;
                        break;
                    }
                }
            }
	        // Eliminate x^c in row r by subtracting another row's
	        // corresponding element times the ratio for each element in r
            ratio = matrix[r][c] / matrix[index][c];
            for(k = 0; k < n; k++){
                matrix[r][k] = matrix[r][k] - (ratio * matrix[index][k]);
            }
            b[r] = b[r] - (ratio * b[index]);
        }
	}
    return 1;
}


// Solves for x on each of the rows starting from top to bottom
// such that A * x = b. Uses omp parallel
// matrix is a 2d array representing A
// m is the number of rows and n the number of columns in matrix
// x is the vector representing x which we are solving for
// b is the vector representing b in the equation
// threadCount is the number of thread we will divide work amoung
int backwardsSub(float **matrix, int m, int n, float *x, float *b, int threadCount){
	int r, c;
	float temp;

    #pragma omp parallel num_threads(threadCount) default(none) \
        private(r,c) shared(matrix, x, b, m, n, temp)

	for(r = m - 1; r >= 0 ; r--){
		#pragma omp single
		temp = b[r];
		#pragma omp for reduction(-: temp)
		for(c = r + 1; c < n; c++){
			temp -= matrix[r][c]*x[c];
		}
		#pragma omp single
		temp /= matrix[r][r];
		x[r] = temp;
	}
	return 1;
}	
