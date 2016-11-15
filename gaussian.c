/*
 * File: gaussian.c
 * Author: Katie Levy and Michael Poyatt
 * Date: 11/10/16
 * Description: Parallel execution of matrix-vector 
 * multiplication via gaussian elimination
 */

#include<stdlib.h>
#include<stdio.h>
#include<string.h>

/* Function declarations */
int eliminate(float **matrix, int m, int n);

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
    int size = m * n;
    float *matrix[m];
    float *b = malloc(sizeof(float)*m);
    float *X = malloc(sizeof(float)*n);

    int i, j, temp;
    for(i = 0; i < m; i++){
        matrix[i] = (float *)malloc(sizeof(float)*n);
    }
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            fscanf(input, "%d", &temp);
            matrix[i][j] = (float)temp;
            if(j % n == 0){ printf("\n"); }
            printf("%d", temp);
        }
    }
    for(i = 0; i < m; i++){
        fscanf(input, "%d", &temp);
        b[i] = (float)temp;
    }

    printf("\nNEW MATRIX:\n");
    int ret = eliminate(matrix, m, n);

    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            if(j % n == 0){ printf("\n"); }
            printf("%f\t", matrix[i][j]);
        }
    }

    exit(0);
}

int eliminate(float **matrix, int m, int n){
    int r, c, k, i, index;
    float ratio;
    for(c = 0; c < n - 1; c++){
        index = c;
        for(r = (c + 1); r < m; r++){
            // Find a non-zero entry to divide by
            if(matrix[index][c] == 0){
                for(i = 0; i < m; i++){
                    if(matrix[i][c] != 0){
                        index = i;
                        break;
                    }
                }
            }
            ratio = matrix[r][c] / matrix[index][c];
            for(k = c; k < n; k++){
                matrix[r][k] = matrix[r][k] - (ratio * matrix[index][k]);
            }
        }
        int i, j;
        for(i = 0; i < m; i++){
            for(j = 0; j < n; j++){
                if(j % n == 0){ printf("\n"); }
                printf("%f\t", matrix[i][j]);
            }
        }
        printf("\n");
    }
    return 1;
}
