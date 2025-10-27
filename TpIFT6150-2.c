/*------------------------------------------------------*/
/* Prog    : TpIFT6150-2.c                              */
/* Auteurs : Samuel Fournier et Alexandre Toutant       */
/* Date    : 20 octobre 2025                            */
/* version : 1.0                                        */
/* langage : C                                          */
/* labo    : DIRO                                       */
/*------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "FonctionDemo2.h"

/*------------------------------------------------*/
/* DEFINITIONS -----------------------------------*/
/*------------------------------------------------*/
#define NAME_IMG_IN "photograph"
#define NAME_IMG_GAUSS "TpIFT6150-2-filtre-gaussien"
#define NAME_IMG_GRADIENT "TpIFT6150-2-gradient"
#define NAME_IMG_SUPPRESSION "TpIFT6150-2-suppression"
#define NAME_IMG_CANNY "TpIFT6150-2-canny"

/*
Utilities structures
*/

typedef struct {
    float sum;          
    int count;          
    float* content;     
    int capacity;       
} Bin;

#define NUM_BINS 100 
#define MAX_BIN_CAPACITY 10 

/*
Utilities functions
*/

int** imatrix_allocate_2d(int height, int width) {
    int** matrix;
    int* imptr;
    int i;
    matrix = (int**)malloc(sizeof(int*) * height);
    if(matrix == NULL) printf("Could not create Int matrix");

    imptr = (int*)malloc(sizeof(int) * height * width);
    if(imptr == NULL) printf("Could not create Int matrix");

    for(i = 0; i < height; i++, imptr += width) {
        matrix[i] = imptr;
    } 
    
    return matrix;
}

void free_imatrix_2d(int** mat) {
    free(mat[0]);
    free(mat);
}

void visualizeImage(float** image, char* imageName, int width, int height) {
    float** imageViz = fmatrix_allocate_2d(height, width);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            imageViz[i][j] = image[i][j];
        }
    }

    Recal(imageViz, height, width);
    SaveImagePgm(imageName, imageViz, height, width);
    free_fmatrix_2d(imageViz);
}

void sortList(float* list, int size) {
    float key;
    int j;

    for (int i = 1; i < size; i++) {
        key = list[i]; // Store the current element to be inserted
        j = i - 1;

        // Move elements of sortedGrads[0..i-1] that are greater than key
        // to one position ahead of their current position
        while (j >= 0 && list[j] > key) {
            list[j + 1] = list[j];
            j = j - 1;
        }
        list[j + 1] = key; // Insert the key into its correct spot
    }
}

/*
Main parts of the Canny Algorithm
*/

// Recursive method for canny filter

void follow(float** sups, int x, int y, int** orient, float** result, int width, int height, float tauL) {
    // If higher than lower bound and not a contour, then it becomes one

    if(sups[y][x] > tauL && result[y][x] == 0.f) {
        result[y][x] = 255.f;
        for(int i = 0; i < 4; i++) {
            int dir = orient[y][x];
            switch (dir) {
            case 0:
                if(x - 1 >= 0) {
                    follow(sups, x - 1, y, orient, result, width, height, tauL);
                }
                else if(x + 1 < width) {
                    follow(sups, x + 1, y, orient, result, width, height, tauL);
                }
            case 45:
                if(x + 1 < width && y - 1 >= 0) {
                    follow(sups, x + 1, y - 1, orient, result, width, height, tauL);
                }
                else if(x - 1 >= 0 && y + 1 < height) {
                    follow(sups, x - 1, y + 1, orient, result, width, height, tauL);
                }
            case 90:
                if(y - 1 >= 0) {
                    follow(sups, x, y - 1, orient, result, width, height, tauL);
                }
                else if(y + 1 < height) {
                    follow(sups, x, y + 1, orient, result, width, height, tauL);
                }
            case 135:
                if(x - 1 >= 0 && y - 1 >= 0) {
                    follow(sups, x - 1, y - 1, orient, result, width, height, tauL);
                }
                else if(x + 1 < width && y + 1 < height) {
                    follow(sups, x + 1, y + 1, orient, result, width, height, tauL);
                }
            }
        }
    }
}

void recursiveHysteresis(float** result, float** sups, int** orient, int width, int height, float tauL, float tauH) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(sups[y][x] > tauH) {
                follow(sups, x, y, orient, result, width, height, tauL);
            }
        }
    }
}

void queueHysteresis(float** result, float** sups, int** direction, int width, int height, float tauL, float tauH) {
    int maxq = height * width;
    int *qY = (int*)malloc(sizeof(int)*maxq);
    int *qX = (int*)malloc(sizeof(int)*maxq);
    int head = 0, tail = 0;

    /* Marquer les forts (>= tau_H) */
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (sups[i][j] > tauH) {
                result[i][j] = 255.0f;
                qY[tail] = i; qX[tail] = j; ++tail;
            }
        }
    }

    /* Propager aux faibles (>= tau_L) connectés aux forts (8-connexité) */
    while (head < tail) {
        int y = qY[head], x = qX[head]; ++head;
        for (int dy8 = -1; dy8 <= 1; ++dy8) {
            for (int dx8 = -1; dx8 <= 1; ++dx8) {
                if (dy8 == 0 && dx8 == 0) continue;
                int ny = y + dy8, nx = x + dx8;
                if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;

                if (result[ny][nx] == 0.0f && sups[ny][nx] >= tauL) {
                    result[ny][nx] = 255.0f;
                    qY[tail] = ny; qX[tail] = nx; ++tail;
                }
            }
        }
    }  
}

void applyGaussienFilter(float** imgReal, float** imgImaginary, float var, int width, int height) {
    float** kerReal = fmatrix_allocate_2d(height, width);
    float** kerImaginary = fmatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            kerImaginary[i][j] = 0.f;
            int x = (j < width / 2) ? j : j - width;
            int y = (i < height / 2) ? i : i - height;

            kerReal[i][j] = funcgauss2D(x, y, var);
        }
    }

    FFTDD(imgReal, imgImaginary, height, width);
    FFTDD(kerReal, kerImaginary, height, width);

    // Convolution in frequence space
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            float imgR, imgI, kerR, kerI;
            imgR = imgReal[i][j];
            imgI = imgImaginary[i][j];
            kerR = kerReal[i][j];
            kerI = kerImaginary[i][j];

            float realPart = imgR * kerR - imgI * kerI;
            float imaginaryPart = imgR * kerI + imgI * kerR;

            imgReal[i][j] = realPart;
            imgImaginary[i][j] = imaginaryPart;
        }
    }

    IFFTDD(imgReal, imgImaginary, height, width);
    visualizeImage(imgReal, NAME_IMG_GAUSS, height, width);

    free_fmatrix_2d(kerReal);
    free_fmatrix_2d(kerImaginary);
}

void gradientFilter(float** imgReal, float** gx, float** gy, float** norms, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            /* (-1,1) avant; pour la dernière col/ligne, arrière */
            float dx = (j + 1 < width)  ? (imgReal[i][j + 1] - imgReal[i][j])
                                        : (imgReal[i][j] - imgReal[i][j - 1]);
            float dy = (i + 1 < height) ? (imgReal[i + 1][j] - imgReal[i][j])
                                        : (imgReal[i][j] - imgReal[i - 1][j]);

            gx[i][j] = dx;
            gy[i][j] = dy;
            norms[i][j] = sqrtf(SQUARE(dx) + SQUARE(dy));
        }
    }
}

void getAngles(int** directions, float** gx, float** gy, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            float dx = gx[i][j], dy = gy[i][j];

            float angle = atan2f(dy, dx) * (180.f / PI);

            if(angle < 0.f) angle += 180.f;

            int approx;
            if((angle >= 0.f && angle < 22.5) || (angle >= 157.5 && angle < 180.f)) {
                approx = 0;
            }
            else if(angle < 67.5) {
                approx = 45;
            }
            else if(angle < 112.5) {
                approx = 90;
            }
            else {
                approx = 135;
            }

            directions[i][j] = approx;
        }
    }
}

void nonMaxSuppression(float** nmSupps, int** directions, float** norms, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            float norm = norms[i][j];
            float qn = 0.f, rn = 0.f;
            int direction = directions[i][j];

            switch (direction)
            {
            case 0:
                qn = (j-1 >= 0) ? norms[i][j-1] : 0.0f;
                rn = (j+1 <  width) ? norms[i][j+1] : 0.0f;
            case 45:
                qn = (i-1 >= 0) ? norms[i-1][j] : 0.0f;
                rn = (i+1 <  height) ? norms[i+1][j] : 0.0f;
            case 90:
                qn = (i-1 >= 0 && j+1 < width)  ? norms[i-1][j+1] : 0.0f;
                rn = (i+1 < height && j-1 >= 0) ? norms[i+1][j-1] : 0.0f;
            case 135:
                qn = (i-1 >= 0 && j-1 >= 0) ? norms[i-1][j-1] : 0.0f;
                rn = (i+1 < height && j+1 < width) ? norms[i+1][j+1] : 0.0f;
            }

            nmSupps[i][j] = (norm > qn && norm > rn) ? norm : 0.f;
        }
    }
}

void histNoBin(float** norms, int** directions, int width, int height, float pH) {
    float* sortedGrads = fmatrix_allocate_1d(height * width);

    int idx = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            sortedGrads[idx++] = norms[i][j];
        }
    }
    
    sortList(sortedGrads, height * width);

    // 0. Find the Maximum Gradient (needed to calculate bin width)
    float g_max = 0.0f;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (norms[i][j] > g_max) {
                g_max = norms[i][j];
            }
        }
    }

    // Safety check for empty or uniform image
    if (g_max <= 0.0f) {
        printf("Erreur: Le gradient maximum est zero. Abandon.\n");
        return;
    }

    int indexToCut = (int)floor((float)(height * width) * pH);

    if(indexToCut >= height * width) indexToCut = (height * width) - 1;

    float tauH = sortedGrads[indexToCut];
    printf("Max: %f\n", sortedGrads[(height * width) - 1]);
    printf("Calculated tau_h (ph = %f): %f\n", pH, tauH);
    float tauL = 0.5 * tauH;
    printf("Calculated tau_l (ph = %f): %f\n", pH, tauL);

    float** nmSupps = fmatrix_allocate_2d(height, width);
    nonMaxSuppression(nmSupps, directions, norms, width, height);

    Recal(nmSupps, height, width);
    visualizeImage(nmSupps, NAME_IMG_SUPPRESSION, width, height);

    int hysteresisChoice;
    printf("Choose a way of doing the hysteresis.\n");
    printf("0: Queue based methode\n");
    printf("1: Recursive method\n");
    scanf("%d", &hysteresisChoice);

    float** result = fmatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            result[i][j] = 0.f;
        }
    }

    switch (hysteresisChoice)
    {
    case 0:
        queueHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    
    case 1:
        recursiveHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    }
    visualizeImage(result, NAME_IMG_CANNY, width, height);

    free_fmatrix_1d(sortedGrads);

    free_fmatrix_2d(nmSupps);

    free_fmatrix_2d(result);
}

void histBin(float** norms, int** directions, int width, int height, float pH) {
    float* sortedGrads = fmatrix_allocate_1d(height * width);

    int idx = 0;
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            sortedGrads[idx++] = norms[i][j];
        }
    }
    
    sortList(sortedGrads, height * width);

    // 0. Find the Maximum Gradient (needed to calculate bin width)
    float g_max = 0.0f;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (norms[i][j] > g_max) {
                g_max = norms[i][j];
            }
        }
    }

    // Safety check for empty or uniform image
    if (g_max <= 0.0f) {
        printf("Erreur: Le gradient maximum est zero. Abandon.\n");
        return;
    }

    float bin_width = g_max / (float)NUM_BINS;
    
    // 1. Initialize Histogram Bins (Array of structs)
    Bin histogram[NUM_BINS];
    for (int i = 0; i < NUM_BINS; ++i) {
        histogram[i].sum = 0.0f;
        histogram[i].count = 0;
        // Allocate initial memory for the content (will be reallocated later)
        histogram[i].capacity = MAX_BIN_CAPACITY;
        histogram[i].content = (float*)malloc(sizeof(float) * MAX_BIN_CAPACITY);
        if (histogram[i].content == NULL) {
             printf("Erreur d'allocation mémoire pour un bin.\n"); return;
        }
    }

    // 2. Populate Buckets
    int bin_index;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float g = norms[i][j];
            
            // Calculate bin index (clamp to last bin if exactly g_max)
            bin_index = (int)floorf(g / bin_width);
            if (bin_index >= NUM_BINS) {
                bin_index = NUM_BINS - 1; 
            }
            
            // Check if capacity needs to be increased (dynamic list)
            if (histogram[bin_index].count >= histogram[bin_index].capacity) {
                histogram[bin_index].capacity *= 2;
                histogram[bin_index].content = (float*)realloc(histogram[bin_index].content, 
                                                sizeof(float) * histogram[bin_index].capacity);
                if (histogram[bin_index].content == NULL) {
                     printf("Erreur de réallocation mémoire pour un bin.\n"); return;
                }
            }

            // Store gradient, update sum and count
            histogram[bin_index].content[histogram[bin_index].count] = g;
            histogram[bin_index].sum += g;
            histogram[bin_index].count++;
        }
    }

    // 3. Find the Percentile Bin and Calculate tau_H
    int total_pixels = height * width;
    int target_count = (int)floorf((float)total_pixels * pH);
    int cumulative_count = 0;
    int percentile_bin_index = -1;

    for (int i = 0; i < NUM_BINS; ++i) {
        cumulative_count += histogram[i].count;
        if (cumulative_count >= target_count) {
            percentile_bin_index = i;
            break;
        }
    }

    int tauChoice;
    printf("Choose how tauH is found\n");
    printf("0: Average of bin\n");
    printf("1: Median of bin\n");
    scanf("%d", &tauChoice);

    float tauH, tauL;

    switch (tauChoice)
    {
    case 0:
        if (percentile_bin_index != -1 && histogram[percentile_bin_index].count > 0) {
        // Calculate the average of all gradients in the percentile bin
            tauH = histogram[percentile_bin_index].sum / (float)histogram[percentile_bin_index].count;
        } else {
        // Fallback: Use the max gradient if no data is found (shouldn't happen)
            tauH = g_max; 
        }
        tauL = 0.5f * tauH;
        break;
    
    case 1:
        Bin binThreshold = histogram[percentile_bin_index];
        float* contents = binThreshold.content;
        int binSize = binThreshold.count;
        sortList(contents, binSize);
        tauH = contents[binThreshold.count / 2];
        tauL = 0.5f * tauH;
        break;
    }

    printf("Max: %f\n", sortedGrads[(height * width) - 1]);
    printf("Calculated tau_h (ph = %f): %f\n", pH, tauH);
    printf("Calculated tau_l (ph = %f): %f\n", pH, tauL);

    float** nmSupps = fmatrix_allocate_2d(height, width);
    nonMaxSuppression(nmSupps, directions, norms, width, height);

    Recal(nmSupps, height, width);
    visualizeImage(nmSupps, NAME_IMG_SUPPRESSION, width, height);

    int hysteresisChoice;
    printf("Choose a way of doing the hysteresis.\n");
    printf("0: Queue based methode\n");
    printf("1: Recursive method\n");
    scanf("%d", &hysteresisChoice);

    float** result = fmatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            result[i][j] = 0.f;
        }
    }

    switch (hysteresisChoice)
    {
    case 0:
        queueHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    
    case 1:
        recursiveHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    }
    visualizeImage(result, NAME_IMG_CANNY, width, height);

    free_fmatrix_1d(sortedGrads);

    for (int i = 0; i < NUM_BINS; ++i) {
        free(histogram[i].content);
    }

    free_fmatrix_2d(nmSupps);

    free_fmatrix_2d(result);
}

void doNormalCanny() {
    int width, height;
    float sigma, tauH, tauL, var;

    // Parameter input
    printf("Entrez l'écart type du filtre Gaussien: ");
    scanf("%f", &sigma);
    var = SQUARE(sigma);
    printf("Entrez la valeur de tauL: ");
    scanf("%f", &tauL);
    printf("Entrez la valeur de tauH: ");
    scanf("%f", &tauH);

    // Matrix allocation/creation
    float** imgReal = LoadImagePgm(NAME_IMG_IN, &height, &width);
    float** imgImaginary = fmatrix_allocate_2d(height, width); 

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            imgImaginary[i][j] = 0.f;
        }
    }

    applyGaussienFilter(imgReal, imgImaginary, var, width, height);

    float** gx = fmatrix_allocate_2d(height, width);
    float** gy = fmatrix_allocate_2d(height, width);
    float** norms = fmatrix_allocate_2d(height, width);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            gx[i][j] = 0.f;
            gy[i][j] = 0.f;
            norms[i][j] = 0.f;
        }
    }

    gradientFilter(imgReal, gx, gy, norms, width, height);

    visualizeImage(norms, NAME_IMG_GRADIENT, height, width);

    int** directions = imatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            directions[i][j] = 0;
        }
    }

    getAngles(directions, gx, gy, width, height);

    float** nmSupps = fmatrix_allocate_2d(height, width);
    nonMaxSuppression(nmSupps, directions, norms, width, height);

    Recal(nmSupps, height, width);
    visualizeImage(nmSupps, NAME_IMG_SUPPRESSION, width, height);

    int hysteresisChoice;
    printf("Choose a way of doing the hysteresis.\n");
    printf("0: Queue based methode\n");
    printf("1: Recursive method\n");
    scanf("%d", &hysteresisChoice);

    float** result = fmatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            result[i][j] = 0.f;
        }
    }

    switch (hysteresisChoice)
    {
    case 0:
        queueHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    
    case 1:
        recursiveHysteresis(result, nmSupps, directions, width, height, tauL, tauH);
        break;
    }
    visualizeImage(result, NAME_IMG_CANNY, width, height);

    free_fmatrix_2d(imgReal);
    free_fmatrix_2d(imgImaginary);

    free_fmatrix_2d(gx);
    free_fmatrix_2d(gy);
    free_fmatrix_2d(norms);

    free_imatrix_2d(directions);

    free_fmatrix_2d(nmSupps);

    free_fmatrix_2d(result);
}

void doHistogramCanny() {
    int width, height;
    float sigma, var, pH;

    // Parameter input
    printf("Entrez l'écart type du filtre Gaussien: ");
    scanf("%f", &sigma);
    var = SQUARE(sigma);
    printf("Entrez la valeur de pH: ");
    scanf("%f", &pH);

    // Matrix allocation/creation
    float** imgReal = LoadImagePgm(NAME_IMG_IN, &height, &width);
    float** imgImaginary = fmatrix_allocate_2d(height, width); 

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            imgImaginary[i][j] = 0.f;
        }
    }

    applyGaussienFilter(imgReal, imgImaginary, var, width, height);

    float** gx = fmatrix_allocate_2d(height, width);
    float** gy = fmatrix_allocate_2d(height, width);
    float** norms = fmatrix_allocate_2d(height, width);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            gx[i][j] = 0.f;
            gy[i][j] = 0.f;
            norms[i][j] = 0.f;
        }
    }

    gradientFilter(imgReal, gx, gy, norms, width, height);

    visualizeImage(norms, NAME_IMG_GRADIENT, height, width);

    int** directions = imatrix_allocate_2d(height, width);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            directions[i][j] = 0;
        }
    }

    getAngles(directions, gx, gy, width, height);
    Recal(norms, height, width);

    int histChoice;
    printf("Choose histogram method\n");
    printf("0: No bins\n");
    printf("1: With bins\n");
    scanf("%d", &histChoice);

    switch (histChoice)
    {
    case 0:
        histNoBin(norms, directions, width, height, pH);
        break;
    
    case 1:
        histBin(norms, directions, width, height, pH);
        break;
    }

    free_fmatrix_2d(imgReal);
    free_fmatrix_2d(imgImaginary);

    free_fmatrix_2d(gx);
    free_fmatrix_2d(gy);
    free_fmatrix_2d(norms);

    free_imatrix_2d(directions);
}

/*
Main function
*/

int main(int argc, char** argv) {
    int methodChoice;
    printf("Choisissez la méthode du programme!\n");
    printf("0: Méthode normale\n");
    printf("1: Méthode histogramme\n");
    scanf("%d", &methodChoice);

    switch (methodChoice)
    {
    case 0:
        doNormalCanny();    
    case 1:
        doHistogramCanny();
    }
}