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
#define NAME_IMG_GRADIENT "TpIFT6150-2-gradient"
#define NAME_IMG_SUPPRESSION "TpIFT6150-2-suppression"
#define NAME_IMG_CANNY "TpIFT6150-2-canny"

typedef struct {
    float sum;          // Total sum of gradients in this bin
    int count;          // Current number of gradients in this bin
    float* content;     // Dynamically allocated array to store gradients
    int capacity;       // Current capacity of the 'content' array
} Bin;

#define NUM_BINS 100 // Use 100 bins as a constant
#define MAX_BIN_CAPACITY 10 // A reasonable guess for initial capacity

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

// Recursive method for canny filter

void follow(float** sups, int x, int y, float** orient, float** result, int width, int height, float tauL) {
    // If higher than lower bound and not a contour, then it becomes one

    if(sups[y][x] > tauL && result[y][x] == 0.f) {
        result[y][x] = 255.f;
        for(int i = 0; i < 4; i++) {
            int dir = (int)orient[y][x];
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

void recursiveHysteresis(float** result, float** sups, float** orient, int width, int height, float tauL, float tauH) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(sups[y][x] > tauH) {
                follow(sups, x, y, orient, result, width, height, tauL);
            }
        }
    }
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

/*------------------------------------------------*/
/* PROGRAMME PRINCIPAL   -------------------------*/
/*------------------------------------------------*/
int main(int argc, char** argv){
    int i, j;
    int length, width;
    float sigma;
    float p_H;

    /* ---- Entrées utilisateur ---- */
    printf("Entrez l'ecart type du filtre Gaussien: ");
    scanf("%f", &sigma);
    printf("Entrez la valeur de p_H: ");
    scanf("%f", &p_H);

    /* Charger l’image */
    float** img = LoadImagePgm(NAME_IMG_IN, &length, &width);

    /* Dimensions FFT = prochaines puissances de 2*/
    int N = 1; while (N < length) N <<= 1;
    int M = 1; while (M < width)  M <<= 1;

    /* Allocation FFT */
    float** img_re = fmatrix_allocate_2d(N, M);
    float** img_im = fmatrix_allocate_2d(N, M);
    float** ker_re = fmatrix_allocate_2d(N, M);
    float** ker_im = fmatrix_allocate_2d(N, M);
    float** ker_mod = fmatrix_allocate_2d(N, M);



    /* ---- Étape 1.1: Flou gaussien (spectral) ---- */
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            img_re[i][j] = (i < length && j < width) ? img[i][j] : 0.0f;
            img_im[i][j] = 0.0f;
        }
    }
    float var = SQUARE(sigma);
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            int x = (j < M / 2) ? j : j - M;
            int y = (i < N / 2) ? i : i - N;
            ker_re[i][j] = funcgauss2D(x, y, var);
            ker_im[i][j] = 0.0f;
        }
    }

    FFTDD(img_re, img_im, N, M);
    FFTDD(ker_re, ker_im, N, M);

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            float a = img_re[i][j], b = img_im[i][j];
            float c = ker_re[i][j], d = ker_im[i][j];
            img_re[i][j] = a * c - b * d;
            img_im[i][j] = a * d + b * c;
        }
    }

    IFFTDD(img_re, img_im, N, M);

    float** img_blur = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
            img_blur[i][j] = img_re[i][j];

    Recal(img_blur, length, width);
    visualizeImage(img_blur, "blur", length, width);

    /* ---- Étape 1.2 : Gradient avec filtres (-1, 1) ---- */
    float** dx   = fmatrix_allocate_2d(length, width);
    float** dy   = fmatrix_allocate_2d(length, width);
    float** grad = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            /* (-1,1) avant; pour la dernière col/ligne, arrière */
            float gx = (j + 1 < width)  ? (img_blur[i][j + 1] - img_blur[i][j])
                                        : (img_blur[i][j] - img_blur[i][j - 1]);
            float gy = (i + 1 < length) ? (img_blur[i + 1][j] - img_blur[i][j])
                                        : (img_blur[i][j] - img_blur[i - 1][j]);

            dx[i][j] = gx;
            dy[i][j] = gy;
            grad[i][j] = sqrtf(gx*gx + gy*gy);
        }
    }

    visualizeImage(grad, NAME_IMG_GRADIENT, length, width);

    /* ---- Étape 1.3 : Angle (0,45,90,135) à partir de dx,dy ---- */
    float** thetaQ = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float gx = dx[i][j], gy = dy[i][j];
            float ang = atan2f(gy, gx) * (180.0f / 3.14159265f);
            if (ang < 0.0f) ang += 180.0f;

            float q;
            if ((ang >= 0.0f && ang < 22.5f) || (ang >= 157.5f && ang < 180.0f)) q = 0.0f;
            else if (ang < 67.5f)  q = 45.0f;
            else if (ang < 112.5f) q = 90.0f;
            else                   q = 135.0f;

            thetaQ[i][j] = q;
        }
    }

    Recal(grad, length, width);

    float* sortedGrads = fmatrix_allocate_1d(length * width);

    int idx = 0;
    for(int i = 0; i < length; i++) {
        for(int j = 0; j < width; j++) {
            sortedGrads[idx++] = grad[i][j];
        }
    }
    
    sortList(sortedGrads, length * width);

    // 0. Find the Maximum Gradient (needed to calculate bin width)
    float g_max = 0.0f;
    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            if (grad[i][j] > g_max) {
                g_max = grad[i][j];
            }
        }
    }

    // Safety check for empty or uniform image
    if (g_max <= 0.0f) {
        printf("Erreur: Le gradient maximum est zero. Abandon.\n");
        return 1;
    }

    float bin_width = g_max / (float)NUM_BINS;
    
    // 1. Initialize Histogram Bins (Array of structs)
    Bin histogram[NUM_BINS];
    for (i = 0; i < NUM_BINS; ++i) {
        histogram[i].sum = 0.0f;
        histogram[i].count = 0;
        // Allocate initial memory for the content (will be reallocated later)
        histogram[i].capacity = MAX_BIN_CAPACITY;
        histogram[i].content = (float*)malloc(sizeof(float) * MAX_BIN_CAPACITY);
        if (histogram[i].content == NULL) {
             printf("Erreur d'allocation mémoire pour un bin.\n"); return 1;
        }
    }

    // 2. Populate Buckets
    int bin_index;
    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float g = grad[i][j];
            
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
                     printf("Erreur de réallocation mémoire pour un bin.\n"); return 1;
                }
            }

            // Store gradient, update sum and count
            histogram[bin_index].content[histogram[bin_index].count] = g;
            histogram[bin_index].sum += g;
            histogram[bin_index].count++;
        }
    }

    // 3. Find the Percentile Bin and Calculate tau_H
    int total_pixels = length * width;
    int target_count = (int)floorf((float)total_pixels * p_H);
    int cumulative_count = 0;
    int percentile_bin_index = -1;

    for (i = 0; i < NUM_BINS; ++i) {
        cumulative_count += histogram[i].count;
        if (cumulative_count >= target_count) {
            percentile_bin_index = i;
            break;
        }
    }

    float tau_h_avg = 0.0f;
    if (percentile_bin_index != -1 && histogram[percentile_bin_index].count > 0) {
        // Calculate the average of all gradients in the percentile bin
        tau_h_avg = histogram[percentile_bin_index].sum / (float)histogram[percentile_bin_index].count;
    } else {
        // Fallback: Use the max gradient if no data is found (shouldn't happen)
        tau_h_avg = g_max; 
    }

    float tau_l_avg = 0.5f * tau_h_avg;

    printf("Max Gradient: %f\n", g_max);
    printf("Calculated tau_h_avg (ph = %f): %f (from Bin %d)\n", p_H, tau_h_avg, percentile_bin_index);
    printf("Calculated tau_l_avg: %f\n", tau_l_avg);

    float tau_h_med = 0.0f;
    float tau_l_med;

    Bin binThreshold = histogram[percentile_bin_index];
    float* contents = binThreshold.content;
    int binSize = binThreshold.count;

    sortList(contents, binSize);
    tau_h_med = contents[binThreshold.count / 2];
    tau_l_med = 0.5f * tau_h_med;

    printf("Max Gradient: %f\n", g_max);
    printf("Calculated tau_h_med (ph = %f): %f (from Bin %d)\n", p_H, tau_h_med, percentile_bin_index);
    printf("Calculated tau_l_med: %f\n", tau_l_med);

    // 4. Free Histogram Memory
    for (i = 0; i < NUM_BINS; ++i) {
        free(histogram[i].content);
    }

    int indexToCut = (int)floor((float)(length * width) * p_H);

    if(indexToCut >= length * width) indexToCut = (length * width) - 1;

    float tau_h = sortedGrads[indexToCut];
    printf("Max: %f\n", sortedGrads[(length * width) - 1]);
    printf("Calculated tau_h (ph = %f): %f\n", p_H, tau_h);
    float tau_l = 0.5 * tau_h;
    printf("Calculated tau_l (ph = %f): %f\n", p_H, tau_l);

    /* ---- Étape 2 : Suppression des non-maximums ---- */
    float** supp = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float g = grad[i][j];
            float qn = 0.0f, rn = 0.0f;
            float dir = thetaQ[i][j]; /* 0, 45, 90, 135 */

            if (dir == 0.0f) {
                qn = (j-1 >= 0) ? grad[i][j-1] : 0.0f;
                rn = (j+1 <  width) ? grad[i][j+1] : 0.0f;
            } else if (dir == 90.0f) {
                qn = (i-1 >= 0) ? grad[i-1][j] : 0.0f;
                rn = (i+1 <  length) ? grad[i+1][j] : 0.0f;
            } else if (dir == 45.0f) {
                qn = (i-1 >= 0 && j+1 < width)  ? grad[i-1][j+1] : 0.0f;
                rn = (i+1 < length && j-1 >= 0) ? grad[i+1][j-1] : 0.0f;
            } else { /* 135.0f */
                qn = (i-1 >= 0 && j-1 >= 0) ? grad[i-1][j-1] : 0.0f;
                rn = (i+1 < length && j+1 < width) ? grad[i+1][j+1] : 0.0f;
            }

            supp[i][j] = (g > qn && g > rn) ? g : 0.0f;
        }
    }

    /* ---- Étape 3 : Hystérésis (tau_L/tau_H saisis) ---- */
    /* Normaliser */
    float** nms_norm = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; ++i)
        for (j = 0; j < width; ++j)
            nms_norm[i][j] = supp[i][j];
    Recal(nms_norm, length, width);

    visualizeImage(nms_norm, NAME_IMG_SUPPRESSION, N, M);


    float** imageCanny = fmatrix_allocate_2d(length, width);
    for(int i = 0; i < length; i++) {
        for(int j = 0; j < width; j++) {
            imageCanny[i][j] = 0.f;
        }
    }

    /*
    
    float** edges = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; ++i)
        for (j = 0; j < width; ++j)
            edges[i][j] = 0.0f;

    int maxq = length * width;
    int *qY = (int*)malloc(sizeof(int)*maxq);
    int *qX = (int*)malloc(sizeof(int)*maxq);
    int head = 0, tail = 0;

    
    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            if (nms_norm[i][j] > tau_h) {
                edges[i][j] = 255.0f;
                qY[tail] = i; qX[tail] = j; ++tail;
            }
        }
    }
    

    
    while (head < tail) {
        int y = qY[head], x = qX[head]; ++head;
        for (int dy8 = -1; dy8 <= 1; ++dy8) {
            for (int dx8 = -1; dx8 <= 1; ++dx8) {
                if (dy8 == 0 && dx8 == 0) continue;
                int ny = y + dy8, nx = x + dx8;
                if (ny < 0 || ny >= length || nx < 0 || nx >= width) continue;

                if (edges[ny][nx] == 0.0f && nms_norm[ny][nx] >= tau_l) {
                    edges[ny][nx] = 255.0f;
                    qY[tail] = ny; qX[tail] = nx; ++tail;
                }
            }
        }
    }
    */

    recursiveHysteresis(imageCanny, supp, thetaQ, width, length, tau_l, tau_h);

    SaveImagePgm(NAME_IMG_CANNY, imageCanny, length, width);

    /* Libérations */
    free_fmatrix_2d(img);
    free_fmatrix_2d(img_blur);
    free_fmatrix_2d(img_re);
    free_fmatrix_2d(img_im);
    free_fmatrix_2d(ker_re);
    free_fmatrix_2d(ker_im);
    free_fmatrix_2d(dx);
    free_fmatrix_2d(dy);
    free_fmatrix_2d(grad);
    free_fmatrix_2d(thetaQ);
    free_fmatrix_2d(nms_norm);
    free_fmatrix_2d(supp);
    free_fmatrix_2d(imageCanny);

    printf("\n C'est fini ... \n\n\n");
    return 0;
}
