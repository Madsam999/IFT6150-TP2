/*------------------------------------------------------*/
/* Prog    : TpIFT6150-2.c                              */
/* Auteur  :                                            */
/* Date    :                                            */
/* version :                                            */ 
/* langage : C                                          */
/* labo    : DIRO                                       */
/*------------------------------------------------------*/

/*------------------------------------------------*/
/* FICHIERS INCLUS -------------------------------*/
/*------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "FonctionDemo2.h"

/*------------------------------------------------*/
/* DEFINITIONS -----------------------------------*/                              
/*------------------------------------------------*/
#define NAME_IMG_IN  "photograph"
#define NAME_IMG_GRADIENT "TpIFT6150-2-gradient"
#define NAME_IMG_SUPPRESSION "TpIFT6150-2-suppression"
#define NAME_IMG_CANNY "TpIFT6150-2-canny"

/*------------------------------------------------*/
/* PROGRAMME PRINCIPAL   -------------------------*/                     
/*------------------------------------------------*/

void applyGaussianFilter(float** matReal, float sigma, int width, int height) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            matReal[y][x] = funcgauss2D(x, y, sigma);
        }
    }
}

void center(float** matReal, int width, int height) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            matReal[y][x] *= pow(-1, x + y);
        }
    }
}

int main(int argc, int** argv) {
    int width, height;
    float tau_L, tau_H;
    float p_H;
    float sigma; //skibidi

    printf("Entrez la valeur de tau_L: ");
    scanf("%f", &tau_L);
    printf("Entrez la valeur de tau_H: ");
    scanf("%f", &tau_H);
    printf("Entrez l'écart type du filtre Gaussien: ");
    scanf("%f", &sigma);
    printf("Entrez la valeure de p_H: ");
    scanf("%f", &p_H);

    float** matReal;
    float** matImaginary;
    float** matModule;

    matReal = LoadImagePgm(NAME_IMG_IN, &height, &width);
    matImaginary = fmatrix_allocate_2d(height, width);
    matModule = fmatrix_allocate_2d(height, width);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            matImaginary[i][j] = 0.f;
            matModule[i][j] = 0.f;
        }
    }

    center(matReal, width, height);

    FFTDD(matReal, matImaginary, height, width);
    Mod(matModule, matReal, matImaginary, height, width);

    Recal(matModule, height, width);
    Mult(matModule, 100, height, width);

    SaveImagePgm("Module", matModule, height, width);

    free_fmatrix_2d(matReal);
    free_fmatrix_2d(matImaginary);
    free_fmatrix_2d(matModule);
    
}