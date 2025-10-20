/*------------------------------------------------------*/
/* Prog    : TpIFT6150-2.c                              */
/* Auteurs : Samuel Fournier et Alexandre Toutant       */
/* Date    : 20 octobre 2025                            */
/* version : 1.0                                        */
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
int main(int argc, char** argv){
    int i, j;
    int length, width;
    float sigma = 1.0;

    /* Charger l’image */
    float** img = LoadImagePgm(NAME_IMG_IN, &length, &width);

    /* Dimensions FFT = prochaines puissances de 2 */
    int N = 1; while (N < length) N <<= 1;
    int M = 1; while (M < width)  M <<= 1;

    /* Allocation */
    float** img_re = fmatrix_allocate_2d(N, M);
    float** img_im = fmatrix_allocate_2d(N, M);
    float** ker_re = fmatrix_allocate_2d(N, M);
    float** ker_im = fmatrix_allocate_2d(N, M);



    /* ---- Étape 1.1: Flou gaussien ---- */
    /* Copier l’image dans la partie réelle (zéro-padding) */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            img_re[i][j] = (i < length && j < width) ? img[i][j] : 0.0;
            img_im[i][j] = 0.0;
        }

    /* Créer le noyau gaussien centré dans le domaine spatial */
    float var = sigma * sigma;
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            int x = (j < M / 2) ? j : j - M;
            int y = (i < N / 2) ? i : i - N;
            ker_re[i][j] = expf(-(x * x + y * y) / (2 * var));
            ker_im[i][j] = 0.0;
        }

    /* FFT 2D de l’image et du noyau */
    FFTDD(img_re, img_im, N, M);
    FFTDD(ker_re, ker_im, N, M);

    /* Multiplication spectrale : F = I * H */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            float a = img_re[i][j], b = img_im[i][j];
            float c = ker_re[i][j], d = ker_im[i][j];
            img_re[i][j] = a * c - b * d;
            img_im[i][j] = a * d + b * c;
        }

    /* FFT inverse pour revenir au domaine spatial */
    IFFTDD(img_re, img_im, N, M);

    /* Recentrer et recadrer l’image (crop au centre) */
    float** img_blur = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
            img_blur[i][j] = img_re[i][j];

    /* Recaler  */
    Recal(img_blur, length, width);



    /* ---- Étape 1.2 : Gradient avec filtres [-1, 1] ---- */
    float** grad = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float dx = (j + 1 < width)  ? (img_blur[i][j + 1] - img_blur[i][j]) : 0.0f;
            float dy = (i + 1 < length) ? (img_blur[i + 1][j] - img_blur[i][j]) : 0.0f;
            grad[i][j] = sqrtf(dx * dx + dy * dy);
        }
    }

    /* Normaliser et sauvegarder le module du gradient */
    Recal(grad, length, width);
    SaveImagePgm(NAME_IMG_GRADIENT, grad, length, width);



    /* ---- Étape 1.3 : Angle (normal aux contours) */
    float** thetaQ = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float gx = (j + 1 < width)  ? (img_blur[i][j + 1] - img_blur[i][j]) : 0.0f;
            float gy = (i + 1 < length) ? (img_blur[i + 1][j] - img_blur[i][j]) : 0.0f;

            /* angle du gradient en degrés */
            float ang = atan2f(gy, gx) * (180.0f / 3.14159265f);
            if (ang < 0.0f) ang += 180.0f;

            /* 4 directions : 0,45,90,135 */
            float q;
            if ( (ang >= 0.0f && ang < 22.5f) || (ang >= 157.5f && ang < 180.0f) )
                q = 0.0f;            /* ~ horizontal */
            else if (ang < 67.5f)
                q = 45.0f;           /* ~ diag montante */
            else if (ang < 112.5f)
                q = 90.0f;           /* ~ vertical */
            else
                q = 135.0f;          /* ~ diag descendante */

            thetaQ[i][j] = q;
        }
    }


    
    /* Libérations */
    free_fmatrix_2d(img);
    free_fmatrix_2d(img_blur);
    free_fmatrix_2d(img_re);
    free_fmatrix_2d(img_im);
    free_fmatrix_2d(ker_re);
    free_fmatrix_2d(ker_im);
    free_fmatrix_2d(grad);
    free_fmatrix_2d(thetaQ);

    return 0;
}