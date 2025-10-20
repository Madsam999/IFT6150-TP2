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

/*------------------------------------------------*/
/* PROGRAMME PRINCIPAL   -------------------------*/
/*------------------------------------------------*/
int main(int argc, char** argv){
    int i, j;
    int length, width;
    float sigma = 1.0f;

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



    /* ---- Étape 1.1: Flou gaussien (spectral) ---- */
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            img_re[i][j] = (i < length && j < width) ? img[i][j] : 0.0f;
            img_im[i][j] = 0.0f;
        }

    float var = sigma * sigma;
    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            int x = (j < M / 2) ? j : j - M;
            int y = (i < N / 2) ? i : i - N;
            ker_re[i][j] = expf(-(x * x + y * y) / (2.0f * var));
            ker_im[i][j] = 0.0f;
        }

    FFTDD(img_re, img_im, N, M);
    FFTDD(ker_re, ker_im, N, M);

    for (i = 0; i < N; i++)
        for (j = 0; j < M; j++) {
            float a = img_re[i][j], b = img_im[i][j];
            float c = ker_re[i][j], d = ker_im[i][j];
            img_re[i][j] = a * c - b * d;
            img_im[i][j] = a * d + b * c;
        }

    IFFTDD(img_re, img_im, N, M);

    float** img_blur = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; i++)
        for (j = 0; j < width; j++)
            img_blur[i][j] = img_re[i][j];

    Recal(img_blur, length, width);



    /* ---- Étape 1.2 : Gradient avec filtres (-1, 1) ---- */
    float** dx   = fmatrix_allocate_2d(length, width);
    float** dy   = fmatrix_allocate_2d(length, width);
    float** grad = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            /* (-1,1) avant; pour la dernière col/ligne, on prend arrière */
            float gx = (j + 1 < width)  ? (img_blur[i][j + 1] - img_blur[i][j])
                                        : (img_blur[i][j]     - img_blur[i][j - 1]);
            float gy = (i + 1 < length) ? (img_blur[i + 1][j] - img_blur[i][j])
                                        : (img_blur[i][j]     - img_blur[i - 1][j]);

            dx[i][j] = gx;
            dy[i][j] = gy;
            grad[i][j] = sqrtf(gx*gx + gy*gy);
        }
    }

    /* Visualisation du gradient */
    float** grad_vis = fmatrix_allocate_2d(length, width);
    for (i = 0; i < length; ++i)
        for (j = 0; j < width; ++j)
            grad_vis[i][j] = grad[i][j];
    Recal(grad_vis, length, width);
    SaveImagePgm(NAME_IMG_GRADIENT, grad_vis, length, width);



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



    /* ---- Étape 2 : Suppression des non-maximums ---- */
    float** supp = fmatrix_allocate_2d(length, width);

    for (i = 0; i < length; ++i) {
        for (j = 0; j < width; ++j) {
            float g = grad[i][j];
            float qn = 0.0f, rn = 0.0f; /* voisins à comparer selon la direction */
            float dir = thetaQ[i][j]; /* 0, 45, 90, 135 */

            if (dir == 0.0f) {
                float g1 = (j-1 >= 0) ? grad[i][j-1] : 0.0f;
                float g2 = (j+1 <  width) ? grad[i][j+1] : 0.0f;
                qn = g1; rn = g2;

            } else if (dir == 90.0f) {
                float g1 = (i-1 >= 0) ? grad[i-1][j] : 0.0f;
                float g2 = (i+1 <  length) ? grad[i+1][j] : 0.0f;
                qn = g1; rn = g2;

            } else if (dir == 45.0f) {
                float g1 = (i-1 >= 0 && j+1 < width)  ? grad[i-1][j+1] : 0.0f;
                float g2 = (i+1 < length && j-1 >= 0) ? grad[i+1][j-1] : 0.0f;
                qn = g1; rn = g2;

            } else { /* 135.0f */
                float g1 = (i-1 >= 0 && j-1 >= 0) ? grad[i-1][j-1] : 0.0f;
                float g2 = (i+1 < length && j+1 < width) ? grad[i+1][j+1] : 0.0f;
                qn = g1; rn = g2;
            }

            /* Conserver seulement les maximums le long de la direction */
            supp[i][j] = (g >= qn && g >= rn) ? g : 0.0f;
        }
    }

    Recal(supp, length, width);
    SaveImagePgm(NAME_IMG_SUPPRESSION, supp, length, width);



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
    free_fmatrix_2d(grad_vis);
    free_fmatrix_2d(thetaQ);
    free_fmatrix_2d(supp);

    return 0;
}