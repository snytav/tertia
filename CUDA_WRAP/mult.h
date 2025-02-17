#ifndef CUDA_WRAP_MULT_H
#define CUDA_WRAP_MULT_H

#include <cublas_v2.h>
#include <cuda.h>
#include "cutil.h"
//include <cublas.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <driver_types.h>
#include <sys/time.h>

#include "diagnostic_print.h"

#define CUDA_DOUBLE_TOLERANCE 1e-12

int CUDA_WRAP_blas_init();

int CUDA_WRAP_MakeTurnVectors(int n,double *h_alpha,double *h_omega);

int CUDA_WRAP_MakeTurnMatrices(int n1,int n2,double *h_alpha,double *h_omega);

int CUDA_WRAP_vectorD_mult_vector_from_host(int n1,int n2,double *h_y1,double *h_x_re,double *h_a_re);

//COMPONENT-WISE MULTIPLICATION OF TWO COMPLEX VECTORS (A = XxY)
//PARAMETERS:
//h_y1 - COMPLEX VECTOR (Y)
//h_x_re,h_x_im _ REAL VECTORS ASSUMED TO BE REAL AND IMAGINARY PARTS OF THE X VECTOR
//h_a_re,h_a_im _ REAL VECTORS GIVING THE REAL AND IMAGINARY PARTS OF THE RESULT VECTOR A
int CUDA_WRAP_vectorZ_mult_vector_from_host(int n1,int n2,double *h_y1,double *h_x_re,double *h_x_im,double *h_a_re,double *h_a_im);

//h_m - COMPLEX MATRIX TO ROTATE
//mode - 0 FOR alpha TURN MATRIX
//       1 FOR OMEGA TURN MATRIX 
int CUDA_WRAP_rotateZmatrix_from_host(int n1,int n2,double *h_m,int mode,double *h_result);

//int CUDA_WRAP_ComputePhaseShift_onDevice(int n,double *d_alp,double *d_omg);

int CUDA_WRAP_ComputePhaseShift_onDevice(int n,double *d_alp,double *d_omg);

int CUDA_WRAP_copyMatrix_toDevice(int n1,int n2,double **d_cm,double *cm);

#endif
