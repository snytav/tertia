#ifndef CUDA_DEBUG_DIAGNOSTIC_PRINT_H
#define CUDA_DEBUG_DIAGNOSTIC_PRINT_H

#include <cublas_v2.h>
#include <cuda.h>
#include "cutil.h"
//include <cublas.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <driver_types.h>
#include <sys/time.h>

#include "1d_batch.h"

#define N1 112
#define N2 112

typedef double MATRIX[N1][N2];
typedef double CMATRIX[N1][N2][2];

//int CUDA_DEBUG_print_error(cublasStatus_t stat,char *where);

int CUDA_WRAP_GetError();


int CUDA_DEBUG_printDhost_array(int n,double *d,char *legend);

int CUDA_DEBUG_printDhost_matrix(int n1,int n2,double *d,char *legend);

int CUDA_DEBUG_printDdevice_matrix(int n1,int n2,double *d,char *legend);

int CUDA_DEBUG_printDdevice_array(int n,double *d,char *legend);

int CUDA_DEBUG_printZdevice_array(int n,double *d,char *legend);

int CUDA_DEBUG_printZdevice_matrix(int n1,int n2,double *d,char *legend);

int VerifyComplexMatrixTransposed(int n1,int n2,CMATRIX cm,CMATRIX exact,char *s);

int VerifyComplexMatrix(int n1,int n2,CMATRIX cm,CMATRIX exact,char *s);

int VerifyComplexMatrixTransposed_fromDevice(int n1,int n2,double *d_cm,CMATRIX exact,char *s);

int VerifyComplexMatrix_fromDevice(int n1,int n2,double *d_cm,CMATRIX exact,char *s);

int CUDA_WRAP_output_fields(int n1,int n2,int iLayer,double *ex,double *ey,double *ez);
void printExplicitRho(int n1,int n2,char *legend);

int CUDA_DEBUG_printDdevice_matrixCentre(int n1,int n2,double *d,char *legend);

int CUDA_DEBUG_print3DmatrixLayer(double *d_Ey3D,int iLayer,int Ny,int Nz,char *legend);


#endif
