#ifndef CUDA_WRAP_COPY_HYDRO_H
#define CUDA_WRAP_COPY_HYDRO_H

int copyBeamFourierDataToP_TEMPORARILY_FROM_HOST(int size,double *b1,double *b2,double *a1,double *a2);
int copyBeamFourierDataToP(int size,double *b1,double *b2,double *a1,double *a2);



int CUDA_WRAP_copyArraysHost(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9
);

#endif