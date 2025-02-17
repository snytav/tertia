#ifndef CUDA_WRAP_1D_BATCH_H
#define CUDA_WRAP_1D_BATCH_H




//int CUDA_WRAP_BatchOfYfourier1D_alongX_PI(int n1,int n2,CMATRIX *par_m);

int CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(int n1,int n2,double  *d_par_m);

#endif