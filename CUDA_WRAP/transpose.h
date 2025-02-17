#ifndef CUDA_WRAP_TRANSPOSE_H
#define CUDA_WRAP_TRANSPOSE_H

int CUDA_WRAP_transposeMatrix_from_hostCOMPLEX(int n1,int n2,double *m,double *ktime,int zero_flag);

int CUDA_WRAP_transposeMatrix_from_deviceCOMPLEX(int n1,int n2,double *d_m,double *ktime,int zero_flag,int dir2FFTW);

#endif