#ifndef CUDA_WRAP_HALF_INTEGER_H
#define CUDA_WRAP_HALF_INTEGER_H

//#define DEBUG_CUDA_WRAP_HALF_INTEGER

//#define DEBUG_CUDA_WRAP_1D_FFT

int CUDA_WRAP_fourierHalfInteger2D(int n1,int n2,double *m,double *d_m,double* fres,int flagFFTW_dir1,int flagFFTW_dir2,int iLayer);

int CUDA_WRAP_fourierHalfInteger2D_fromDevice(int n1,int n2,double *m,double* fres,int flagFFTW_dir1,int flagFFTW_dir2);

void CUDA_WRAP_buffer_init(int n1,int n2);

#endif