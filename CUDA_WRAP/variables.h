#ifndef CUDA_WRAP_VARIABLES_H
#define CUDA_WRAP_VARIABLES_H

extern double *CUDA_WRAP_d_rRhoBeam,*CUDA_WRAP_d_fft_of_RhoBeam;

int CUDA_WRAP_alloc_variables(int n1,int n2);

#endif