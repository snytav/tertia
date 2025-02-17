#include "variables.h"

double *CUDA_WRAP_d_rRhoBeam,*CUDA_WRAP_d_fft_of_RhoBeam;

int CUDA_WRAP_alloc_variables(int n1,int n2)
{
    cudaMalloc((void**)&CUDA_WRAP_d_rRhoBeam,n1*n2*sizeof(double));
    cudaMalloc((void**)&CUDA_WRAP_d_fft_of_RhoBeam,n1*n2*sizeof(double));
    
    
  
    return 0;
}
