#include <stdio.h>

__device__ double profile(int iLayer,int nx)
{
    if(iLayer < nx-3) return 1.0;
    else return 0.0;
}

__global__ void subtractKernel(int NX,int NY,int NZ,double *drho,int iLayer)
{
    unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
    int n = (nx*(gridDim.y*blockDim.y) + ny);
     
    double t = drho[n];
    drho[n] = profile(iLayer,NX)- drho[n];
     
    printf("subtract %d before %e result %e \n",n,t,drho[n]);
}


int CUDA_WRAP_SubtractProfileFromDensity(int nx,int ny,int nz,double *drho,int iLayer)
{
    dim3 dimBlock(4, 4,1); 
    dim3 dimGrid(ny/4 ,nz/4, 1);

    //if()
    //{
       subtractKernel<<<dimGrid, dimBlock>>>(nx,ny,nz,drho,iLayer);
    //}
    
    return 0;
}