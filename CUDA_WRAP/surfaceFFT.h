#ifndef SURFACE_FFT_H
#define SURFACE_FFT_H

#define NZ 9
#define NX 4
#define NY 4
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

int CUDA_WRAP_copyGlobalToSurfaceLayer(int n1,int n2,int layer,double *d_m,dim3 dimBlock,dim3 dimGrid);
int CUDA_WRAP_getSurfaceLayer(int n1,int n2,int layer,double *d_res,dim3 dimBlock,dim3 dimGrid);
int CUDA_WRAP_prepareFFTfromDevice(int n1,int n2,int n3,double *d_m);
int CUDA_WRAP_setGrid(int ny,int nz,int layers,dim3 &dimBlock,dim3 &dimGrid);
int CUDA_WRAP_getFlags(int **d_flagsX,int **d_flagsY,dim3 dimBlock,dim3 dimGrid,char *,char *);
int CUDA_WRAP_surfaceFFT(int n1,int n2,double *ktime,dim3 &dimBlock,dim3 &dimGrid,int *d_flagsX,int *d_flagsY,double *debug_device,double *debug_host);
int CUDA_WRAP_surfaceFFTfree();

#endif