#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>

#include "cuda_grid.h"
#include "cuda_wrap_control.h"

#include "diagnostic_print.h"

#define MAX(a,b) ( ((a) < (b)) ? (b) : (a) )


double *bEx,*bEy,*bEz,*bBx,*bBy,*bBz;
double *maxF,*maxFback;




int alloc_first_call = 1;

int CUDA_WRAP_alloc_backup_fields(int size)
{
    if(alloc_first_call == 1) alloc_first_call = 0;
    else
    {
        return 1;
    }
      
    cudaMalloc((void **)&bEx,size*sizeof(double));
    cudaMalloc((void **)&bEy,size*sizeof(double));
    cudaMalloc((void **)&bEz,size*sizeof(double));
    cudaMalloc((void **)&bBx,size*sizeof(double));
    cudaMalloc((void **)&bBy,size*sizeof(double));
    cudaMalloc((void **)&bBz,size*sizeof(double));

    cudaMalloc((void **)&maxF,6*sizeof(double));
    cudaMalloc((void **)&maxFback,6*sizeof(double));

    return 0;
}

int CUDA_WRAP_backUpFields(int size,double *ex,double *ey,double *ez,double *hx,double *hy,double *hz)
{
    CUDA_DEBUG_printDdevice_matrix(4,4,ex,"ex"); 
  
    cudaMemcpy(bEx,ex,size*sizeof(double),cudaMemcpyDeviceToDevice);
    CUDA_DEBUG_printDdevice_matrix(4,4,bEx,"bEx"); 
    cudaMemcpy(bEy,ey,size*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(bEz,ez,size*sizeof(double),cudaMemcpyDeviceToDevice);

    cudaMemcpy(bBx,hx,size*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(bBy,hy,size*sizeof(double),cudaMemcpyDeviceToDevice);
    cudaMemcpy(bBz,hz,size*sizeof(double),cudaMemcpyDeviceToDevice);

    return 0;
}


__global__ void deviation(int ny,int nz,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz,
double *d_rEx_back,
double *d_rEy_back,
double *d_rEz_back,
double *d_rBx_back,
double *d_rBy_back,
double *d_rBz_back
)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
        int n1 = j + ny*k;
	
	
	d_rEx_back[n1] -= d_rEx[n1];
	d_rEy_back[n1] -= d_rEy[n1];
	d_rEz_back[n1] -= d_rEz[n1];
	d_rBx_back[n1] -= d_rBx[n1];
	d_rBy_back[n1] -= d_rBy[n1];
	d_rBz_back[n1] -= d_rBz[n1];
	
//	printf("%d %e %e %e %e %e %e \n",n1,d_rEx_back[n1],d_rEy_back[n1],d_rEy_back[n1],d_rBx_back[n1],d_rBy_back[n1],d_rBz_back[n1]);
}


__global__ void getMax1e5(int ny,int nz,double *ex,double *max)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
        max[0] = 1e-5;
	if(k == 0 && j == 0) 
	{
           for(int i = 0;i < ny*nz;i++)
	   {
	       if(max[0] < fabs(ex[i])) max[0] = fabs(ex[i]);
	     //  printf("%d %e\n",i,ex[i]);
	   }
	  // printf("max %e \n",max[0]);
	}
//	printf("\n qq \n");
	
}

__global__ void getMax0(int ny,int nz,double *ex,double *max)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
        max[0] = 0;
	if(k == 0 && j == 0) 
	{
           for(int i = 0;i < ny*nz;i++)
	   {
	       if(max[0] < fabs(ex[i])) max[0] = fabs(ex[i]);
	     //  printf("%d %e\n",i,ex[i]);
	   }
	  // printf("max %e \n",max[0]);
	}
//	printf("\n qq \n");
	
}


double CUDA_WRAP_getMaxDeviation(
int ny,int nz,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz  
)
{
     double f_max[6],f_max_back[6];
    // double *maxF,*maxFback;
     
     dim3 dimBlock; 
     dim3 dimGrid;
        
     getCudaGrid(ny,nz,&dimBlock,&dimGrid);
     
     
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rEx,&maxF[0]);      
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rEy,&maxF[1]);      
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rEz,&maxF[2]);      
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rBx,&maxF[3]);      
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rBy,&maxF[4]);      
     getMax1e5<<<dimGrid, dimBlock>>>(ny,nz,d_rBz,&maxF[5]);      
    
     deviation<<<dimGrid, dimBlock>>>(ny,nz,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,bEx,bEy,bEz,bBx,bBy,bBz);
    
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bEx,&maxFback[0]);      
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bEy,&maxFback[1]);      
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bEz,&maxFback[2]);      
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bBx,&maxFback[3]);      
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bBy,&maxFback[4]);      
     getMax0<<<dimGrid, dimBlock>>>(ny,nz,bBz,&maxFback[5]);      

     cudaMemcpy(f_max,maxF,6*sizeof(double),cudaMemcpyDeviceToHost);
     cudaMemcpy(f_max_back,maxFback,6*sizeof(double),cudaMemcpyDeviceToHost);
     
     return MAX(MAX(f_max_back[0]/f_max[0],MAX(f_max_back[1]/f_max[1],f_max_back[2]/f_max[2])),
                MAX(f_max_back[3]/f_max[3],MAX(f_max_back[4]/f_max[4],f_max_back[5]/f_max[5])));
}

__global__ void normalization_kernel(int ny,int nz,double inv_norm,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz,
double *d_rJx,
double *d_rJy,
double *d_rJz
)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
        int n1 = j + ny*k;
	
	d_rRho[n1] *= inv_norm;    
        d_rEx[n1] *= inv_norm;    
        d_rEy[n1] *= inv_norm;    
        d_rEz[n1] *= inv_norm;    
        d_rBx[n1] *= inv_norm;    
        d_rBy[n1] *= inv_norm;    
        d_rBz[n1] *= inv_norm;    
        d_rJx[n1] *= inv_norm;    
        d_rJy[n1] *= inv_norm;    
        d_rJz[n1] *= inv_norm;    
}

__global__ void normalizationIteratekernel(int ny,int nz,double inv_norm,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz
)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
        int n1 = j + ny*k;
	
        d_rRho[n1] *= inv_norm;    
        d_rEx[n1]  *= inv_norm;    
        d_rEy[n1]  *= inv_norm;    
        d_rEz[n1]  *= inv_norm;    
        d_rBx[n1]  *= inv_norm;    
        d_rBy[n1]  *= inv_norm;    
        d_rBz[n1]  *= inv_norm;    
}

int normalizationLoop(
int ny,
int nz,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz,
double *d_rJx,
double *d_rJy,
double *d_rJz
)
{
    dim3 dimBlock; 
    dim3 dimGrid;
    double inv_norm = 1.0/(4.0*ny*nz);
    
    getCudaGrid(ny,nz,&dimBlock,&dimGrid);    
    timeBegin(5);
    normalization_kernel<<<dimGrid, dimBlock>>>(ny,nz,inv_norm,d_rRho,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,d_rJx,d_rJy,d_rJz);
    timeEnd(5);

    return 0;
}

int normalizationIterateLoop(
int ny,
int nz,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz
)
{
    dim3 dimBlock; 
    dim3 dimGrid;
    double inv_norm = 1.0/(4.0*ny*nz);
    
    getCudaGrid(ny,nz,&dimBlock,&dimGrid);    
    timeBegin(5);
    normalizationIteratekernel<<<dimGrid, dimBlock>>>(ny,nz,inv_norm,d_rRho,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz);
    timeEnd(5);

    return 0;
}

