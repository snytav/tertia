#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>

int HalfInteger(int n1,int n2,double *fft1d_tab)
{
    for(int i = 0;i < n1; i++) 
    {
       
        for(int j = 0;j < n2;j++)
	{
	    fft1d_tab[2*(i*n2+j)]   = 2.0*cos((i+0.5)*(j+0.5)*M_PI/((double)n1));
	    fft1d_tab[2*(i*n2+j)+1] = 2.0*sin((i+0.5)*(j+0.5)*M_PI/((double)n1));
//	    printf("ij %5d %5d cos %e sin %e \n",i,j,fft1d_tab[2*(i*n2+j)],fft1d_tab[2*(i*n2+j)+1]);
	}
	
    }
    return 0;    
}

//CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice
int CUDA_WRAP_create_output_surfaceCOMPLEX_fromDevice(int width,int height,int depth,surface<void,2> &surf,cudaArray *array)
{
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
        
    cudaMallocArray(&array, &channelDesc2, width, height*depth, cudaArraySurfaceLoadStore); 

    cudaBindSurfaceToArray(surf,  array); 

    return 0;
}

//CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice
int CUDA_WRAP_create3Dsurface(int width,int height,int depth,surface<void,3> &surf,cudaArray *array)
{
    cudaExtent ext;
    
    ext.depth = depth;
    ext.width = width;
    ext.height = height;
  
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
        
    cudaMalloc3DArray(&array, &channelDesc2, ext); 

    cudaBindSurfaceToArray(surf,  array); 

    return 0;
}
