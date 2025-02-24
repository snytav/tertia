#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <fftw3.h>
#include <cuda.h>
#include "cutil.h"
#include <cuda_runtime.h>
//#include "cutil_inline.h"
#include <cuda_runtime_api.h>
#include <cuComplex.h> 


#define N 4

//#include "cuPrintf.cu"

//#include "../cudaParticle/cudaPIC.h"

double *in_surfaceT,*out_surface;
int in_surface_N;
cudaArray       *cuInputArrayTranspose;//,*cuOutputArray; 

int transposeFirstCall = 1;

__device__ void surf2Dread
(double *x_re,
                           double *in_surfaceT,
                           int nx,int ny,
                           int NY)
{
         double t = in_surfaceT[nx*NY + ny];
         *x_re = t;
}

__device__ void surf2Dwrite
(
                           double *in_surfaceT,
                           int nx,int ny,
                           int NY,
                           double t
 )
{
         in_surfaceT[nx*NY + ny] = t;
//          *x_re = t;
}



//"height" of a matrix (number of rows)
//result - complex result matrix
// alpha - complex turn vector
__global__ void transposeKernelCOMPLEX(int height,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
  
     
        //CUDA array is column-major. Thus here the FIRST DIMENSION index doubled (actuall it is the SECOND)	
        surf2Dread(&x_re,  result, 2*nx, ny,height);
        surf2Dread(&x_im,  result, (2*nx+1), ny,height);
        

//	cmult(x_re,x_im,alpha[2*ny],alpha[2*ny+1],&re,&im);
	
//        cuPrintf("%d %d x (%10.3e,%10.3e) \n",nx,ny,x_re,x_im);
        
        result[2*(nx*height +ny)  ] =  x_re;
        result[2*(nx*height +ny)+1] =  x_im;
}

//"height" of a matrix (number of rows)
//result - complex result matrix
// alpha - complex turn vector
__global__ void transposeKernelCOMPLEX_imaginaryZero(int height,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
  
     
        //CUDA array is column-major. Thus here the FIRST DIMENSION index doubled (actuall it is the SECOND)	
        surf2Dread(&x_re,  result, 2*nx , ny,height);
        surf2Dread(&x_im,  result, (2*nx+1) , ny,height);
        

//	cmult(x_re,x_im,alpha[2*ny],alpha[2*ny+1],&re,&im);
	
//        cuPrintf("trans %d %d x (%10.3e,%10.3e) \n",nx,ny,x_re,x_im);
        
        result[2*(nx*height +ny)  ] =  x_re;
        result[2*(nx*height +ny)+1] =  0.0;
}

//"height" of a matrix (number of rows)
//result - complex result matrix
// alpha - complex turn vector
__global__ void transposeKernelCOMPLEX_realZero(int height,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
  
     
        //CUDA array is column-major. Thus here the FIRST DIMENSION index doubled (actuall it is the SECOND)	
        surf2Dread(&x_re,  result, 2*nx, ny,height);
        surf2Dread(&x_im,  result, (2*nx+1), ny,height);
        

//	cmult(x_re,x_im,alpha[2*ny],alpha[2*ny+1],&re,&im);
	
//        cuPrintf("%d %d x (%10.3e,%10.3e) \n",nx,ny,x_re,x_im);
        
        result[2*(nx*height +ny)  ] =  x_im;
        result[2*(nx*height +ny)+1] =  0.0;
}


//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//h_data_in     - COMPLEX matrix  (in)  
// int CUDA_WRAP_create_particle_surfaceCOMPLEX_transpose(int width,int height,double *h_data_in)
// {
//
//     //THE SIZE IN double's is 2*width*height
//     int size = 2*width*height*sizeof(double);
//
//     cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
//
//     //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
//     cudaMalloc(&in_surfaceT, 2*width*height*sizeof(double) );
//
// //     cudaMemcpyToArray(cuInputArrayTranspose, 0, 0,      h_data_in, size, cudaMemcpyHostToDevice);
// //
// //     cudaBindSurfaceToArray(in_surfaceT,  cuInputArrayTranspose);
//
//     return 0;
// }

/*
int CUDA_WRAP_create_particle_surfaceCOMPLEX_transpose_fromDevice(int width,int height,double *d_data_in)
{
  
    //THE SIZE IN double's is 2*width*height 
    int size = 2*width*height*sizeof(double);
        
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
    
    if(transposeFirstCall == 1)
    {
       //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
       cudaMallocArray(&cuInputArrayTranspose, &channelDesc2, 2*width, height, cudaArraySurfaceLoadStore); 
       
       transposeFirstCall = 0;
    }

    cudaMemcpyToArray(cuInputArrayTranspose, 0, 0,      d_data_in, size, cudaMemcpyDeviceToDevice); 
	
    cudaBindSurfaceToArray(in_surfaceT,  cuInputArrayTranspose); 

    return 0;
}
*/


/*
//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//m     - COMPLEX matrix  (in)  
// ktime - kernel execution time
int CUDA_WRAP_transposeMatrix_from_hostCOMPLEX(int n1,int n2,double *m,double *ktime,int zero_flag,int dir2FFTW)
{
    double *res,*d_output_m;//,*d_phi;
    int ny = n1,nz = n2;
    
    res = (double *)malloc(2*n1*n2*sizeof(double));
    
//    cudaMalloc((void**)&d_phi,2*n2*sizeof(double));
//    cudaMemcpy(d_phi,phi,2*n2*sizeof(double),cudaMemcpyHostToDevice);
    
    CUDA_WRAP_create_particle_surfaceCOMPLEX_transpose(n1,n2,m);

    dim3 dimBlock; 
    dim3 dimGrid; 

    if(ny > 16) 
    {
       dimBlock.x = 16;
       dimGrid.x  = ny/16;
    }
    else
    {
       dimBlock.x = ny;
       dimGrid.x  = 1;
    }

    if(nz > 16) 
    {
       dimBlock.y = 16;
       dimGrid.y  = nz/16;
    }
    else
    {
       dimBlock.y = nz;
       dimGrid.y  = 1;
    }
    
    cudaMalloc((void **)&d_output_m,2*n1*n2*sizeof(double));
    cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
//	    printf("b kernel %d %d %e %e\n",i,j,res[2*(i*n2 + j)],res[2*(i*n2 + j)+1]); 
	}
    }    
 
    struct timeval tv2,tv1;
    
    if(zero_flag == 1)
    {
        if(dir2FFTW == FFTW_REDFT11)
	{
            transposeKernelCOMPLEX_imaginaryZero<<<dimGrid, dimBlock>>>(n2,d_output_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
	}
	else
	{
            transposeKernelCOMPLEX_realZero<<<dimGrid, dimBlock>>>(n2,d_output_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
	}
        *ktime = 0.0;//tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
    }
    else
    {
        gettimeofday(&tv1,NULL);
        transposeKernelCOMPLEX<<<dimGrid, dimBlock>>>(n2,d_output_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
        gettimeofday(&tv2,NULL); 
        *ktime = tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
    }
    
    cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
//	    printf("assign %d %d %e %e\n",i,j,res[2*(i*n2 + j)],res[2*(i*n2 + j)+1]); 
	    m[2*(i*n2 + j)  ] = res[2*(i*n2 + j)];
	    m[2*(i*n2 + j)+1] = res[2*(i*n2 + j)+1];
	}
    }
    
    free(res);  
    return 0; 
}
*/
////////////////// from DEVICE
//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//m     - COMPLEX matrix  (in)  
// ktime - kernel execution time
int CUDA_WRAP_transposeMatrix_from_deviceCOMPLEX(int n1,int n2,double *d_m,double *ktime,int zero_flag,int dir2FFTW)
{
//    double *res,*d_output_m;//,*d_phi;
    int ny = n1,nz = n2;
    struct timeval tv[10];
    
    gettimeofday(&tv[0],NULL);
    
//    res = (double *)malloc(2*n1*n2*sizeof(double));
    
//    cudaMalloc((void**)&d_phi,2*n2*sizeof(double));
//    cudaMemcpy(d_phi,phi,2*n2*sizeof(double),cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_m,sizeof(double)*n1*n2);
//     CUDA_WRAP_create_particle_surfaceCOMPLEX_transpose_fromDevice(n1,n2,d_m);
       
    gettimeofday(&tv[1],NULL);
    
    dim3 dimBlock; 
    dim3 dimGrid; 

    if(ny > 16) 
    {
       dimBlock.x = 16;
       dimGrid.x  = ny/16;
    }
    else
    {
       dimBlock.x = ny;
       dimGrid.x  = 1;
    }

    if(nz > 16) 
    {
       dimBlock.y = 16;
       dimGrid.y  = nz/16;
    }
    else
    {
       dimBlock.y = nz;
       dimGrid.y  = 1;
    }
    gettimeofday(&tv[2],NULL);
    
    //cudaMalloc((void **)&d_output_m,2*n1*n2*sizeof(double));
   // cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
/*    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
//	    printf("b kernel %d %d %e %e\n",i,j,res[2*(i*n2 + j)],res[2*(i*n2 + j)+1]); 
	}
    }    */
    gettimeofday(&tv[3],NULL);
 
    struct timeval tv2,tv1;
   // printf("dir2 %d %d \n",dir2FFTW,FFTW_REDFT11);
//    exit(0);
    //cudaPrintfInit();
    if(zero_flag == 1)
    {
        if(dir2FFTW == FFTW_REDFT11)
	{
	   // printf("near dir2 %d %d \n",dir2FFTW,FFTW_REDFT11);
            transposeKernelCOMPLEX_imaginaryZero<<<dimGrid, dimBlock>>>(n2,d_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
	}
	else
	{
            transposeKernelCOMPLEX_realZero<<<dimGrid, dimBlock>>>(n2,d_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
	}
    }
    else
    {
        //gettimeofday(&tv1,NULL);
        transposeKernelCOMPLEX<<<dimGrid, dimBlock>>>(n2,d_m); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
       // gettimeofday(&tv2,NULL); 
       // *ktime = tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
    }
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
//    exit(0);
//    cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    gettimeofday(&tv[4],NULL);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    printf("\t\t\t\t TRANS init     %e\n",tv[1].tv_sec - tv[0].tv_sec + 1e-6*(tv[1].tv_usec - tv[0].tv_usec));
    printf("\t\t\t\t TRANS grid     %e\n",tv[2].tv_sec - tv[1].tv_sec + 1e-6*(tv[2].tv_usec - tv[1].tv_usec));
    printf("\t\t\t\t TRANS empty    %e\n",tv[3].tv_sec - tv[2].tv_sec + 1e-6*(tv[3].tv_usec - tv[2].tv_usec));
    printf("\t\t\t\t TRANS kernels  %e\n",tv[4].tv_sec - tv[3].tv_sec + 1e-6*(tv[4].tv_usec - tv[3].tv_usec));       
#endif    
  //  free(res);  
    return 0; 
}


/*
int main()
{ 
        int ny = N, nz = N;
        int width = ny;
        int height = nz;
//        int size = width * height;
        double *h_data_in   = (double*) malloc(2*width*height*sizeof(double));
//        double *h_data_out  = (double*) malloc(size*sizeof(double));
	double phi[8] = {1,0,2,0,3,0,4,0};//*phi   = (double*) malloc(width*2*sizeof(double));
	double ktime;
        
        for(int i = 0;i < width;i++)
        {
           for(int j = 0;j < height;j++)
           {
              h_data_in[2*(i*height + j)]    =  i*10+j;
              h_data_in[2*(i*height + j)+1]  =  (i*10+j)*0.1;
	      printf(" (%10.3e,%10.3e)  ",h_data_in[2*(i*height + j)],h_data_in[2*(i*height + j)+1]);
           }
           printf("\n");
        }

        cudaPrintfInit();
        CUDA_WRAP_transposeMatrix_from_hostCOMPLEX(width,height,h_data_in,phi,&ktime,0);

        for(int i = 0;i < width;i++)
        {
	   printf("transpose ");
           for(int j = 0;j < height;j++)
           {
             printf(" (%10.3e,%10.3e) ",i,j,h_data_in[2*(i*height + j)],h_data_in[2*(i*height + j)+1]);
           }
           printf("\n");
        }

        cudaFreeArray(cuInputArrayTranspose); 
        cudaFreeArray(cuOutputArray); 

        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
	
        printf("kernel time %e \n",ktime);
        
        return 0;
}
*/
