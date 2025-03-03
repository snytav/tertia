//////////////////////////////////////////////////////
//2D multiple FFT based on 2D surfaces. 
//////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
//#include <cutil.h>
#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include "cuda_wrap_control.h"
#include "cuda_wrap_vector_list.h"
//#include <fftw3.h>

#include "fft_matrix.h"

#include "diagnostic_print.h"

//#include "cuPrintf.cu"

#include "surfaceFFT.h"
#include "surf.h"

/*
#define NZ 2
#define NX 80
#define NY 80
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
*/

double *in_surface,*out_surfaceFFT;
double *       cuFFT_InputArray;
double *cuFFT_OutputArray;
double *alpha_surface;
int alpha_size;
double        *cuFFT_AlphaArray;
double *surf3D;
double *cu3DArray;



// __device__ void surf2Dread
// (double *x_re,
//                            double *in_surfaceT,
//                            int nx,int ny,
//                            int ny1)
// {
//          double t = in_surfaceT[nx*ny1 + ny];
//          *x_re = t;
// }
//
// __device__ void surf2Dwrite
// (
//                            double *in_surfaceT,
//                            int nx,int ny,
//                            int ny1,
//                            double t
//  )
// {
//          in_surfaceT[nx*ny1 + ny] = t;
// //          *x_re = t;
// }


__global__ void outKernel(int height,double *in_surface)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int nz = threadIdx.z;
	double x_re;
	
	//printf("grid %d %d %d dim %d %d %d\n",blockIdx.x,blockIdx.y,blockIdx.z, blockDim.x, blockDim.y, blockDim.z);
    surf2Dread(&x_re,  in_surface, nx, ny+height*nz,height);
//	cuPrintf("inKERNEL %d %d %d %e\n",nx,ny,nz,x_re);
}

__global__ void resultKernel(int height,double *in_surface)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int nz = threadIdx.z;
	double x_re;
        surf2Dread(&x_re,  in_surface, nx, ny+nz*height,height);
	
	//printf("nx %d ny %d nz %d %f\n",nx,ny,nz,x_re);
//	cuPrintf("RESULT nx %d ny %d nz %d %e\n",nx,ny,nz,x_re);
}

__global__ void inSurfaceToGlobal(int height,double *d_m,int layer,
                                  double *in_surface
                                  )
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re;
	
        surf2Dread(&x_re,  in_surface, nx * 8, ny+height*layer,height);
	d_m[nx*height + ny] = x_re;
	//cuPrintf("RESULT %d %d %e\n",nx,ny,x_re);
}

__global__ void globalToSurface(int height,double *d_m,int layer,double *in_surface)
{
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
//	unsigned int nz = threadIdx.z;
	double x_re;
	
	x_re = d_m[nx*height + ny];
	surf2Dwrite(in_surface,nx, ny+height*layer,height,x_re);
}



__global__ void outKernelAlpha(double *alpha_surface,int alpha_size)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;
        surf2Dread(&x_re,  alpha_surface, 2*ny,     nx,alpha_size);
        surf2Dread(&x_im,  alpha_surface, (2*ny+1), nx,alpha_size);
//	cuPrintf("outALPHA nx %d ny %d %e %e\n",nx,ny,x_re,x_im);
}


__global__ void fft1D_X(int height,int *odd_layers,int alpha_size,
                        double *in_surface,
                        double *out_surface,
                        double *alpha_surface)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int nz = threadIdx.z,shift,odd;
	double x_re,t = 0.0;
	double surf_alpha_re;
	
	odd = odd_layers[nz];
	
	//cuPrintf("blockDim %d %d %d blockIdx  %d %d %d thread %d %d %d \n",blockDim.x,blockDim.y,blockDim.z,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);
	
	shift = height*nz;

	for(int i = 0;i < height;i++)
	{
           surf2Dread(&x_re,  in_surface, i, ny+shift,height);
           surf2Dread(&surf_alpha_re,  alpha_surface, (2*nx+odd),i,alpha_size);
   	   t += x_re*surf_alpha_re; 
	   //printf("i %d ny %d nz %d value %e \n",i,ny,nz,x_re);
	}
	surf2Dwrite(out_surface,nx,ny+shift,height,t);
}

__global__ void fft1D_Y(int width,int height,int *odd_layers,int alpha_size,
                        double *alpha_surface,double *in_surface,double *out_surface)
{
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int nz = threadIdx.z,shift,odd;
	double x_re,t = 0.0;
	double surf_alpha_re;
	
	odd = odd_layers[nz];
	
	//printf("nz %d height %d width %d \n",nz,height,width);
	
	shift = height*nz;
	//printf("nz %d height %d width %d shift %d \n",nz,height,width,shift);
	
	for(int i = 0;i < height;i++)
	{
           surf2Dread(&x_re,  out_surface, nx, i+shift,height);
           surf2Dread(&surf_alpha_re,  alpha_surface, (2*i+odd),i+shift,alpha_size);
   	   t += x_re*surf_alpha_re; 
	  // printf("nx %d ny %d (2*i+odd) %d i+shift %d\n",nx,ny,(2*i+odd),i+shift);
	}
	
	//surf2Dwrite(t,in_surface,ny*8,nx+shift);
	surf2Dwrite(in_surface,ny,nx+shift,height,t);
	//printf("FINAL 1DY ny*8 %d nx %d shift %d nx+shift %d \n",ny*8,nx,shift, nx+shift);
}





//CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice
int CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice(int width,int height,int depth,double *d_data_in)
{
    int size = depth*width*height*sizeof(double);
    double *h_def = (double *)malloc(size);
    
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
        
    //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
    cudaMalloc(&cuFFT_InputArray,  size); //cudaArraySurfaceLoadStore);

    for(int i = 0;i < width;i++)
    {
       for(int j = 0;j < depth*height;j++)
       {
	   h_def[i*height*depth+j] = -1e6;
       }
    }
        
    cudaMemcpy(cuFFT_InputArray, h_def, size, cudaMemcpyHostToDevice);
	
//     cudaBindSurfaceToArray(in_surface,  cuFFT_InputArray);
    
    return 0;
}

//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//h_data_in     - COMPLEX matrix  (in)  
int CUDA_WRAP_create_alpha_surfaceCOMPLEX(int width,int height)
{
    //THE SIZE IN double's is 2*width*height 
    double *d_phi;
    int size = 2*width*height*sizeof(double);
    double *fft1d_tab = (double *)malloc(2*width*height*sizeof(double));
    
    HalfInteger(width,height,fft1d_tab);
    
    cudaMalloc((void **)&d_phi,2*width*height*sizeof(double));
    cudaMemcpy(d_phi,fft1d_tab,2*height*width*sizeof(double),cudaMemcpyHostToDevice);
        
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
        
    //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
     cudaMalloc(&cuFFT_AlphaArray,2*width*height*sizeof(double));
     //cudaArraySurfaceLoadStore);

    cudaMemcpy(cuFFT_AlphaArray,d_phi, size, cudaMemcpyDeviceToDevice);
	
//     cudaBindSurfaceToArray(alpha_surface,  cuFFT_AlphaArray);

    return 0;
}

int CUDA_WRAP_copyGlobalToSurfaceLayer(int n1,int n2,int layer,double *d_m,dim3 dimBlock,dim3 dimGrid)
{
    int size = n1*n2*sizeof(double);
//    int numBlocksZ = dimBlock.z;
    
    cudaMemcpy(cuFFT_InputArray, d_m, size, cudaMemcpyDeviceToDevice);
    
   // dimBlock.z = 1;
    //printf("WriteBlock %d %d %d grid %d %d %d \n",dimBlock.x,dimBlock.y,dimBlock.z,dimGrid.x,dimGrid.y,dimGrid.z);
 //   write2to3D<<<dimGrid, dimBlock>>>(layer);
    
    //dimBlock.z = numBlocksZ;
    
    return 0;
}

int CUDA_WRAP_surfaceFFTfree()
{
    cudaFree(cuFFT_InputArray);
    cudaFree(cuFFT_OutputArray);
    cudaFree(cuFFT_AlphaArray);

    return 0;
}


int CUDA_WRAP_getSurfaceLayer(int n1,int n2,int layer,double *d_res,dim3 dimBlock,dim3 dimGrid)
{
    inSurfaceToGlobal<<<dimGrid, dimBlock>>>(n2,d_res,layer,in_surface);
    
    return 0;
}


int CUDA_WRAP_prepareFFTfromDevice(int n1,int n2,int n3,double *d_m)
{
//    int ny = n1,nz = n2;

    
    if (n1 < n2) alpha_size = n2;
    else alpha_size = n1;
    
    CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice(n1,n2,n3,d_m);
    CUDA_WRAP_create_output_surfaceCOMPLEX_fromDevice(n1,n2,n3,
                                                      &out_surfaceFFT,
                                                      cuFFT_OutputArray);
    CUDA_WRAP_create_alpha_surfaceCOMPLEX(alpha_size,alpha_size);//,alpha_surface,cuFFT_AlphaArray);
    
    return 0;
}

int CUDA_WRAP_setGrid(int ny,int nz,int layers,dim3 &dimBlock,dim3 &dimGrid)
{
    if(ny > THREADS_PER_BLOCK_X) 
    {
       dimBlock.x = THREADS_PER_BLOCK_X;
       dimGrid.x  = ny/THREADS_PER_BLOCK_X;
    }
    else
    {
       dimBlock.x = ny;
       dimGrid.x  = 1;
    }

    if(nz > THREADS_PER_BLOCK_Y) 
    {
       dimBlock.y = THREADS_PER_BLOCK_Y;
       dimGrid.y  = nz/THREADS_PER_BLOCK_Y;
    }
    else
    {
       dimBlock.y = nz;
       dimGrid.y  = 1;
    }
    
    dimBlock.z = layers;
  
    return 0;
}

int CUDA_WRAP_getFlags(int **d_flagsX,int **d_flagsY,dim3 dimBlock,dim3 dimGrid,char *cfX,char *cfY)
{
    cudaMalloc((void**)d_flagsX,dimBlock.z*sizeof(int));
    cudaMalloc((void**)d_flagsY,dimBlock.z*sizeof(int));
    
    //printf("dimGrid.z %d \n",dimBlock.z);
    
    int    *h_flagsX = (int *)malloc(dimBlock.z*sizeof(int)),*h_flagsY = (int *)malloc(dimBlock.z*sizeof(int));
    
    for(int i = 0;i < dimBlock.z;i++)
    {
        h_flagsX[i] = (int)(cfX[i]-'0');
        h_flagsY[i] = (int)(cfY[i]-'0');
	//printf("FLAGS %d %d %d %c %c \n",i,h_flagsX[i],h_flagsY[i],cfX[i],cfY[i]);
    }
    
    cudaMemcpy(*d_flagsX,h_flagsX,dimBlock.z*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(*d_flagsY,h_flagsY,dimBlock.z*sizeof(int),cudaMemcpyHostToDevice);    
    
    free(h_flagsX);
    free(h_flagsY);
  
    return 0;   
}


int CUDA_WRAP_surfaceFFT(int n1,int n2,double *ktime,dim3 &dimBlock,dim3 &dimGrid,int *d_flagsX,int *d_flagsY,double *debug_device,double *debug_host)
{
//    int ny = n1,nz = n2;
      
    struct timeval tv2,tv1;
    
//    cudaPrintfInit();
    double frac_ideal,frac_rude;
    
    CUDA_WRAP_compare_device_array(n1*n2,debug_host,debug_device,&frac_ideal,&frac_rude,"RhoP","in surface",DETAILS);
     outKernel<<<dimGrid, dimBlock>>>(n2,in_surface);
     
    //return 0;
    gettimeofday(&tv1,NULL);
    fft1D_X<<<dimGrid, dimBlock>>>(n2,d_flagsX,alpha_size,in_surface,out_surfaceFFT,alpha_surface);
    CUDA_WRAP_compare_device_array(n1*n2,debug_host,debug_device,&frac_ideal,&frac_rude,"RhoP","in surface1",DETAILS);
    //printf("GRID %d %d %d %d %d %d \n",dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
    //outKernel<<<dimGrid, dimBlock>>>(n2);
    fft1D_Y<<<dimGrid, dimBlock>>>(n1,n2,d_flagsY,alpha_size,alpha_surface,in_surface,out_surfaceFFT);
    CUDA_WRAP_compare_device_array(n1*n2,debug_host,debug_device,&frac_ideal,&frac_rude,"RhoP","in surface1.2",DETAILS);
    resultKernel<<<dimGrid, dimBlock>>>(n2,in_surface);
    gettimeofday(&tv2,NULL);
    CUDA_WRAP_compare_device_array(n1*n2,debug_host,debug_device,&frac_ideal,&frac_rude,"RhoP","in surface2",DETAILS);
    //puts("2=======================================================");
//    cudaPrintfDisplay(stdout, true);
//    cudaPrintfEnd();
 //   inSurfaceToGlobal<<<dimGrid, dimBlock>>>(n2,d_res);
    
    *ktime = tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
     
    return 0; 
}

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
int CUDA_WRAP_create_output_surfaceCOMPLEX_fromDevice(int width,int height,int depth,
                                                      double **surf,double *array)
{
//     cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
    //double *array;
    cudaMalloc(surf, height*depth*sizeof(double));

    cudaMemcpy(*surf, array,height*depth*sizeof(double),cudaMemcpyHostToDevice);

    return 0;
}

/*
int main()
{ 
        int ny = NX, nz = NY;
        int width = ny;
        int height = nz;
        double *h_data_in   = (double*) malloc(width*height*sizeof(double));
        double *h_data_in1  = (double*) malloc(width*height*sizeof(double));
	double *d_data_in,h_data_out[NX][NY],h_data_out1[NX][NY];
	double *d_data_out;
	double test_in[NX][NY];
	double fftw_out[NX][NY];
	double test_in1[NX][NY];
	double fftw_out1[NX][NY];
	double ktime;
	double *fft1d_tab = (double*) malloc(2*width*height*sizeof(double));
	fftw_plan plan2D,plan2D_1;
	dim3 cudaBlock,cudaGrid;
	int *d_flagsX,*d_flagsY;
        
	plan2D   = fftw_plan_r2r_2d(ny,nz,(double *)test_in, (double *)fftw_out, FFTW_RODFT11, FFTW_REDFT11, FFTW_ESTIMATE);
	plan2D_1 = fftw_plan_r2r_2d(ny,nz,(double *)test_in1, (double *)fftw_out1, FFTW_REDFT11, FFTW_RODFT11, FFTW_ESTIMATE);
//	planY = fftw_plan_r2r_1d(ny,(double *)inY, (double *)outY, FFTW_RODFT11, FFTW_ESTIMATE);
//	planX = fftw_plan_r2r_1d(ny,(double *)inX, (double *)outX, FFTW_RODFT11, FFTW_ESTIMATE);
        for(int i = 0;i < width;i++)
        {
           for(int j = 0;j < height;j++)
           {
              h_data_in[i*height + j]     =  10*i+j;
	      test_in[i][j]               =  10*i+j; 
              h_data_in1[i*height + j]    =  -10*i+j;
	      test_in1[i][j]              =  -10*i+j; 
           }
        }
        fftw_execute(plan2D);
	fftw_execute(plan2D_1);
	
        cudaMalloc((void **)&d_data_in,width*height*sizeof(double));
        cudaMalloc((void **)&d_data_out,width*height*sizeof(double));
	cudaMemcpy(d_data_in,h_data_in,width*height*sizeof(double),cudaMemcpyHostToDevice);
	
	CUDA_WRAP_prepareFFTfromDevice(width,height,NZ,d_data_in);
	CUDA_WRAP_setGrid(width,height,NZ,cudaBlock,cudaGrid);
	copyGlobalToSurfaceLayer(width,height,0,d_data_in,cudaBlock,cudaGrid);
	
	cudaMemcpy(d_data_in,h_data_in1,width*height*sizeof(double),cudaMemcpyHostToDevice);
	copyGlobalToSurfaceLayer(width,height,1,d_data_in,cudaBlock,cudaGrid);
        
	CUDA_WRAP_getFlags(&d_flagsX,&d_flagsY,cudaBlock,cudaGrid);
	
	cudaPrintfInit();
        CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(width,height,&ktime,cudaBlock,cudaGrid,d_flagsX,d_flagsY);

        cudaFreeArray(cuInputArray); 
        cudaFreeArray(cuFFT_OutputArray); 

        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
	
	getSurfaceLayer(width,height,0,d_data_out,cudaBlock,cudaGrid);
	
	
	cudaMemcpy(h_data_out,d_data_out,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	
	double t = 0.0;
	for(int j = 0;j < width;j++)
        {
           for(int i = 0;i < height;i++)
           {
	       if(fabs(fftw_out[i][j]-h_data_out[i][j]) > t) t = fabs(fftw_out[i][j]-h_data_out[i][j]);
           }

        }
        getSurfaceLayer(width,height,1,d_data_out,cudaBlock,cudaGrid);
	
	cudaMemcpy(h_data_out1,d_data_out,width*height*sizeof(double),cudaMemcpyDeviceToHost);
        double t1 = 0.0;
	for(int j = 0;j < width;j++)
        {
           for(int i = 0;i < height;i++)
           {
	       if(fabs(fftw_out1[i][j]-h_data_out1[i][j]) > t1) t1 = fabs(fftw_out1[i][j]-h_data_out1[i][j]);
           }

        }
		
        printf("kernel time %e diff %e %e\n",ktime,t,t1);
	
        
        return 0;
}
*/
