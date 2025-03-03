#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
//#include <cutil.h>
#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h> 

#include "diagnostic_print.h"
#include "surf.h"

//#include "cuPrintf.cu"

#define N 4

//#include "f.cu"

//#include "../cudaParticle/cudaPIC.h"
#include "surf.h"

double *in_surface_turn,*out_surface_turn;
cudaArray       *cuInputArray,*cuOutputArray; 

double *alpha_surface_turn;
cudaArray        *cuAlphaArray; 

double *d_ctrl;

int turnFirstCall = 1;
int turnAlphaFirstCall = 1;
int turnGlobalFirstCall = 1;


// __device__ void surf2Dread
// (double *x_re,
//                            double *in_surface_turn_turnT,
//                            int nx,int ny,
//                            int NY)
// {
//          double t = in_surface_turn_turnT[nx*NY + ny];
//          *x_re = t;
// }



__device__ void cmult(double a,double b,double c, double d,double *re,double *im)
{
    *re = (a*c -b*d);
    *im = (a*d + b*c);
   // cuPrintf("cmult %e %e %e %e \n ",a,b,c,d);
}


//"height" of a matrix (number of rows)
//result - complex result matrix
// alpha - complex turn vector
__global__ void turnKernelCOMPLEX(int height,double *result,double *alpha,double *ctrl_alpha)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im,re,im;
	double surf_alpha_re,surf_alpha_im;
     
	//cuPrintf("qq \n");
	
        //CUDA array is column-major. Thus here the FIRST DIMENSION index doubled (actuall it is the SECOND)	
        surf2Dread(&x_re,  result, 2*ny, nx,height);
        surf2Dread(&x_im,  result, (2*ny+1) , nx,height);

	//cuPrintf("qq %d %d %e %e \n",nx,ny,x_re,x_im);
	
	surf2Dread(&surf_alpha_re,  result, 0,ny,height );
	surf2Dread(&surf_alpha_im,  result, 8,ny,height );
//	ctrl_alpha[2*ny]   = surf_alpha_re;//alpha[2*ny];
//	ctrl_alpha[2*ny+1] = surf_alpha_im;//alpha[2*ny+1];

	cmult(x_re,x_im,surf_alpha_re,surf_alpha_im,&re,&im);
	//cuPrintf("alp %d %d %e %e \n",nx,ny,alpha[2*ny],alpha[2*ny+1]);
	
	
  //      cuPrintf("%d %d x (%10.3e,%10.3e) alpha (%10.3e,%10.3e) surf (%10.3e,%10.3e)\n",nx,ny,x_re,x_im,alpha[2*ny],alpha[2*ny+1],surf_alpha_re,surf_alpha_im);
        
        result[2*(nx*height +ny)  ] =  re;
        result[2*(nx*height +ny)+1] =  im; 
}

//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//h_data_in     - COMPLEX matrix  (in)  
int CUDA_WRAP_create_particle_surfaceCOMPLEX(int width,int height,double *h_data_in)
{
  
    //THE SIZE IN double's is 2*width*height 
    int size = 2*width*height*sizeof(double);


    cudaMalloc(&in_surface_turn,size);
    cudaMemcpy(in_surface_turn,h_data_in,size,cudaMemcpyHostToDevice);
        
//     cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
//
//     //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
//     cudaMallocArray(&cuInputArray, &channelDesc2, 2*width, height, cudaArraySurfaceLoadStore);
//
//     cudaMemcpyToArray(cuInputArray, 0, 0,      h_data_in, size, cudaMemcpyHostToDevice);
//
//     cudaBindSurfaceToArray(in_surface_turn,  cuInputArray);

    return 0;
}

//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//h_data_in     - COMPLEX matrix  (in)  
int CUDA_WRAP_create_alpha_surface_turnCOMPLEX(int width,int height,double *d_phi)
{
  
    //THE SIZE IN double's is 2*width*height 
    int size = 2*height*sizeof(double);
        
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
    
    if(turnAlphaFirstCall)
    {
        //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
        cudaMallocArray(&cuAlphaArray, &channelDesc2, 2, height, cudaArraySurfaceLoadStore); 
	turnAlphaFirstCall = 0;
        cudaMalloc(&alpha_surface_turn,size);

    }



    cudaMemcpy(alpha_surface_turn,d_phi, size, cudaMemcpyDeviceToDevice);
	
//     cudaBindSurfaceToArray(alpha_surface_turn,  cuAlphaArray);

    return 0;
}


//CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice
int CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice(int width,int height,double *d_data_in)
{
  
    //THE SIZE IN double's is 2*width*height 
    int size = 2*width*height*sizeof(double);
        
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned); 
        
    //CUDA array is column-major. Thus here the FIRST DIMENSION is twice more (actuall it is the SECOND)
    if(turnFirstCall == 1)
    {
       cudaMallocArray(&cuInputArray, &channelDesc2, 2*width, height, cudaArraySurfaceLoadStore); 
       turnFirstCall = 0;
       cudaMalloc(&in_surface_turn,size);
    }
//     cudaMemcpyToArray(cuInputArray, 0, 0,      d_data_in, size, cudaMemcpyDeviceToDevice);
	
    cudaMemcpy(in_surface_turn,d_data_in,size,cudaMemcpyDeviceToDevice);

    return 0;
}



//width, height - THE REAL SIZE OF THE MATRIX (NOT REGARDING THAT THE ELEMENTS ARE COMPLEX) THE SIZE IN double's is 2*width*height 
//m     - COMPLEX matrix  (in)  
// ktime - kernel execution time
int CUDA_WRAP_turnMatrix_from_hostCOMPLEX(int n1,int n2,double *m,double *phi,double *ktime)
{
    double *res,*d_output_m,*d_phi;
    int ny = n1,nz = n2;
    
    res = (double *)malloc(2*n1*n2*sizeof(double));
    
    cudaMalloc((void**)&d_phi,2*n2*sizeof(double));
    cudaMemcpy(d_phi,phi,2*n2*sizeof(double),cudaMemcpyHostToDevice);
    
    CUDA_WRAP_create_particle_surfaceCOMPLEX(n1,n2,m);

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
//    cudaPrintfInit();
    
    struct timeval tv2,tv1;
    gettimeofday(&tv1,NULL);
//    turnKernelCOMPLEX<<<dimGrid, dimBlock>>>(n2,d_output_m,d_phi); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
    gettimeofday(&tv2,NULL); 
    *ktime = tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
     
  //  cudaPrintfDisplay(stdout, true);
   // cudaPrintfEnd();
    
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

int CUDA_DEBUG_printZdevice_matrixT(int n1,int n2,cuDoubleComplex *d,char *legend)
{
    double *h;
    
    h = (double *)malloc(n1*n2*sizeof(cuDoubleComplex));
    
    cudaMemcpy(h,d,n1*n2*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);

    for(int i = 0;i < n1;i++)
    {
       printf("%s %5d ",legend,i);
       for(int j = 0;j < n2;j++)
       {
           printf(" (%10.3e,%10.3e)",h[2*(i*n2+j)],h[2*(i*n2+j)+1]);
       }
       printf("\n");
    }
    free(h);
    
    return 0;
}


int CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(int n1,int n2,double *d_m,double *d_phi,double *ktime)
{
  //  double *res,*d_output_m;
    int ny = n1,nz = n2;
//    double *result;
    struct timeval tv[10];
    static double *d_ctrl;
    static dim3 dimBlock,dimGrid;
    
 //    CUDA_DEBUG_printZdevice_matrixT(n1,n2,(cuDoubleComplex *)d_m,"in turn zero ");
    
    //CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector 0 ");
    
  //  res = (double *)malloc(2*n1*n2*sizeof(double));
    
    //cudaMalloc((void**)&result,2*n1*n2*sizeof(double));
//    cudaMemcpy(d_phi,phi,2*n2*sizeof(double),cudaMemcpyHostToDevice);
     //CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector 0.1 ");   
    
  //  CUDA_WRAP_create_particle_surfaceCOMPLEX(n1,n2,m);
    //CUDA_DEBUG_printZdevice_matrixT(n1,n2,(cuDoubleComplex *)d_m,"in turn device ");
  //CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector 0.5 ");   
    
   gettimeofday(&tv[0],NULL);

   CUDA_WRAP_create_particle_surfaceCOMPLEX_fromDevice(n1,n2,d_m);
   CUDA_WRAP_create_alpha_surface_turnCOMPLEX(n1,n2,d_phi);
   
   if(turnGlobalFirstCall == 1)
   {
      cudaMalloc((void**)&d_ctrl,2*n2*sizeof(double));
      
      turnGlobalFirstCall = 0;
      
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
   }
   gettimeofday(&tv[1],NULL);
    

    
 //CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector 1 ");   
   // CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector 2 ");
//    cudaMalloc((void **)&d_output_m,2*n1*n2*sizeof(double));
//    cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
    
  //  printf("n2 %d %d %d \n",n2,dimBlock.x,dimBlock.y);
 //   CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_phi,"turn vector ");
    
    //cudaPrintfInit();
    gettimeofday(&tv[2],NULL);

    struct timeval tv2,tv1;
    //gettimeofday(&tv1,NULL);
    double fi,fw;
//    double *d_ctrl;
    gettimeofday(&tv[3],NULL);
    
    turnKernelCOMPLEX<<<dimGrid, dimBlock>>>(n2,d_m,d_phi,d_ctrl); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
    gettimeofday(&tv[4],NULL);
    
   // CUDA_DEBUG_printZdevice_matrixT(1,n2,(cuDoubleComplex *)d_ctrl,"turn surface");
    //CUDA_WRAP_compare_device_array(n2, ,d_ctrl,&fi,&fw,"angle","after kernel",DETAILS);
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    //gettimeofday(&tv2,NULL); 
    //*ktime = tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec -tv1.tv_usec);
     
 //   
  //  CUDA_DEBUG_printZdevice_matrixT(n1,n2,(cuDoubleComplex *)d_m,"turn device output ");
 //   exit(0);
//    cudaMemcpy(res,d_output_m,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
//     gettimeofday(&tv[3],NULL);
     
/*     printf("\t\t\t\t init   %e\n",tv[1].tv_sec - tv[0].tv_sec + 1e-6*(tv[1].tv_usec - tv[0].tv_usec));
     printf("\t\t\t\t grid   %e\n",tv[2].tv_sec - tv[1].tv_sec + 1e-6*(tv[2].tv_usec - tv[1].tv_usec));
     printf("\t\t\t\t malloc %e\n",tv[3].tv_sec - tv[2].tv_sec + 1e-6*(tv[3].tv_usec - tv[2].tv_usec));
     printf("\t\t\t\t kernel %e\n",tv[4].tv_sec - tv[3].tv_sec + 1e-6*(tv[4].tv_usec - tv[3].tv_usec));
*/
//    free(res);  
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
	      printf("%5d %5d %e \n",i,j,h_data_in[2*(i*height + j)]);
           }
        }

        cudaPrintfInit();
        CUDA_WRAP_turnMatrix_from_hostCOMPLEX(width,height,h_data_in,phi,&ktime);

        for(int i = 0;i < width;i++)
        {
           for(int j = 0;j < height;j++)
           {
             printf("output %d %d (%10.3e,%10.3e) \n",i,j,h_data_in[2*(i*height + j)],h_data_in[2*(i*height + j)+1]);
           }
        }

        cudaFreeArray(cuInputArray); 
        cudaFreeArray(cuOutputArray); 

        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
	
        printf("kernel time %e \n",ktime);
        
        return 0;
}
*/
