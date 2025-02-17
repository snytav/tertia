
#include <cufft.h>
//#include <cutil.h>
#include <cuComplex.h> 

#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <driver_types.h>

#include "cuda_wrap_control.h"
#include "half_integer2D.h"


/*
#define NX 4
#define NY 4
#define BATCH 4
#define COMPARE_FFTW 1


typedef double CMATRIX[NX][NY][2];
typedef double CMATRIX2[NX][2*NY][2];
*/

cufftHandle loc_plan;
cufftDoubleComplex *loc_data = NULL,*loc_d_result = NULL ,*loc_tmp = NULL;

int batchFourierFirstCall = 1;

//double *d_cm;
/*
int fourierOnePi1Dcomplex(int n,double *f,double *f_im,double *res_real,double *res_imag)
{
    double in_big[1000],out_big[1000];
    fftw_plan plan_big;
    

    for(int i = 0;i < 4*n;i++)
    {
      in_big[i]   = 0.0;
    }
    
    for(int i = 0;i < n;i++)
    {
      in_big[2*i]   = f[i];
      in_big[2*i+1] = f_im[i];
    }
    plan_big = fftw_plan_dft_1d(n, (fftw_complex*)in_big,       (fftw_complex*)out_big,     FFTW_FORWARD,  FFTW_ESTIMATE);
    
    fftw_execute(plan_big);
   
    for(int i = 0;i < n;i++)
    {
	res_real[i] =   out_big[2*i];
	res_imag[i] =   out_big[2*i+1];
    }
    
    return 0;
}

int BatchOfYfourier1D_alongX(int n1,int n2,CMATRIX *m)
{
    double shift_im[1000],shift_re[1000]; 
     double f[1000],f_im[1000];
     
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = (*m)[i][j][0];
	    f_im[j] = (*m)[i][j][1];
        }
        fourierOnePi1Dcomplex(n2,f,f_im,shift_re,shift_im);
    
     //   phase_shift_after_pi_k_div_2N(n2,1,shift_re,shift_im,pi4_re,pi4_im);
    
 //       phase_shift(n2,1,M_PI/4/n2,pi4_re,pi4_im,res_re,res_im);
//        phase_shift(n2,1,M_PI/4/n2,shift_re,shift_im,res_re,res_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////
        for(int j = 0;j < n2;j++)
        {
	    (*m)[i][j][0] = shift_re[j];
	    (*m)[i][j][1] = shift_im[j];
        }
    }
    return 0;
}

int CUDA_WRAP_BatchOfYfourier1D_alongX(int n1,int n2,CMATRIX *par_m)
{
   cufftHandle loc_plan;
   cufftDoubleComplex *loc_data,*loc_d_result; 
   
   cudaMalloc((void **)&loc_data,sizeof(cufftDoubleComplex)*NX*NY);
   cudaMalloc((void **)&loc_d_result,sizeof(cufftDoubleComplex)*NX*NY);
   

   cudaMemcpy(loc_data,(void*)par_m,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyHostToDevice);
   cufftPlan1d(&loc_plan,NY,CUFFT_Z2Z,NX);
   
   cufftExecZ2Z(loc_plan,loc_data,loc_d_result,CUFFT_FORWARD);
   
   cudaMemcpy((void*)par_m,loc_d_result,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyDeviceToHost);
      
   return 0;
}
*/
//////////////////// END 2pi FOURIER ///////////////////////////////////////////////////////
int fourierOnePi1Dcomplex_PI(int n,double *f,double *f_im,double *res_real,double *res_imag)
{
    double in_big[1000],out_big[1000];
    fftw_plan plan_big;
    

    for(int i = 0;i < 4*n;i++)
    {
      in_big[i]   = 0.0;
    }
    
    for(int i = 0;i < n;i++)
    {
      in_big[2*i]   = f[i];
      in_big[2*i+1] = f_im[i];
    }
    plan_big = fftw_plan_dft_1d(2*n, (fftw_complex*)in_big,       (fftw_complex*)out_big,     FFTW_FORWARD,  FFTW_ESTIMATE);
    
    fftw_execute(plan_big);
   
    for(int i = 0;i < n;i++)
    {
	res_real[i] =   out_big[2*i];
	res_imag[i] =   out_big[2*i+1];
    }
    
    return 0;
}
/*
int BatchOfYfourier1D_alongX_PI(int n1,int n2,CMATRIX *m)
{
    double shift_im[1000],shift_re[1000]; 
     double f[1000],f_im[1000];
     
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = (*m)[i][j][0];
	    f_im[j] = (*m)[i][j][1];
        }
        fourierOnePi1Dcomplex_PI(n2,f,f_im,shift_re,shift_im);
    
     //   phase_shift_after_pi_k_div_2N(n2,1,shift_re,shift_im,pi4_re,pi4_im);
    
 //       phase_shift(n2,1,M_PI/4/n2,pi4_re,pi4_im,res_re,res_im);
//        phase_shift(n2,1,M_PI/4/n2,shift_re,shift_im,res_re,res_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////
        for(int j = 0;j < n2;j++)
        {
	    (*m)[i][j][0] = shift_re[j];
	    (*m)[i][j][1] = shift_im[j];
        }
    }
    return 0;
}
  */
int printComplexMatrixFromDevice(int n1,int n2,double *d_m,char *s)
{
   double *big;
   
   big = (double *)malloc(2*n1*n2*sizeof(double));

   cudaMemcpy((void*)big,d_m,sizeof(cufftDoubleComplex)*n1*n2,cudaMemcpyDeviceToHost);
   
   for(int i = 0;i < n1;i++)
   {
      printf("%s ",s);
      for(int j = 0;j < n2;j++)
      {
          printf(" (%10.3e,%10.3e) ",big[2*(i*n2+j)],big[2*(i*n2+j)+1]);
      }
      printf("\n");
   }   
   free(big);
   
   return 0;
}

int printComplexMatrixFromDevice_asArray(int n1,int n2,double *d_m,char *s)
{
   double *big;
   
   big = (double *)malloc(2*n1*n2*sizeof(double));

   cudaMemcpy((void*)big,d_m,sizeof(cufftDoubleComplex)*n1*n2,cudaMemcpyDeviceToHost);
   
   for(int i = 0;i < n1*n2;i++)
   {
      printf("%d  (%10.3e,%10.3e) ",i,big[2*i],big[2*i+1]);
      printf("\n");
   }   
   
   free(big);
   
   return 0;
}

/*
int printComplexMatrix_batchFourier(int n1,int n2,CMATRIX cm,char *s)
{
    for(int i = 0;i < n1;i++)
    {
        printf("%s ",s);
        for(int j = 0;j < n2;j++)
	{
	    printf(" (%10.3e,%10.3e)",cm[i][j][0],cm[i][j][1]);
	}
	printf("\n"); 
    }
    
    return 0;
}*/

void __global__ add_zeros_toY(int height,double *dst,double *src)
{
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;    
        double t1,t2;
        
        t1 = src[2*(nx*height +ny)  ];
        t2 = src[2*(nx*height +ny)+1];        

        dst[2*(2*nx*height +ny)  ] =  t1;
        dst[2*(2*nx*height +ny)+1] =  t2;        
        
}

void __global__ remove_zeros_fromY(int height,double *dst,double *src)
{
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;    
        double t1,t2;
        
        t1 = src[2*(2*nx*height +ny)  ];
        t2 = src[2*(2*nx*height +ny)+1];        

        dst[2*(nx*height +ny)  ] =  t1;
        dst[2*(nx*height +ny)+1] = -t2;        
        
        
}


/*
int CUDA_WRAP_BatchOfYfourier1D_alongX_PI(int n1,int n2,CMATRIX *par_m)
{
   cufftHandle loc_plan;
   cufftDoubleComplex *loc_data,*loc_d_result,*loc_tmp;
 //  CMATRIX2 big; 
 
   printComplexMatrix_batchFourier(n1,n2,*par_m,"source "); 
 
    int ny = n1,nz = n2;
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
   
   cudaMalloc((void **)&loc_data,sizeof(cufftDoubleComplex)*2*NX*NY);
   cudaMemset(loc_data,0.0, sizeof(double)*4*NX*NY);
   cudaMalloc((void **)&loc_tmp,sizeof(cufftDoubleComplex)*2*NX*NY);
   cudaMemset(loc_tmp,0.0, sizeof(double)*4*NX*NY);
   
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_data,"matrix ini ");        
   cudaMalloc((void **)&loc_d_result,sizeof(cufftDoubleComplex)*2*NX*NY);
   //puts("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq");

   cudaMemcpy(loc_tmp,(void*)par_m,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyHostToDevice);
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_tmp,"tmp set ");
   struct timeval tv1,tv2;
   
   gettimeofday(&tv1,NULL);
   add_zeros_toY<<<dimGrid, dimBlock>>>(n2,(double*)loc_data,(double *)loc_tmp);
   gettimeofday(&tv2,NULL);
   printf("kernel time %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec));
   
                
//   printComplexMatrixFromDevice_asArray(n1,2*n2,(double *)loc_data,"matrix set ");        
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_data,"matrix set ");        
//   exit(0);
   cufftPlan1d(&loc_plan,2*NY,CUFFT_Z2Z,NX);
   
   cufftExecZ2Z(loc_plan,loc_data,loc_d_result,CUFFT_FORWARD);
   
   remove_zeros_fromY<<<dimGrid, dimBlock>>>(n2,(double *)loc_tmp,(double*)loc_d_result);
   
   cudaMemcpy((void*)par_m,loc_tmp,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyDeviceToHost);
   
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_d_result,"end cuda1pi ");        
   return 0;
}
*/

/////////// THE SAME ON DEVICE
int CUDA_DEBUG_printZdevice_matrixF(int n1,int n2,cuDoubleComplex *d,char *legend)
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


int CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(int n1,int n2,double  *d_par_m)
{
   static dim3 dimBlock,dimGrid; 

   struct timeval tv1,tv2,tv[10];
   static cufftHandle loc_plan;
   static cufftDoubleComplex *loc_data,*loc_d_result,*loc_tmp;
 //  CMATRIX2 big; 
 
   gettimeofday(&tv[0],NULL);
 
#ifdef DEBUG_CUDA_WRAP_1D_FFT
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)d_par_m,"DevFourInit ");
#endif
   
   //printComplexMatrix_batchFourier(n1,n2,*par_m,"source "); 
   
//   timeBegin(13);
   gettimeofday(&tv[1],NULL);
   
   if(batchFourierFirstCall == 1)
   {
      int ny = n1,nz = n2;
    
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
      
      batchFourierFirstCall = 0;
      
      cudaMalloc((void **)&loc_data,sizeof(cufftDoubleComplex)*2*n1*n2);
      cudaMalloc((void **)&loc_tmp,sizeof(cufftDoubleComplex)*2*n1*n2);
      cudaMalloc((void **)&loc_d_result,sizeof(cufftDoubleComplex)*2*n1*n2);
      cudaMemset(loc_data,0.0, sizeof(double)*4*n1*n2);
      cudaMemset(loc_tmp,0.0, sizeof(double)*4*n1*n2);
   
      cufftPlan1d(&loc_plan,2*n2,CUFFT_Z2Z,n1);
   }
   gettimeofday(&tv[2],NULL);
   
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   timeBegin(14);
   
#ifdef DEBUG_CUDA_WRAP_1D_FFT   
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)d_par_m,"DevFourA ");
#endif   
   
 //  cudaMalloc((void **)&loc_data,sizeof(cufftDoubleComplex)*2*n1*n2);
    
   
 //  cudaMalloc((void **)&loc_tmp,sizeof(cufftDoubleComplex)*2*n1*n2);
     
 //  cudaMemset(loc_d_result,0.0, sizeof(double)*4*n1*n2);

#ifdef DEBUG_CUDA_WRAP_1D_FFT   
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)d_par_m,"DevFourB ");
#endif   
   gettimeofday(&tv[3],NULL);
   
   //printComplexMatrixFromDevice(n1,2*n2,(double *)loc_data,"matrix ini ");        
  // cudaMalloc((void **)&loc_d_result,sizeof(cufftDoubleComplex)*2*n1*n2);
   //puts("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq");
   timeEnd(14); 
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
   
   gettimeofday(&tv1,NULL);
   cudaMemcpy(loc_tmp,(void*)d_par_m,sizeof(cufftDoubleComplex)*n1*n2,cudaMemcpyDeviceToDevice);
   gettimeofday(&tv2,NULL);
   //printf("memcpy time %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec));   
#ifdef DEBUG_CUDA_WRAP_1D_FFT   
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_tmp,"tmp set ");
#endif   

   timeBegin(15);
   gettimeofday(&tv[4],NULL);
   
#ifdef DEBUG_CUDA_WRAP_1D_FFT   
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_tmp,"loc_tmp ");
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_data,"loc_data ");
#endif

   
  // gettimeofday(&tv1,NULL);
   add_zeros_toY<<<dimGrid, dimBlock>>>(n2,(double*)loc_data,(double *)loc_tmp);
  // gettimeofday(&tv2,NULL);
  // printf("kernel time %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec));
  
#ifdef DEBUG_CUDA_WRAP_1D_FFT  
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_data,"zeros ");
#endif   

#ifdef DEBUG_CUDA_WRAP_1D_FFT
   printComplexMatrixFromDevice_asArray(n1,2*n2,(double *)loc_data,"matrix set ");        
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_data,"matrix set ");        
#endif
   gettimeofday(&tv[5],NULL);

//   exit(0);
//   cufftPlan1d(&loc_plan,2*n2,CUFFT_Z2Z,n1);
 //  timeEnd(15);
#ifdef DEBUG_CUDA_WRAP_1D_FFT
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_data,"DevFourRes -data");   
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_d_result,"DevFourRes -result000 ");
     
#endif    
   
 //  timeBegin(16);
 //  gettimeofday(&tv1,NULL);
  cufftExecZ2Z(loc_plan,loc_data,loc_d_result,CUFFT_FORWARD);
 //  gettimeofday(&tv2,NULL);
//   printf("fft time %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec));
   gettimeofday(&tv[6],NULL);

#ifdef DEBUG_CUDA_WRAP_1D_FFT
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_data,"DevFourRes -data");   
   CUDA_DEBUG_printZdevice_matrixF(n1,n2,(cuDoubleComplex *)loc_d_result,"DevFourRes -result ");
   
#endif   
   
   remove_zeros_fromY<<<dimGrid, dimBlock>>>(n2,(double *)loc_tmp,(double*)loc_d_result);
   
   cudaMemcpy((void*)d_par_m,loc_tmp,sizeof(cufftDoubleComplex)*n1*n2,cudaMemcpyDeviceToDevice);

#ifdef DEBUG_CUDA_WRAP_1D_FFT   
   printComplexMatrixFromDevice(n1,2*n2,(double *)loc_d_result,"end cuda1pi ");        
#endif
   gettimeofday(&tv[7],NULL);
   
#ifdef BATCH_FOURIER_TIME_PRINT   
     printf("\t\t\t\t init    %e\n",tv[1].tv_sec - tv[0].tv_sec + 1e-6*(tv[1].tv_usec - tv[0].tv_usec));
     printf("\t\t\t\t grid    %e\n",tv[2].tv_sec - tv[1].tv_sec + 1e-6*(tv[2].tv_usec - tv[1].tv_usec));
     printf("\t\t\t\t memset  %e\n",tv[3].tv_sec - tv[2].tv_sec + 1e-6*(tv[3].tv_usec - tv[2].tv_usec));
     printf("\t\t\t\t memcpy  %e\n",tv[4].tv_sec - tv[3].tv_sec + 1e-6*(tv[4].tv_usec - tv[3].tv_usec));   
     printf("\t\t\t\t addY    %e\n",tv[5].tv_sec - tv[4].tv_sec + 1e-6*(tv[5].tv_usec - tv[4].tv_usec));   
     printf("\t\t\t\t z2z     %e\n",tv[6].tv_sec - tv[5].tv_sec + 1e-6*(tv[6].tv_usec - tv[5].tv_usec));   
     printf("\t\t\t\t removeY %e\n",tv[7].tv_sec - tv[6].tv_sec + 1e-6*(tv[7].tv_usec - tv[6].tv_usec));   
#endif     

//   timeEnd(16);
//   timeEnd(13);
   return 0;
}

int CUDA_WRAP_FourierInit(int n1,int n2)
{

   cudaMalloc((void **)&loc_data,sizeof(cufftDoubleComplex)*2*n1*n2);
   cudaMalloc((void **)&loc_tmp,sizeof(cufftDoubleComplex)*2*n1*n2);
   cudaMalloc((void **)&loc_d_result,sizeof(cufftDoubleComplex)*2*n1*n2);
   
   cufftPlan1d(&loc_plan,2*n2,CUFFT_Z2Z,n1);
   
   
  
   return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////

/*
int main(void)
{
   double m[NX][NY][2],res[NX][NY][2],w[NX][NY][2];
   struct timeval tv1,tv2;
   
   for(int i = 0;i < NX;i++)
   {
      for(int j = 0;j < NY;j++)
      {
          m[i][j][0] = sin(100*i+j+1);
          m[i][j][1] = 0.0;
          
          w[i][j][0] = m[i][j][0];
          w[i][j][1] = m[i][j][1];
      }
   }
   
   cudaMalloc((void **)&data,sizeof(cufftDoubleComplex)*NX*NY);
   cudaMalloc((void **)&d_result,sizeof(cufftDoubleComplex)*NX*NY);
   
   puts("1");
   cudaMemcpy(data,(void*)m,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyHostToDevice);
   puts("1-copy");
   
   cufftPlan1d(&plan,NY,CUFFT_Z2Z,NX);
   
//   CUDA_WRAP_fft_r2c_2d(NX,NY,host_data,host_out ); 
   puts("2");
   
   gettimeofday(&tv1,NULL);
   cufftExecZ2Z(plan,data,d_result,CUFFT_FORWARD);
   gettimeofday(&tv2,NULL);
   
   puts("3");
   
   cudaMemcpy((void*)res,d_result,sizeof(cufftDoubleComplex)*NX*NY,cudaMemcpyDeviceToHost);
   
   //puts("4");
   for(int i = 0;i < NX;i++)
   {
      for(int j = 0;j < NY;j++)
      {
         // printf("before FFTW (%10.3e,%10.3e) CUDA (%10.3e,%10.3e) \n",m[i][j][0],m[i][j][1],w[i][j][0],w[i][j][1]);
      }
   }
      
   BatchOfYfourier1D_alongX_PI(NX,NY,&m);
   CUDA_WRAP_BatchOfYfourier1D_alongX_PI(NX,NY,&w);
   

   for(int i = 0;i < NX;i++)
   {
      for(int j = 0;j < NY;j++)
      {
     //     printf("(%10.3e,%10.3e) ",res[i][j][0],res[i][j][1]);
      }
      printf("\n");
   }
   puts("FFTW====================================================");
   double t, diff = 0.0;
   for(int i = 0;i < NX;i++)
   {
      for(int j = 0;j < NY;j++)
      {
          if((t = fabs(m[i][j][0] - res[i][j][0])) > diff) diff = t;
          if((t = fabs(m[i][j][1] - res[i][j][1])) > diff) diff = t;
          
   //       printf("(%10.3e,%10.3e) ",m[i][j][0],m[i][j][1]);
      }
      printf("\n");
   }

   printf("diff %15.5e time %15.5e \n",diff,(tv2.tv_sec-tv1.tv_sec) + (tv2.tv_usec-tv1.tv_usec)*1e-6);
   
   diff = 0.0;
   //double t, diff = 0.0;
   for(int i = 0;i < NX;i++)
   {
      for(int j = 0;j < NY;j++)
      {
          if((t = fabs(m[i][j][0] - w[i][j][0])) > diff) diff = t;
          if((t = fabs(m[i][j][1] - w[i][j][1])) > diff) diff = t;
          
          printf("final FFTW (%10.3e,%10.3e) CUDA (%10.3e,%10.3e) \n",m[i][j][0],m[i][j][1],w[i][j][0],w[i][j][1]);
      }
      //printf("\n");
   }

   printf("diff %15.5e time %15.5e \n",diff,(tv2.tv_sec-tv1.tv_sec) + (tv2.tv_usec-tv1.tv_usec)*1e-6);
      
   
   puts( "Hello, Half-integer CUDA world!");

   return 0;
}


*/
