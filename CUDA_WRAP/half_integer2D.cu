// TODO rearrange the CUDA grids for auxiliary kernels (getCudaGrid)

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>

#include "../run_control.h"

#include "mult.h"

#include "phase.h"

#include "half_integer1D.h"

#include "turn.h"

#include "1d_batch.h"

#include "transpose.h"

#include "diagnostic_print.h"

#include <driver_types.h>

#include "half_integer2D.h"

#include "cuda_grid.h"

#include "cuda_wrap_control.h"

//#include "cuPrintf.cu"

int d_cm_FirstCall = 1;
double *d_cm;
double *d_alp,*d_omg;


/////////////////////////////////////////    POST-SHIFT
int fourier1DalongX(int n1,int n2,CMATRIX *m)
{
//    double trans_im[1000],trans_re[1000];
     double shift_re[1000],shift_im[1000];//,pi4_re[1000],pi4_im[1000],res_re[1000],res_im[1000]; 
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
////////////////////////////////////////////////////////////////////////////////////////////////////
int fourier1DalongY(int n1,int n2,CMATRIX *cm)
{
    double f[1000],f_im[1000],shift_re[1000],shift_im[1000];
    
    for(int j = 0;j < n2;j++)
    {
        for(int i = 0;i < n1;i++)
        {
	    f[i]    = (*cm)[i][j][0];//predRe[i][j];
	    f_im[i] = (*cm)[i][j][1];//predIm[i][j];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
        //fourierHalfInteger1D(n1,f,f_im,res_re,res_im);
	
       // phase_shift_after_pi_k_div_2N(n1,-1,f,f_im,trans_re,trans_im);
    
        fourierOnePi1Dcomplex(n1,f,f_im,shift_re,shift_im);
 
        for(int i = 0;i < n1;i++)
        {
	    (*cm)[i][j][0] = shift_re[i];
	    (*cm)[i][j][1] = shift_im[i];
        }
    }
    return 0;
}

//////////////////////*****************************************************************************++
int ShiftPi4(int n1,int n2,int sign,CMATRIX *m)
{
//    double trans_im[1000],trans_re[1000],shift_re[1000],shift_im[1000],pi4_re[1000],pi4_im[1000]
     double res_re[1000],res_im[1000]; 
     double f[1000],f_im[1000];
     
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = (*m)[i][j][0];
	    f_im[j] = (*m)[i][j][1];
        }
        //fourierOnePi1Dcomplex(n2,f,f_im,shift_re,shift_im);
    
     //   phase_shift_after_pi_k_div_2N(n2,1,shift_re,shift_im,pi4_re,pi4_im);
    
 //       phase_shift(n2,1,M_PI/4/n2,pi4_re,pi4_im,res_re,res_im);
        phase_shift(n2,sign,M_PI/4/n2,f,f_im,res_re,res_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////
        for(int j = 0;j < n2;j++)
        {
	    (*m)[i][j][0] = res_re[j];
	    (*m)[i][j][1] = res_im[j];
        }
    }
    return 0;
}

int phaseShiftX(int n1,int n2,int sign,MATRIX m_real,MATRIX m_imag,MATRIX *predRe,MATRIX *predIm)
{
    double f[1000],f_im[1000],trans_re[1000],trans_im[1000];
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = m_real[i][j];
	    f_im[j] = m_imag[i][j];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
//        fourierHalfInteger1D(n2,f,f_im,res_re,res_im);
        
        phase_shift_after_pi_k_div_2N(n2,sign,f,f_im,trans_re,trans_im);
        for(int j = 0;j < n2;j++)
        {
	    (*predRe)[i][j] = trans_re[j];
	    (*predIm)[i][j] = trans_im[j];
        }
    }  
    return 0;
}

int phaseShiftXcmplx(int n1,int n2,int sign,CMATRIX *par)
{
    double f[1000],f_im[1000],trans_re[1000],trans_im[1000];
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = (*par)[i][j][0];
	    f_im[j] = (*par)[i][j][1];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
//        fourierHalfInteger1D(n2,f,f_im,res_re,res_im);
        
        phase_shift_after_pi_k_div_2N(n2,sign,f,f_im,trans_re,trans_im);
        for(int j = 0;j < n2;j++)
        {
	    (*par)[i][j][0] = trans_re[j];
	    (*par)[i][j][1] = trans_im[j];
        }
    }  
    return 0;
}

int phaseShiftY(int n1,int n2,MATRIX m_real,MATRIX m_imag,MATRIX *predRe,MATRIX *predIm)
{
    double f[1000],f_im[1000],trans_re[1000],trans_im[1000];

    for(int j = 0;j < n2;j++)
    {
        for(int i = 0;i < n1;i++)
        {
	    f[i] = m_real[i][j];
	    f_im[i] = m_imag[i][j];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
        //fourierHalfInteger1D(n1,f,f_im,res_re,res_im);
	
        phase_shift_after_pi_k_div_2N(n1,-1,f,f_im,trans_re,trans_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////

        for(int i = 0;i < n1;i++)
        {
	    (*predRe)[i][j] = trans_re[i];
	    (*predIm)[i][j] = trans_im[i];
        }
    }
    return 0;
}

int phaseShiftYcmplx(int n1,int n2,int sign,CMATRIX *m)
{
    double f[1000],f_im[1000],trans_re[1000],trans_im[1000];

    for(int j = 0;j < n2;j++)
    {
        for(int i = 0;i < n1;i++)
        {
	    f[i]    = (*m)[i][j][0];
	    f_im[i] = (*m)[i][j][1];
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
        //fourierHalfInteger1D(n1,f,f_im,res_re,res_im);
	
        phase_shift_after_pi_k_div_2N(n1,sign,f,f_im,trans_re,trans_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////

        for(int i = 0;i < n1;i++)
        {
	    (*m)[i][j][0] = trans_re[i];
	    (*m)[i][j][1] = trans_im[i];
        }
    }
    return 0;
}

/////////////////////////////////////////    POST-SHIFT
int AfterShiftX(int n1,int n2,MATRIX m_real,MATRIX m_imag,MATRIX *predRe,MATRIX *predIm)
{
//    double trans_im[1000],trans_re[1000],
     double shift_re[1000],shift_im[1000],pi4_re[1000],pi4_im[1000],res_re[1000],res_im[1000]; 
     double f[1000],f_im[1000];
     
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
	    f[j]    = m_real[i][j];
	    f_im[j] = m_imag[i][j];
        }
        fourierOnePi1Dcomplex(n2,f,f_im,shift_re,shift_im);
    
        phase_shift_after_pi_k_div_2N(n2,1,shift_re,shift_im,pi4_re,pi4_im);
    
        phase_shift(n2,1,M_PI/4/n2,pi4_re,pi4_im,res_re,res_im);
    
	/////////////////////////////////////////////////////////////////////////////////////////////
        for(int j = 0;j < n2;j++)
        {
	    (*predRe)[i][j] = res_re[j];
	    (*predIm)[i][j] = res_im[j];
        }
    }
    return 0;
}
//////////////////////*****************************************************************************++

int printComplexMatrix(int n1,int n2,CMATRIX cm,char *s)
{
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
	    printf("%s %d %d (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1]);
	}
    }
    
    return 0;
}

int CUDA_DEBUG_printZdevice_arrayM(int n,cuDoubleComplex *d,char *legend)
{
    double *h;
    
    h = (double *)malloc(n*2*sizeof(double));
    
    cudaMemcpy(h,d,2*n*sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n;i++)
    {
       printf("DEVICE: %s %5d (%10.3e,%10.3e) \n",legend,i,h[2*i],h[2*i+1]);
    }
    free(h);
    return 0;
}

int CUDA_DEBUG_printZdevice_matrixM2D(int n1,int n2,cuDoubleComplex *d,char *legend)
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

__global__ void complex2realKernel(int height,double *src,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
   
	x_re =  src[2*(nx*height +ny)];
	x_im =  src[2*(nx*height +ny)+1];
	
        result[(ny*height +nx)  ] = 4.0*x_re;
}

__global__ void complex2imagKernel(int height,double *src,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
   
	x_re =  src[2*(nx*height +ny)];
	x_im =  src[2*(nx*height +ny)+1];
	
        result[(ny*height +nx)  ] =  4.0*x_im;
}

__global__ void real2complexKernel(int height,double *src,double *result)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double x_re,x_im;//,re,im;
   
	result[2*(nx*height +ny)]   = src[nx*height +ny];
	result[2*(nx*height +ny)+1] = 0.0;
//	cuPrintf("src %e dst (%e,%e)\n",src[nx*height +ny],result[2*(nx*height +ny)],result[2*(nx*height +ny)+1]);
	
//        result[(ny*height +nx)  ] =  4.0*x_im;
}


int CUDA_WRAP_fourierHalfInteger2D_fromHost(int n1,int n2,double *m,double* fres,int flagFFTW_dir1,int flagFFTW_dir2)
{
//    MATRIX f1,predRe,predIm,zero,shRe,shIm;
    CMATRIX cm,cm1;
    double *h_cm = (double *)malloc(sizeof(double)*n1*n2*2);
    double shift_re[1000],shift_im[1000],pi4_re[1000],pi4_im[1000],res_re[1000],res_im[1000];
//    double f[1000],f_im[1000];
    double *alp,*omg,ktime;
    double *d_alp,*d_omg,*d_cm;
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER   
    puts("half-int in ");
    //exit(0);
#endif

    alp = (double*)malloc(2*n2*sizeof(double));
    omg = (double*)malloc(2*n2*sizeof(double));
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER   
    puts("half-int in 1 ");
  //  exit(0);
#endif
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
//	    zero[i][j]  = 0.0;
            double tmp =  m[i*n2 + j]; 
	    h_cm[2*(i*n2 + j)]    = tmp;
	    h_cm[2*(i*n2 + j)+1]  = 0.0;//m[i*n2 + j];
	    cm[i][j][0] = m[i*n2 + j];
	    cm1[i][j][0] = m[i*n2 + j];
	    printf("%d %e %e \n",i,h_cm[2*(i*n2 + j)],h_cm[2*(i*n2 + j)+1]);
        }
    } 
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER   
    puts("half-int in 2");
    //exit(0);
#endif
    
  //  CUDA_WRAP_copyMatrix_toDevice(n1,n2,&d_cm,(double *)h_cm);
    int err = cudaMalloc((void **)&d_cm,n1*n2*sizeof(double)*2);
    err = cudaMemcpy(d_cm,h_cm,n1*n2*sizeof(double)*2,cudaMemcpyHostToDevice);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"INITqq ");
    CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"A ");

    ComputePhaseShift(n2,(double *)alp,(double *)omg);
    CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"B ");
#endif
    
    CUDA_WRAP_ComputePhaseShift_onDevice(n2,&d_alp,&d_omg);

    CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"C ");
    
    
//////////////////////////////////////////////////////////////    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    phaseShiftXcmplx(n1,n2,-1,&cm);
    CUDA_DEBUG_printZdevice_arrayM(n2,(cuDoubleComplex *)d_alp,"turn vector before "); 
    CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"D ");
#endif    

    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_alp,&ktime);

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"compare cfDev");
    fourier1DalongX(n1,n2,&cm);
#endif    
    
    CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(n1,n2,d_cm);
    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"fourier cfDev");
    phaseShiftXcmplx(n1,n2,1,&cm);
    ShiftPi4(n1,n2,1,&cm);
#endif    

    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_omg,&ktime);

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
    VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"final devX turn");
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
        {
            cm[i][j][1] = 0.0;
        }
    }
#endif

    CUDA_WRAP_transposeMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,&ktime,1,flagFFTW_dir2);

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
    VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"transpose0 dev");
    phaseShiftYcmplx(n1,n2,-1,&cm);
    CUDA_WRAP_turnMatrix_from_hostCOMPLEX(n1,n2,(double *)cm1,alp,&ktime);
#endif
    
    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_alp,&ktime);
    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
    VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"transSHIFT ");
    fourier1DalongY(n1,n2,&cm);
#endif
    
    CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(n1,n2,d_cm);
    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"fourierY ");
    for(int j = 0;j < n2;j++)
    {
        for(int i = 0;i < n1;i++)
        {
	    shift_re[i]    = cm[i][j][0];
	    shift_im[i]    = cm[i][j][1];
        }
    
        phase_shift_after_pi_k_div_2N(n1,1,shift_re,shift_im,pi4_re,pi4_im);
    
        phase_shift(n1,1,M_PI/4/n1,pi4_re,pi4_im,res_re,res_im);

        for(int i = 0;i < n1;i++)
        {
	    fres[i*n2+j] = res_re[i];
	    cm[i][j][0] = res_re[i];
	    cm[i][j][1] = res_im[i];
	    
        }
    }
#endif    

    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_omg,&ktime);

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
    VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"final Y turn");    
#endif    

    cudaMemcpy(h_cm,d_cm,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    for(int j = 0;j < n2;j++)
    {
        for(int i = 0;i < n1;i++)
        {
	    if(flagFFTW_dir1 == FFTW_RODFT11)
	    {
	       fres[i*n2+j] = 4*h_cm[2*(j*n2+i)+1] ;  // MATRIX COMES not TRANSPOSED
	    }
	    if(flagFFTW_dir1 == FFTW_REDFT11)
	    {
	        
	       fres[i*n2+j] = 4*h_cm[2*(j*n2+i)] ;  // MATRIX COMES not TRANSPOSED
	    }
        }
    }
    
    free(alp);
    free(omg);
    
    cudaFree(d_cm);
    
    return 0;
}

int CUDA_WRAP_ComputePhaseShift_onDevice(int n,double **d_alp,double **d_omg)
{
    double beta,beta1,*alp,*omg;
    
    alp = (double *)malloc(2*n*sizeof(double));
    omg = (double *)malloc(2*n*sizeof(double));
    
    for(int i = 0;i < n;i++)
    {
        beta = -((double)i*M_PI/n/2);
        alp[2*i  ] = cos(beta);
        alp[2*i+1] = sin(beta);
	
        beta1 =  ((double)i*M_PI/n/2+M_PI/n/4);
        omg[2*i  ] = cos(beta1);
        omg[2*i+1] = sin(beta1);
	
//	printf("%d beta %e alp %e %e omg %e %e \n",i,beta,alp[2*i  ],alp[2*i +1 ],omg[2*i  ],omg[2*i + 1]);
    }
    
    int err = cudaMalloc((void **)d_alp,2*n*sizeof(double));
    //printf("cuda erroor %d \n",err);
    if(err == cudaErrorMemoryAllocation) 
    {
       puts("NOT ENOUGH MEMORY: ComputePhaseShift_onDevice");
       exit(0);
    }
    err = cudaMalloc((void **)d_omg,2*n*sizeof(double));
    //printf("cuda erroor %d \n",err);
    if(err == cudaErrorMemoryAllocation) 
    {
       puts("NOT ENOUGH MEMORY: ComputePhaseShift_onDevice");
       exit(0);
    }
    
    err = cudaMemcpy(*d_alp,alp,2*n*sizeof(double),cudaMemcpyHostToDevice);
    //printf("cuda erroor %d \n",err);
    if(err == cudaErrorMemoryAllocation) 
    {
       puts("NOT ENOUGH MEMORY: ComputePhaseShift_onDevice");
       exit(0);
    }
    err = cudaMemcpy(*d_omg,omg,2*n*sizeof(double),cudaMemcpyHostToDevice);    
    //printf("cuda erroor %d \n",err);
    if(err == cudaErrorMemoryAllocation) 
    {
       puts("NOT ENOUGH MEMORY: ComputePhaseShift_onDevice");
       exit(0);
    }
    
    free(alp);
    free(omg);
  
    return 0;
}


void CUDA_WRAP_buffer_init(int n1,int n2)
{
  cudaMalloc((void **)&d_cm,sizeof(double)*n1*n2);
  
  CUDA_WRAP_ComputePhaseShift_onDevice(n2,&d_alp,&d_omg);
}

int CUDA_WRAP_fourierHalfInteger2D(int n1,int n2,double *m,double *d_m,double* d_fres,int flagFFTW_dir1,int flagFFTW_dir2,int iLayer)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED   
    return 0;
#endif    
  
//    MATRIX f1,predRe,predIm,zero,shRe,shIm;
    CMATRIX cm,cm1;
    double *h_cm = (double *)malloc(sizeof(double)*n1*n2*2);
    double shift_re[1000],shift_im[1000],pi4_re[1000],pi4_im[1000],res_re[1000],res_im[1000];
//    double f[1000],f_im[1000];
    double *alp,*omg,ktime,*fres;
    //double *d_alp,*d_omg;
    dim3 dimBlock,dimGrid;
    struct timeval tv[HALF_INT_TN];
    
    gettimeofday(&tv[0],NULL);
    if(d_cm_FirstCall == 1)
    {
       cudaMalloc((void **)&d_cm,sizeof(double)*2*n1*n2);
       d_cm_FirstCall = 0;
    }
    
    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printDdevice_matrix(n1,n2,d_m,"entering fft"); 
    }
     
    timeBegin(19);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER   
                                                                                           puts("half-int in ");
                                                                                           //exit(0);
#endif
 //   gettimeofday(&tv[0],NULL);											   
     timeBegin(7);
    alp = (double*)malloc(2*n2*sizeof(double));
    omg = (double*)malloc(2*n2*sizeof(double));
    timeEnd(7);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER   
                                                                                           puts("half-int in 1 ");
                                                                                           //  exit(0);
#endif
    timeBegin(8);
    getCudaGrid(n1,n2,&dimBlock,&dimGrid);
    //CUDA_WRAP_copyMatrix_toDevice(n1,n2,&d_cm,(double *)h_cm);
//    cudaPrintfInit();
     gettimeofday(&tv[1],NULL);
    real2complexKernel<<<dimGrid, dimBlock>>>(n2,d_m,d_cm);
     gettimeofday(&tv[2],NULL);
  //  cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printZdevice_matrix(n1,n2,d_cm,"r2complex"); 
    }
    gettimeofday(&tv[3],NULL);
    timeEnd(8);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER      
                                                                                          for(int i = 0;i < n1;i++)
                                                                                          {
                                                                                              for(int j = 0;j < n2;j++)
                                                                                              {
                                                                                       //	  zero[i][j]  = 0.0;
                                                                                                  double tmp =  m[i*n2 + j]; 
	                                                                                          h_cm[2*(i*n2 + j)]    = tmp;
	                                                                                          h_cm[2*(i*n2 + j)+1]  = 0.0;//m[i*n2 + j];
	                                                                                          cm[i][j][0] = m[i*n2 + j];
	                                                                                          cm1[i][j][0] = m[i*n2 + j];
	                                                                                          printf("%d %e %e \n",i,h_cm[2*(i*n2 + j)],h_cm[2*(i*n2 + j)+1]);
                                                                                               }
                                                                                          } 

                                                                                           puts("half-int in 2");
                                                                                           //exit(0);
#endif
        gettimeofday(&tv[4],NULL);    
    
   // CUDA_WRAP_copyMatrix_toDevice(n1,n2,&d_cm,(double *)h_cm);
    //int err = cudaMalloc((void **)&d_cm,n1*n2*sizeof(double)*2);
    //err = cudaMemcpy(d_cm,h_cm,n1*n2*sizeof(double)*2,cudaMemcpyHostToDevice);
   // CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_m,"A ");
//   VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"INIT00 ");
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
											   
                                                                                           VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"INIT ");
                                                                                           CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"A ");

                                                                                           ComputePhaseShift(n2,(double *)alp,(double *)omg);
                                                                                           CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"B ");
#endif
    
    timeBegin(9);											   
    //CUDA_WRAP_ComputePhaseShift_onDevice(n2,&d_alp,&d_omg);
    timeEnd(9);

    
 //   gettimeofday(&tv[4],NULL);
//////////////////////////////////////////////////////////////    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
                                                                                           CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"C ");
    
                                                                                           phaseShiftXcmplx(n1,n2,-1,&cm);
											   for(int i = 0;i < n1;i++) printf("alp %d %e %e\n",i,alp[2*i],alp[2*i+1]);
                                                                                           
                                                                                           CUDA_DEBUG_printZdevice_arrayM(n2,(cuDoubleComplex *)d_alp,"turn vector before "); 
                                                                                           CUDA_DEBUG_printZdevice_matrixM2D(n1,n2,(cuDoubleComplex *)d_cm,"D ");
#endif    
    timeBegin(10);
    gettimeofday(&tv[5],NULL);
    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_alp,&ktime);
    gettimeofday(&tv[6],NULL);
    timeEnd(10);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
                                                                                           VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"compare cfDev");
                                                                                           fourier1DalongX(n1,n2,&cm);
#endif    
    											   
    timeBegin(11);
     gettimeofday(&tv[7],NULL);
    CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(n1,n2,d_cm);
     gettimeofday(&tv[8],NULL);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printZdevice_matrix(n1,n2,d_cm,"batch fft"); 
    }

    timeEnd(11);
                                                                                           VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"fourier cfDev");
                                                                                           phaseShiftXcmplx(n1,n2,1,&cm);
                                                                                           ShiftPi4(n1,n2,1,&cm);
#endif    
        gettimeofday(&tv[9],NULL);											   
    timeBegin(31); 
     gettimeofday(&tv[10],NULL);
    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_omg,&ktime);
     gettimeofday(&tv[11],NULL);
    timeEnd(31);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
                                                                                           VerifyComplexMatrix_fromDevice(n1,n2,d_cm,cm,"final devX turn");
    
                                                                                           for(int i = 0;i < n1;i++)
                                                                                           {
                                                                                              for(int j = 0;j < n2;j++)
                                                                                              {
                                                                                                  cm[i][j][1] = 0.0;
                                                                                              }
                                                                                           }
#endif

    timeBegin(12); 
     gettimeofday(&tv[12],NULL);
    CUDA_WRAP_transposeMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,&ktime,1,flagFFTW_dir2);
     gettimeofday(&tv[13],NULL);
    timeEnd(12);
            gettimeofday(&tv[14],NULL);
    
    

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        

    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printZdevice_matrix(n1,n2,d_cm,"transpose"); 
    }


                                                                                           VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"transpose0 dev");
                                                                                           phaseShiftYcmplx(n1,n2,-1,&cm);
                                                                                           CUDA_WRAP_turnMatrix_from_hostCOMPLEX(n1,n2,(double *)cm1,alp,&ktime);
#endif
    
    timeBegin(32);											   
    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_alp,&ktime);
    timeEnd(32);
    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        
                                                                                           VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"transSHIFT ");
                                                                                           fourier1DalongY(n1,n2,&cm);
#endif
            gettimeofday(&tv[15],NULL);
    timeBegin(33);											   
    CUDA_WRAP_BatchOfYfourier1D_alongX_PI_fromDevice(n1,n2,d_cm);
     gettimeofday(&tv[16],NULL);
    timeEnd(33);

    
    
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER    
    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printZdevice_matrix(n1,n2,d_cm,"batch2"); 
    }

                                                                                           VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"fourierY ");
                                                                                           for(int j = 0;j < n2;j++)
                                                                                           {
                                                                                               for(int i = 0;i < n1;i++)
                                                                                               {
	                                                                                           shift_re[i]    = cm[i][j][0];
	                                                                                           shift_im[i]    = cm[i][j][1];
                                                                                               }
    
                                                                                               phase_shift_after_pi_k_div_2N(n1,1,shift_re,shift_im,pi4_re,pi4_im);
    
                                                                                               phase_shift(n1,1,M_PI/4/n1,pi4_re,pi4_im,res_re,res_im);

                                                                                               for(int i = 0;i < n1;i++)
                                                                                               {
                                                                                        //	    fres[i*n2+j] = res_re[i];
	                                                                                            cm[i][j][0] = res_re[i];
	                                                                                            cm[i][j][1] = res_im[i];
	    
                                                                                               }
                                                                                            }
#endif    
        gettimeofday(&tv[17],NULL);
    timeBegin(34);
    CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(n1,n2,(double *)d_cm,d_omg,&ktime);
     gettimeofday(&tv[18],NULL);
    timeEnd(34);

    

#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER        

    if(iLayer == CONTROL_FFT_LAYER)
    {
       CUDA_DEBUG_printZdevice_matrix(n1,n2,d_cm,"after turn2"); 
    }


                                                                                            VerifyComplexMatrixTransposed_fromDevice(n1,n2,d_cm,cm,"final Y turn");    
#endif    
           gettimeofday(&tv[19],NULL);
                                                                                            //    double *d_fres;
    
   // cudaMalloc((void **)&d_fres,n1*n2*sizeof(double));
    timeBegin(35);
    getCudaGrid(n1,n2,&dimBlock,&dimGrid);

    if(flagFFTW_dir1 == FFTW_RODFT11)
    {
        complex2imagKernel<<<dimGrid, dimBlock>>>(n2,d_cm,d_fres);
    }
    if(flagFFTW_dir1 == FFTW_REDFT11)
    {
        complex2realKernel<<<dimGrid, dimBlock>>>(n2,d_cm,d_fres);
    }
    timeEnd(35);
     gettimeofday(&tv[20],NULL);
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER      
											    cudaMemcpy(h_cm,d_cm,2*n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
                                                                                            fres = (double*)malloc(n1*n2*sizeof(double));
                                                                                            for(int j = 0;j < n2;j++)
                                                                                            {
                                                                                                for(int i = 0;i < n1;i++)
                                                                                                {
	                                                                                            if(flagFFTW_dir1 == FFTW_RODFT11)
	                                                                                            {
	                                                                                               fres[i*n2+j] = 4*h_cm[2*(j*n2+i)+1] ;  // MATRIX COMES not TRANSPOSED
	                                                                                            }
	                                                                                            if(flagFFTW_dir1 == FFTW_REDFT11)
	                                                                                            {
            	                                                                                       fres[i*n2+j] = 4*h_cm[2*(j*n2+i)] ;  // MATRIX COMES not TRANSPOSED
	                                                                                            }
                                                                                                }
                                                                                            }
                                                                                            cudaMemcpy(h_cm,d_fres,n1*n2*sizeof(double),cudaMemcpyDeviceToHost); 
                                                                                            double dmax = 0.0,t;
                                                                                            for(int j = 0;j < n2;j++)
                                                                                            {
                                                                                                for(int i = 0;i < n1;i++)
                                                                                                {
	                                                                                            if((t = fabs(h_cm[i*n2+j]- fres[i*n2+j])) > dmax) dmax = t;
                                                                                                }
                                                                                            }    
                                                                                            printf("diff %e \n",dmax);
#endif    
    
    free(alp);
    free(omg);
    gettimeofday(&tv[21],NULL); 
   // cudaFree(d_cm);
    timeEnd(19);
//
#ifdef DEBUG_CUDA_WRAP_HALF_INTEGER
    
   double sum = 0.0;
   for(int i = 1;i < HALF_INT_TN;i++)
   {
       double t = tv[i].tv_sec - tv[i-1].tv_sec + 1e-6*(tv[i].tv_usec - tv[i-1].tv_usec);
       sum += t;
       printf("                             2Dstage %2d time %15.5e total %15.5e \n",i,t,sum);  
   }  
#endif   
    return 0;
}

