#include <iostream>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>
#include <stdlib.h>

#include "mult.h"

/*
#include <cufft.h>
#include <cutil.h>
#include <cuComplex.h>
#include <fftw.h>
#include "mult.cu"
*/

#define N 4

double alphaTurn[N][2],omegaTurn[N][2];

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
    plan_big = fftw_plan_dft_1d(2*n, (fftw_complex*)in_big,       (fftw_complex*)out_big,     FFTW_FORWARD,  FFTW_ESTIMATE);
    
    fftw_execute(plan_big);
   
    for(int i = 0;i < n;i++)
    {
	res_real[i] =   out_big[2*i];
	res_imag[i] =   -out_big[2*i+1];
    }
    
    return 0;
}

int get_exp_form(int n,double*re,double *im,double *ph,double *amp)
{

    for(int i = 0;i < n;i++)
    {
        ph[i]  = atan2(im[i],re[i]);
        amp[i] = sqrt(pow(re[i],2)+pow(im[i],2));
    }

  
  return 0;
}

int phase_shift_after_pi_k_div_2N (int n,int sign,double*re,double *im,double *re_new,double *im_new)
{
    double *ph,*amp,beta;
    
    ph  = (double *)malloc(n*sizeof(double));
    amp = (double *)malloc(n*sizeof(double));

    get_exp_form(n,re,im,ph,amp);
    
    for(int i = 0;i < n;i++)
    {
        beta = i*M_PI/n/2;
        ph[i]  += sign*beta; 
    }

    for(int i = 0;i < n;i++)
    {
        re_new[i] = amp[i]*cos(ph[i]);
        im_new[i] = amp[i]*sin(ph[i]);
    }
    
    free(ph);
    free(amp);
    
    return 0;
}

double cmult_re(double a,double b,double c,double d)
{
     return (a*c-b*d);
}

double cmult_im(double a,double b,double c,double d)
{
     return (b*c+a*d);
}


int Algebraic_phase_shift_after_pi_k_div_2N (int n,int sign,double progressive,double phi,double*re,double *im,double *re_new,double *im_new)
{
    double *x,*y,beta;
    
    x  = (double *)malloc(n*sizeof(double));
    y  = (double *)malloc(n*sizeof(double));

    //get_exp_form(n,re,im,ph,amp);
    
    for(int i = 0;i < n;i++)
    {
        beta = (double)sign*(i*M_PI/n/2*progressive+phi);
        x[i] = cos(beta);
	y[i] = sin(beta);  
        //ph[i]  += sign*beta; 
    }

    for(int i = 0;i < n;i++)
    {
        re_new[i] = cmult_re(x[i],y[i],re[i],im[i]);
        im_new[i] = cmult_im(x[i],y[i],re[i],im[i]);
    }
    
    free(x);
    free(y);
    
    return 0;
}

int AlgebraicPhaseShift (int n,double *alpha,double*re,double *im,double *re_new,double *im_new)
{

    for(int i = 0;i < n;i++)
    {
        re_new[i] = cmult_re(alpha[2*i],alpha[2*i+1],re[i],im[i]);
        im_new[i] = cmult_im(alpha[2*i],alpha[2*i+1],re[i],im[i]);
    }
    
    
    return 0;
}




int phase_shift(int n,int sign,double phi,double*re,double *im,double *re_new,double *im_new)
{
    double *ph,*amp;//,beta;
    
    ph  = (double *)malloc(n*sizeof(double));
    amp = (double *)malloc(n*sizeof(double));

    get_exp_form(n,re,im,ph,amp);
    

    for(int i = 0;i < n;i++)
    {
        re_new[i] = amp[i]*cos(ph[i]+sign*phi);
        im_new[i] = amp[i]*sin(ph[i]+sign*phi);
    }

    free(ph);
    free(amp);
    
    return 0;
}

int fourierHalfInteger1D(int n,double *f,double *f_im,double *res_re,double *res_im,double *alpha,double *omega)
{
    double trans_im[1000],trans_re[1000],shift_re[1000],shift_im[1000],pi4_re[1000],pi4_im[1000];
    double cuda_re[1000],cuda_im[1000];
    
    AlgebraicPhaseShift(n,alpha,f,f_im,trans_re,trans_im);
    for(int i = 0; i < n ;i++)
    {
        printf("in %d %15.5e %15.5e %15.5e %15.5e \n",i,f[i],f_im[i],alpha[2*i],alpha[2*i+1]);
    }
//    CUDA_WRAP_vector_mult_vectors_from_host(n,alpha,f,f_im,shift_re,shift_im);
//    CUDA_WRAP_vectorZ_mult_vector_from_host(n,n,alpha,f,f_im,shift_re,shift_im);
    for(int i = 0; i < n ;i++)
    {
        printf("out %d %25.15e %25.15e CUDA %25.15e %25.15e \n",i,trans_re[i],trans_im[i],shift_re[i],shift_im[i]);
    }
   // exit(0);
    
    fourierOnePi1Dcomplex(n,trans_re,trans_im,shift_re,shift_im);
    
    AlgebraicPhaseShift(n,omega,shift_re,shift_im,res_re,res_im);
    CUDA_WRAP_vectorZ_mult_vector_from_host(n,n,omega,shift_re,shift_im,cuda_re,cuda_im);
    for(int i = 0; i < n ;i++)
    {
        printf("pre-fin %d %25.15e %25.15e CUDA %25.15e %25.15e \n",i,res_re[i],res_im[i],cuda_re[i],cuda_im[i]);
    }
    exit(0);
    
    for(int i = 0;i < n;i++)     // TO MIMIC FFTW  "DFT11" SUBROUTINES
    {
        res_re[i] *= 2.0;
        res_im[i] *= 2.0;
    }
    
    return 0;
}


/*
int main(int argc, char **argv) {
    std::cout << "Hello, Fourier!" << std::endl;
    double f[N];
    double re00[N],im00[N],re[N],im[N],res_re[N],res_im[N];
    fftw_plan plan_even,plan_odd;
    double f_im[N],even11[N],odd11[N];
    double m[N][N][2] = {{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1}};
    double r[N][N][2] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    
    CUDA_WRAP_blas_init();
    
    ComputePhaseShift(N,(double *)alphaTurn,(double *)omegaTurn);
    CUDA_WRAP_MakeTurnVectors(N,(double *)alphaTurn,(double *)omegaTurn);
    CUDA_WRAP_MakeTurnMatrices(N,N,(double *)alphaTurn,(double *)omegaTurn);
    
    CUDA_WRAP_rotateZmatrix_from_host(N,N,(double *)m,0,(double *)r);
 //   cufftHandle cu_plan;

    for(int i = 0;i < N;i++)
    {
      f[i]      = i+1; //tan(pow(sin(i+1),4.0)); //cos(1*M_PI*(i+0.5)/N);
      f_im[i]   = 0.0;
    }
    
    
    for(int i = 0;i < N;i++)
    {
      re00[i] = 0.0;
      im00[i] = 0.0;
      re[i] = 0.0;
      im[i] = 0.0;
      
      for(int j = 0 ;j < N;j++)
      {
	 re00[i] +=        f[j]*cos(2*M_PI*(i)*(j)/N);
	 im00[i] +=        f[j]*sin(2*M_PI*(i)*(j)/N);

	 re[i] +=        f[j]*cos(M_PI*(i+0.5)*(j+0.5)/N);
	 im[i] +=        f[j]*sin(M_PI*(i+0.5)*(j+0.5)/N);
      }
    }
    
    plan_even = fftw_plan_r2r_1d(N, f   ,even11, FFTW_REDFT11, FFTW_ESTIMATE);
    plan_odd  = fftw_plan_r2r_1d(N, f   ,odd11,  FFTW_RODFT11, FFTW_ESTIMATE);
    
    
    fourierHalfInteger1D(N,f,f_im,res_re,res_im,(double *)alphaTurn,(double *)omegaTurn);
    fftw_execute(plan_even);
    fftw_execute(plan_odd);
    

    
    double t,max = 0.0;
    for(int i = 0;i < N;i++)
    {
        if((t = (res_re[i]-even11[i])) > max) max = t;
        if((t = (res_im[i]-odd11[i])) > max) max = t;
    }
    printf("max %g \n",max);
    
    
    return 0;
}
*/