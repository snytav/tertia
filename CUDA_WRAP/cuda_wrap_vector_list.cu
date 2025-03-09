#include "cuda_wrap_vector_list.h"
#include "diagnostic_print.h"
#include "../run_control.h"

#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
//#include <cutil.h>



       double *d_fft_of_Ex, *d_fft_of_Ey, *d_fft_of_Ez, *d_fft_of_Bx, *d_fft_of_By, *d_fft_of_Bz,
              *d_fft_of_Jx, *d_fft_of_Jy, *d_fft_of_Jz, *d_fft_of_Rho,
              *d_fft_of_JxP, *d_fft_of_JyP, *d_fft_of_JzP, *d_fft_of_RhoP,
              *d_fft_of_JxBeam, *d_fft_of_JyBeam, *d_fft_of_JzBeam, *d_fft_of_RhoBeam,
              *d_fft_of_JxBeamP, *d_fft_of_JyBeamP, *d_fft_of_JzBeamP, *d_fft_of_RhoBeamP,
              *d_fft_of_ExRho, *d_fft_of_EyRho, *d_fft_of_EzRho;
	      
double *d_fft_of_ExP, *d_fft_of_EyP, *d_fft_of_EzP, *d_fft_of_BxP, *d_fft_of_ByP, *d_fft_of_BzP;  
	      
double *d_rEx,*d_rEy,*d_rEz,*d_rBx,*d_rBy,*d_rBz,*d_rJx,*d_rJy,*d_rJz,*d_rRhoBeam,*d_rJxBeam,*d_rRho;


int compare_vector_from_device(int n,double *h_v,double *d_v,char *s)
{
   double *h_copy = (double *)malloc(n*sizeof(double)),dmax = 0.0,t;
   
   cudaMemcpy(h_copy,d_v,n*sizeof(double),cudaMemcpyDeviceToHost);
   
   for(int i = 0; i < n;i++)
   {
       if((t = fabs(h_v[i] - h_copy[i])) > dmax) dmax = t;
   }

   for(int i = 0; i < n;i++)
   {
       if((t = fabs(h_v[i] - h_copy[i])) > dmax*0.5) printf("%s %5d delta %15.5e device %25.15e host %25.15e \n",s,i,t, h_copy[i],h_v[i]);;
   }
   
   printf("%s i = 0 host %25.15e device %25.15e 3host,device %25.15e,%25.15e \n",s,h_v[0],h_copy[0],h_v[3],h_copy[3]);
   printf("%s %15.5e \n",s,dmax);
   free(h_copy);
   return 0;
}

//k2_dens_inv,ky,kz,  k2_inv,r,ky_k2_Jy, kz_k2_Jz,ky_k2_Jz,kz_k2_Jy, jx_ky,jx_kz

int CUDA_WRAP_device_alloc(
int a_size,
double **d_a1,
double **d_a2,
double **d_a3,
double **d_a4,
double **d_a5,
double **d_a6,
double **d_a7,
double **d_a8,
double **d_a9,
double **d_a10,
double **d_a11,
double **d_a12,
double **d_a13,
double **d_a14,
double **d_a15,
double **d_a16,
double **d_a17,
double **d_a18,
double **d_a19,
double **d_a20,
double **d_a21,
double **d_a22,
double **d_a23,
double **d_a24,
double **d_a25 
)
{
   cudaMalloc((void**)d_a1,sizeof(double)*a_size);
   cudaMalloc((void**)d_a2,sizeof(double)*a_size);
   cudaMalloc((void**)d_a3,sizeof(double)*a_size);
   cudaMalloc((void**)d_a4,sizeof(double)*a_size);

   cudaMalloc((void**)d_a5,sizeof(double)*a_size);
   cudaMalloc((void**)d_a6,sizeof(double)*a_size);
   cudaMalloc((void**)d_a7,sizeof(double)*a_size);
   cudaMalloc((void**)d_a8,sizeof(double)*a_size);

   cudaMalloc((void**)d_a9,sizeof(double)*a_size);
   cudaMalloc((void**)d_a10,sizeof(double)*a_size);
   cudaMalloc((void**)d_a11,sizeof(double)*a_size);
   cudaMalloc((void**)d_a12,sizeof(double)*a_size);

   cudaMalloc((void**)d_a13,sizeof(double)*a_size);
   cudaMalloc((void**)d_a14,sizeof(double)*a_size);
   cudaMalloc((void**)d_a15,sizeof(double)*a_size);
   cudaMalloc((void**)d_a16,sizeof(double)*a_size);
   
   cudaMalloc((void**)d_a17,sizeof(double)*a_size);
   
   cudaMalloc((void**)d_a18,sizeof(double)*a_size);
   cudaMalloc((void**)d_a19,sizeof(double)*a_size);
   cudaMalloc((void**)d_a20,sizeof(double)*a_size);
   
   cudaMalloc((void**)d_a21,sizeof(double)*a_size);
   cudaMalloc((void**)d_a22,sizeof(double)*a_size);
   cudaMalloc((void**)d_a23,sizeof(double)*a_size);
   cudaMalloc((void**)d_a24,sizeof(double)*a_size);
   cudaMalloc((void**)d_a25,sizeof(double)*a_size);
   
   
   cudaMemset(*d_a1, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a2, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a3, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a4, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a5, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a6, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a7, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a8, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a9, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a10, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a11, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a12, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a13, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a14, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a15, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a16, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a17, 0.0,sizeof(double)*a_size);
   
   cudaMemset(*d_a18, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a19, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a20, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a21, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a22, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a23, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a24, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a25, 0.0,sizeof(double)*a_size);
   


 

//   puts("END ALLOC ================================================================");
   return 0;
}



int CUDA_WRAP_copy_all_vectors_to_device(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *a12,
double *a13,
double *a14,
double *a15,
double *a16,
double *a17,
/*
double *a18,
double *a19,
double *a20,
double *a21,
double *a22,
double *a23,
double *a24,
double *a25,
*/
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17
/*,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_a24,
double *d_a25
*/
)
{
   //puts("BEGIN COPY ======================================================================================");
   //exit(0);
   cublasSetVector(a_size,sizeof(double),a1,1,d_a1,1); 
//   compare_vector_from_device(a_size,a1,d_a1,"copy1");
 //  puts("in devcpy0.5");
//   exit(0);

   cublasSetVector(a_size,sizeof(double),a2,1,d_a2,1); 
   //compare_vector_from_device(a_size,a2,d_a2,"copy2");
   cublasSetVector(a_size,sizeof(double),a3,1,d_a3,1); 
   //compare_vector_from_device(a_size,a3,d_a3,"copy3");
   cublasSetVector(a_size,sizeof(double),a4,1,d_a4,1); 
   //compare_vector_from_device(a_size,a4,d_a4,"copy4");
   cublasSetVector(a_size,sizeof(double),a5,1,d_a5,1); 
   //compare_vector_from_device(a_size,a5,d_a5,"copy5");

   //puts("in devcpy0.51");
   //exit(0);

   
   cublasSetVector(a_size,sizeof(double),a6,1,d_a6,1); 
   //compare_vector_from_device(a_size,a6,d_a6,"copy6");
   cublasSetVector(a_size,sizeof(double),a7,1,d_a7,1); 
   //compare_vector_from_device(a_size,a7,d_a7,"copy7");
   cublasSetVector(a_size,sizeof(double),a8,1,d_a8,1); 
   //compare_vector_from_device(a_size,a8,d_a8,"copy8");
   cublasSetVector(a_size,sizeof(double),a9,1,d_a9,1); 
   //compare_vector_from_device(a_size,a9,d_a9,"copy9");
   cublasSetVector(a_size,sizeof(double),a10,1,d_a10,1); 
   //compare_vector_from_device(a_size,a10,d_a10,"copy10");

   //puts("in devcpy1");
  // exit(0);
   
   cublasSetVector(a_size,sizeof(double),a11,1,d_a11,1); 
   //compare_vector_from_device(a_size,a11,d_a11,"copy11");
   //puts("in devcpy1A");
  // exit(0);
   
   cublasSetVector(a_size,sizeof(double),a12,1,d_a12,1); 
   //compare_vector_from_device(a_size,a12,d_a12,"copy12");
   
   //puts("in devcpy1B");
//   exit(0);
   
   cublasSetVector(a_size,sizeof(double),a13,1,d_a13,1); 
   //compare_vector_from_device(a_size,a13,d_a13,"copy13");
   
   //puts("in devcpy1C");
   //exit(0);
   
   cublasSetVector(a_size,sizeof(double),a14,1,d_a14,1); 
   //compare_vector_from_device(a_size,a14,d_a14,"copy14");
   
   //puts("in devcpy1D");
   //exit(0);

   //cublasSetVector(a_size,sizeof(double),a15,1,d_a15,1); 
   cudaMemcpy(d_a15,a15,a_size*sizeof(double),cudaMemcpyHostToDevice);
   //compare_vector_from_device(a_size,a15,d_a15,"copy15");
   
   //puts("in devcpy1E");
 //  exit(0);
   //printf("a16 %e\n",a16[0]);
//   cublasSetVector(a_size,sizeof(double),a16,1,d_a16,1); 
   cudaMemcpy(d_a16,a16,a_size*sizeof(double),cudaMemcpyHostToDevice);

   cudaMemcpy(d_a17,a17,a_size*sizeof(double),cudaMemcpyHostToDevice);
   //compare_vector_from_device(a_size,a16,d_a16,"copy16 v before");

   //puts("END COPY ================================================================================================");
  // exit(0);
   
  /*
   cublasSetVector(a_size,sizeof(double),a17,1,d_a17,1); 
   cublasSetVector(a_size,sizeof(double),a18,1,d_a18,1); 
   cublasSetVector(a_size,sizeof(double),a19,1,d_a19,1); 
   cublasSetVector(a_size,sizeof(double),a20,1,d_a20,1); 
   
   cublasSetVector(a_size,sizeof(double),a21,1,d_a21,1); 
   cublasSetVector(a_size,sizeof(double),a22,1,d_a22,1); 
   cublasSetVector(a_size,sizeof(double),a23,1,d_a23,1); 
   cublasSetVector(a_size,sizeof(double),a24,1,d_a24,1); 
   cublasSetVector(a_size,sizeof(double),a25,1,d_a25,1); 
   */
   return 0;
}

int CUDA_WRAP_copy_all_vectors_to_host(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *a12,
double *a13,
double *a14,
double *a15,
double *a16,
/*
double *a17,
double *a18,
double *a19,
double *a20,
double *a21,
double *a22,
double *a23,
double *a24,
double *a25,
*/
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16
/*,
double *d_a17,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_a24,
double *d_a25
*/
)
{
  
   puts("in");
   cublasGetVector(a_size,sizeof(double),d_a1,1,a1,1); 
   cublasGetVector(a_size,sizeof(double),d_a2,1,a2,1); 
   cublasGetVector(a_size,sizeof(double),d_a3,1,a3,1); 
   cublasGetVector(a_size,sizeof(double),d_a4,1,a4,1); 
   cublasGetVector(a_size,sizeof(double),d_a5,1,a5,1); 

   cublasGetVector(a_size,sizeof(double),d_a6,1,a6,1); 
   cublasGetVector(a_size,sizeof(double),d_a7,1,a7,1); 
   cublasGetVector(a_size,sizeof(double),d_a8,1,a8,1); 
   cublasGetVector(a_size,sizeof(double),d_a9,1,a9,1); 
   cublasGetVector(a_size,sizeof(double),d_a10,1,a10,1); 

   cublasGetVector(a_size,sizeof(double),d_a11,1,a11,1); 
   cublasGetVector(a_size,sizeof(double),d_a12,1,a12,1); 
   cublasGetVector(a_size,sizeof(double),d_a13,1,a13,1); 
   cublasGetVector(a_size,sizeof(double),d_a14,1,a14,1); 
   cublasGetVector(a_size,sizeof(double),d_a15,1,a15,1); 
   
   cublasGetVector(a_size,sizeof(double),d_a16,1,a16,1); 
   
/*   
   cublasGetVector(a_size,sizeof(double),d_a17,1,a17,1); 
   cublasGetVector(a_size,sizeof(double),d_a18,1,a18,1); 
   cublasGetVector(a_size,sizeof(double),d_a19,1,a19,1); 
   cublasGetVector(a_size,sizeof(double),d_a20,1,a20,1); 

   cublasGetVector(a_size,sizeof(double),d_a21,1,a21,1); 
   cublasGetVector(a_size,sizeof(double),d_a22,1,a22,1); 
   cublasGetVector(a_size,sizeof(double),d_a23,1,a23,1); 
   cublasGetVector(a_size,sizeof(double),d_a24,1,a24,1); 
   cublasGetVector(a_size,sizeof(double),d_a25,1,a25,1); 
*/

   
   return 0;
}

int CUDA_WRAP_device_free(
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_a24,
double *d_a25
)
{
   cudaFree(d_a1);
   cudaFree(d_a2);
   cudaFree(d_a3);
   cudaFree(d_a4);
   cudaFree(d_a5);

   cudaFree(d_a6);
   cudaFree(d_a7);
   cudaFree(d_a8);
   cudaFree(d_a9);
   cudaFree(d_a10);

   cudaFree(d_a11);
   cudaFree(d_a12);
   cudaFree(d_a13);
   cudaFree(d_a14);
   cudaFree(d_a15);

   cudaFree(d_a16);
   cudaFree(d_a17);
   cudaFree(d_a18);
   cudaFree(d_a19);
   cudaFree(d_a20);

   cudaFree(d_a21);
   cudaFree(d_a22);
   cudaFree(d_a23);
   cudaFree(d_a24);
   cudaFree(d_a25);
   
   return 0;
}


void CUDA_WRAP_free(double *d)
{
    cudaFree(d);
}

int CUDA_WRAP_verify_all_vectors_on_host(
int a_size,
double *a1,double *d_a1,char *s1,
double *a2,double *d_a2,char *s2,
double *a3,double *d_a3,char *s3,
double *a4,double *d_a4,char *s4,
double *a5,double *d_a5,char *s5,
double *a6,double *d_a6,char *s6,
double *a7,double *d_a7,char *s7,
double *a8,double *d_a8,char *s8,
double *a9,double *d_a9,char *s9,
double *a10,double *d_a10,char *s10,
double *a11,double *d_a11,char *s11,
double *a12,double *d_a12,char *s12,
double *a13,double *d_a13,char *s13,
double *a14,double *d_a14,char *s14,
double *a15,double *d_a15,char *s15,
double *a16,double *d_a16,char *s16,

double *a17,double *d_a17,char *s17,
double *a18,double *d_a18,char *s18,
double *a19,double *d_a19,char *s19,
double *a20,double *d_a20,char *s20,
double *a21,double *d_a21,char *s21,
double *a22,double *d_a22,char *s22

)
{
    puts("BEGIN VERIFY =========================================================================================");
    compare_vector_from_device(a_size,a15,d_a15,"in ver 15");
    
    compare_vector_from_device(a_size,a1,d_a1,s1);
    compare_vector_from_device(a_size,a2,d_a2,s2);    
    
    compare_vector_from_device(a_size,a3,d_a3,s3);
    compare_vector_from_device(a_size,a4,d_a4,s4);    
    
    compare_vector_from_device(a_size,a5,d_a5,s5);
    compare_vector_from_device(a_size,a6,d_a6,s6);    
    
    
    
    compare_vector_from_device(a_size,a7,d_a7,s7);
    compare_vector_from_device(a_size,a8,d_a8,s8);    
    
    compare_vector_from_device(a_size,a9,d_a9,s9);
    compare_vector_from_device(a_size,a10,d_a10,s10);    
    
    compare_vector_from_device(a_size,a11,d_a11,s11);
    compare_vector_from_device(a_size,a12,d_a12,s12);  
    
    
    
    compare_vector_from_device(a_size,a13,d_a13,s13);
        
    compare_vector_from_device(a_size,a14,d_a14,s14);    
    compare_vector_from_device(a_size,a15,d_a15,s15);
    compare_vector_from_device(a_size,a16,d_a16,s16);    

    compare_vector_from_device(a_size,a17,d_a17,s17);    
    compare_vector_from_device(a_size,a18,d_a18,s18);    
    compare_vector_from_device(a_size,a19,d_a19,s19);    
    compare_vector_from_device(a_size,a20,d_a20,s20);    
    compare_vector_from_device(a_size,a21,d_a21,s21);    
    compare_vector_from_device(a_size,a22,d_a22,s22);    
    
    puts("END VERIFY ========================================================================================="); 
    
    return 0;
}

int CUDA_WRAP_verify_all_vectors_on_hostReal(
int a_size,
double *a1,double *d_a1,char *s1,
double *a2,double *d_a2,char *s2,
double *a3,double *d_a3,char *s3,
double *a4,double *d_a4,char *s4,
double *a5,double *d_a5,char *s5,
double *a6,double *d_a6,char *s6,
double *a7,double *d_a7,char *s7,
double *a8,double *d_a8,char *s8,
double *a9,double *d_a9,char *s9,
double *a10,double *d_a10,char *s10,
double *a11,double *d_a11,char *s11,
double *a12,double *d_a12,char *s12,
double *a13,double *d_a13,char *s13,
double *a14,double *d_a14,char *s14,
double *a15,double *d_a15,char *s15,
double *a16,double *d_a16,char *s16,

double *a17,double *d_a17,char *s17,
double *a18,double *d_a18,char *s18,
double *a19,double *d_a19,char *s19,
double *a20,double *d_a20,char *s20,
double *a21,double *d_a21,char *s21,
double *a22,double *d_a22,char *s22

)
{
#ifndef VERIFY_ALL_VECTORS
    return 0;
#else
    puts("BEGIN VERIFY =========================================================================================");
    compare_vector_from_device(a_size,a15,d_a15,"in ver 15");
    
    compare_vector_from_device(a_size,a1,d_a1,s1);
    compare_vector_from_device(a_size,a2,d_a2,s2);    
    
    compare_vector_from_device(a_size,a3,d_a3,s3);
    compare_vector_from_device(a_size,a4,d_a4,s4);    
    
    compare_vector_from_device(a_size,a5,d_a5,s5);
    compare_vector_from_device(a_size,a6,d_a6,s6);    
    
    
    
    compare_vector_from_device(a_size,a7,d_a7,s7);
    compare_vector_from_device(a_size,a8,d_a8,s8);    
    
    compare_vector_from_device(a_size,a9,d_a9,s9);
    compare_vector_from_device(a_size,a10,d_a10,s10);    
    
    compare_vector_from_device(a_size,a11,d_a11,s11);
    compare_vector_from_device(a_size,a12,d_a12,s12);  
    
    
    puts("END VERIFY ========================================================================================="); 
    
    return 0;
#endif
}



int CUDA_WRAP_copy_all_real_vectors_to_device(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11
)
{
   int err[10];
   //puts("BEGIN COPY ======================================================================================");
   //exit(0);
   err[0] = cublasSetVector(a_size,sizeof(double),a1,1,d_a1,1); 
//   compare_vector_from_device(a_size,a1,d_a1,"copy1");
 //  puts("in devcpy0.5");
//   exit(0);

   err[1] = cublasSetVector(a_size,sizeof(double),a2,1,d_a2,1); 
   //compare_vector_from_device(a_size,a2,d_a2,"copy2");
   err[2] = cublasSetVector(a_size,sizeof(double),a3,1,d_a3,1); 
   //compare_vector_from_device(a_size,a3,d_a3,"copy3");
   err[3] = cublasSetVector(a_size,sizeof(double),a4,1,d_a4,1); 
   //compare_vector_from_device(a_size,a4,d_a4,"copy4");
   err[4] = cublasSetVector(a_size,sizeof(double),a5,1,d_a5,1); 
   //compare_vector_from_device(a_size,a5,d_a5,"copy5");

   //puts("in devcpy0.51");
   //exit(0);

   
   err[5] = cublasSetVector(a_size,sizeof(double),a6,1,d_a6,1); 
   //compare_vector_from_device(a_size,a6,d_a6,"copy6");
   err[6] = cublasSetVector(a_size,sizeof(double),a7,1,d_a7,1); 
   //compare_vector_from_device(a_size,a7,d_a7,"copy7");
   err[7] = cublasSetVector(a_size,sizeof(double),a8,1,d_a8,1); 
   //compare_vector_from_device(a_size,a8,d_a8,"copy8");
   err[8] = cublasSetVector(a_size,sizeof(double),a9,1,d_a9,1);      
   
   //err[9] = cublasSetVector(a_size,sizeof(double),a10,1,d_a10,1); 
   err[9] = cudaMemcpy(d_a10,a10,a_size*sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(d_a11,a11,a_size*sizeof(double),cudaMemcpyHostToDevice);
   return 0;
}

int CUDA_WRAP_EMERGENCY_COPY(int ny,int nz,double *d_x,double *x)
{
#ifndef CUDA_WRAP_EMERGENCY_MATRIX_COPY
   return 0;
#endif
  
   cudaMemcpy(d_x,x,ny*nz*sizeof(double),cudaMemcpyHostToDevice);
   
   CUDA_DEBUG_printDdevice_matrix(ny,nz,d_x,"EMERGENCY ");
   
   return 0;
}

int CUDA_WRAP_device_real_alloc(
int a_size,
double **d_a1,
double **d_a2,
double **d_a3,
double **d_a4,
double **d_a5,
double **d_a6,
double **d_a7,
double **d_a8,
double **d_a9,
double **d_a10,
double **d_a11
)
{
   cudaMalloc((void**)d_a1,sizeof(double)*a_size);
   cudaMalloc((void**)d_a2,sizeof(double)*a_size);
   cudaMalloc((void**)d_a3,sizeof(double)*a_size);
   cudaMalloc((void**)d_a4,sizeof(double)*a_size);

   cudaMalloc((void**)d_a5,sizeof(double)*a_size);
   cudaMalloc((void**)d_a6,sizeof(double)*a_size);
   cudaMalloc((void**)d_a7,sizeof(double)*a_size);
   cudaMalloc((void**)d_a8,sizeof(double)*a_size);

   cudaMalloc((void**)d_a9,sizeof(double)*a_size);
   cudaMalloc((void**)d_a10,sizeof(double)*a_size);

   cudaMalloc((void**)d_a11,sizeof(double)*a_size);
   
   cudaMemset(*d_a1, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a2, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a3, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a4, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a5, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a6, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a7, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a8, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a9, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a10, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_a11, 0.0,sizeof(double)*a_size);

   
   return 0;
}

int CUDA_WRAP_deviceSetZero(
int a_size,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_b1,
double *d_b2,
double *d_b3,
double *d_b4,
double *d_b5,
double *d_b6,
double *d_b7,
double *d_b8,
double *d_b9
/*,
double **d_b10,
double **d_b11
*/
/*
double **d_a24,
double **d_a25 */
)
{
   cudaMemset(d_a1, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a2, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a3, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a4, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a5, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a6, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a7, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a8, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a9, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a10, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a11, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a12, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a13, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a14, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a15, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a16, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a17, 0.0,sizeof(double)*a_size);
   
   cudaMemset(d_a18, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a19, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a20, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a21, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a22, 0.0,sizeof(double)*a_size);
   cudaMemset(d_a23, 0.0,sizeof(double)*a_size);
   
   cudaMemset(d_b1, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b2, 0.0,sizeof(double)*a_size);
   
   cudaMemset(d_b3, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b4, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b5, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b6, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b7, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b8, 0.0,sizeof(double)*a_size);
   cudaMemset(d_b9, 0.0,sizeof(double)*a_size);
/*   cudaMemset(*d_b10, 0.0,sizeof(double)*a_size);
   cudaMemset(*d_b11, 0.0,sizeof(double)*a_size);
  */ 

/*
   cudaMalloc((void**)d_a17,sizeof(double)*a_size);
   cudaMalloc((void**)d_a18,sizeof(double)*a_size);
   cudaMalloc((void**)d_a19,sizeof(double)*a_size);
   cudaMalloc((void**)d_a20,sizeof(double)*a_size);

   cudaMalloc((void**)d_a21,sizeof(double)*a_size);
   cudaMalloc((void**)d_a22,sizeof(double)*a_size);
   cudaMalloc((void**)d_a23,sizeof(double)*a_size);
   cudaMalloc((void**)d_a24,sizeof(double)*a_size);

   cudaMalloc((void**)d_a25,sizeof(double)*a_size);
  */ 

//   puts("END ALLOC ================================================================");
   return 0;
}

