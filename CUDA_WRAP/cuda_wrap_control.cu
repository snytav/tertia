#include "cuda_wrap_control.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "diagnostic_print.h"
#include "../run_control.h"


#define NT 100

double times[NT];

void CUDA_WRAP_emergency_exit(char *where)
{
   cudaError_t err2 = cudaGetLastError();
   if (err2 != cudaSuccess)
   {
       printf("ERROR %d,%s  AT %s **************************************************\n",err2,
              cudaGetErrorString(err2)
              ,where);
       exit(0);
   }
  
}

int CUDA_WRAP_compare_device_array(int n,double *h_m,double *d_m,double *frac_ideal,double *frac_rude,char *legend,char *where,int details_flag)
{
   double *h_copy,t,dmax = 0;
   int num_ideal = 0,num_rude = 0,res = 0,i_max = 0;
   
   int err1 = cudaGetLastError();
   h_copy = (double *) malloc(sizeof(double)*n);

   cudaMemcpy(h_copy,d_m,n*sizeof(double),cudaMemcpyDeviceToHost);

   cudaError_t err2 = cudaGetLastError();

   CUDA_WRAP_emergency_exit(where);
   
   for(int i = 0;i < n;i++)
   {
       t = fabs(h_copy[i]-h_m[i]);
       if(t <= TOLERANCE_IDEAL) num_ideal++;
       if(t >  TOLERANCE_RUDE)  num_rude++;
   //    printf("%3d device %25.15e host %25.15e delta %10.3e \n",i,h_copy[i],h_m[i],t);
       
       if(t > dmax) 
       {
	  dmax = t;
	  i_max = i;
       }
       if((details_flag == DETAILS) && (t >  TOLERANCE_RUDE))
       {
	  int nx = (int)sqrt(n);
	  int ix = i/nx;
	  int iy = i - ix*nx;
	//  printf("i %10d ix %5d iy %5d host %15.5e device %15.5e delta %15.5e\n",i,ix,iy,h_m[i],h_copy[i],fabs(h_m[i]-h_copy[i])); 
       }
   }
   int err3 = cudaGetLastError();
   *frac_ideal = (double)num_ideal/((double)n);
   *frac_rude  = (double)num_rude /((double)n);
   
   res = (*frac_ideal > ((double)(n-1))/((double)n));
   
   if(details_flag == DETAILS)
   {
     // CUDA_DEBUG_printDdevice_matrix(4,4,d_m,legend);
#ifdef CUDA_WRAP_CONTROL
      printf("AT: %s COMPARE %10s : ideal %10.2f wrong %10.2f delta %15.5e MAX(%2d): device %25.15e host %25.15e \n",
	    where,legend,*frac_ideal,*frac_rude,dmax,i_max,h_copy[i_max],h_m[i_max]);
#endif
   }
   int err4 = cudaGetLastError(); 
   
   free(h_copy);

   return res;
}

int timeBegin(int num_test)
{
   struct timeval tv_start;
   gettimeofday(&tv_start,NULL);
   times[num_test] = tv_start.tv_sec+1e-6*tv_start.tv_usec;
}

int timeEnd(int num_test)
{
   struct timeval tv_start;
  
   gettimeofday(&tv_start,NULL);
   times[num_test] = tv_start.tv_sec+1e-6*tv_start.tv_usec -times[num_test];
}

int timeInit()
{
    for(int i = 0;i < NT;i++) times[i] = 0;
    
    return 0;
}

int timePrint()
{
   struct timeval tv;
   gettimeofday(&tv,NULL);
   
   //printf("linEB: host: %15.5e device %15.5e linEB kernel %15.5e norm: host: %15.5e device %15.5e kernel %15.5e\n",times[0],times[1],times[2],times[3],times[4],times[5]);
   printf("fourier whole: %e\n",times[6]);
   printf("halloc %.2f \n",times[7]/times[6]);
   printf(" copy %.2f \n",times[8]/times[6]);
   printf(" turn %.2f \n",times[10]/times[6]);
   printf(" four %.2f \n",times[11]/times[6]);
   printf(" tran %.2f \n",times[12]/times[6]);
   printf("fourier whole inside %e 14 %e 15 %e 16 %e \n",times[19],times[14],times[15],times[16]);
   printf("fourier parts %e \n",times[7]+times[8]+times[9]+times[10]+times[11]+times[12]+times[13]+times[14]+times[31]+times[32]+times[33]+times[34]+times[35]);
   printf("31 %.2f 32 %.2f 33 %.2f 34 %.2f 35  %.2f \n",times[31]/times[6],times[32]/times[6],times[33]/times[6],times[34]/times[6],times[35]/times[6]);
   printf("21 %e 22 %e\n",times[21],times[22]);
   printf("batch fourier %e \n",times[13]);
   printf("FFTW time %e \n",times[99]);
}


double CUDA_WRAP_verify_all_vectors_on_host(
int a_size,char *where,int details_flag,
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
double *a21,double *d_a21,char *s21
)
{

  
    double frac_ideal,frac_rude,total_ideal= 0.0,total_rude = 0.0;;
    int num_params = 21;
//    return 0;
    int err1 = cudaGetLastError();
    
#ifndef  CUDA_WRAP_VERIFICATION_ALLOWED   
    return 1;
#endif        
    
 //   return 0;
   // if(details_flag == DETAILS) puts("BEGIN VERIFY =========================================================================================");
  //  return 0;
    CUDA_WRAP_compare_device_array(a_size,a1,d_a1,&frac_ideal,&frac_rude,s1,where,details_flag);
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    int err2 = cudaGetLastError();
    CUDA_WRAP_compare_device_array(a_size,a2,d_a2,&frac_ideal,&frac_rude,s2,where,details_flag);
    int err3 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a3,d_a3,&frac_ideal,&frac_rude,s3,where,details_flag);
    int err4 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a4,d_a4,&frac_ideal,&frac_rude,s4,where,details_flag);
    int err5 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    //return 0; 
    CUDA_WRAP_compare_device_array(a_size,a5,d_a5,&frac_ideal,&frac_rude,s5,where,details_flag);
    int err6 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a6,d_a6,&frac_ideal,&frac_rude,s6,where,details_flag);
    int err7 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a7,d_a7,&frac_ideal,&frac_rude,s7,where,details_flag);
    int err8 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a8,d_a8,&frac_ideal,&frac_rude,s8,where,details_flag);
    int err9 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a9,d_a9,&frac_ideal,&frac_rude,s9,where,details_flag);
    int err10 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a10,d_a10,&frac_ideal,&frac_rude,s10,where,details_flag);
    int err11 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    //return 0;
    CUDA_WRAP_compare_device_array(a_size,a11,d_a11,&frac_ideal,&frac_rude,s11,where,details_flag);
    int err12 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a12,d_a12,&frac_ideal,&frac_rude,s12,where,details_flag);
    int err13 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a13,d_a13,&frac_ideal,&frac_rude,s13,where,details_flag);
    int err14 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a14,d_a14,&frac_ideal,&frac_rude,s14,where,details_flag);
    int err15 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
     
    CUDA_WRAP_compare_device_array(a_size,a15,d_a15,&frac_ideal,&frac_rude,s15,where,details_flag);
    int err16 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a16,d_a16,&frac_ideal,&frac_rude,s16,where,details_flag);
    int err17 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    
    CUDA_WRAP_compare_device_array(a_size,a17,d_a17,&frac_ideal,&frac_rude,s17,where,details_flag);
    int err18 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    //return 0;
    CUDA_WRAP_compare_device_array(a_size,a18,d_a18,&frac_ideal,&frac_rude,s18,where,details_flag);
    int err19 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a19,d_a19,&frac_ideal,&frac_rude,s19,where,details_flag);
    int err20 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a20,d_a20,&frac_ideal,&frac_rude,s20,where,details_flag);
    int err21 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a21,d_a21,&frac_ideal,&frac_rude,s21,where,details_flag);
    int err22 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    
    
    
    
    printf("AT: %30s TOTAL ideal %15.5f rude %15.5f \n",where,total_ideal/(double)num_params,total_rude/(double)num_params); 
    if(details_flag == DETAILS) puts("END VERIFY ==========================================================================================="); 
    
    return (total_ideal/(double)num_params);
}    


double CUDA_WRAP_verify_all_vectors_on_hostReal(
int a_size,char *where,int details_flag,
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
double *a12,double *d_a12,char *s12
)
{
    double frac_ideal,frac_rude,total_ideal= 0.0,total_rude = 0.0;;
    int num_params = 12;
    
    int err1 = cudaGetLastError();
    
    if(details_flag == DETAILS) puts("BEGIN VERIFY =========================================================================================");
    CUDA_WRAP_compare_device_array(a_size,a1,d_a1,&frac_ideal,&frac_rude,s1,where,details_flag);
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    int err2 = cudaGetLastError();
    CUDA_WRAP_compare_device_array(a_size,a2,d_a2,&frac_ideal,&frac_rude,s2,where,details_flag);
    int err3 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a3,d_a3,&frac_ideal,&frac_rude,s3,where,details_flag);
    int err4 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a4,d_a4,&frac_ideal,&frac_rude,s4,where,details_flag);
    int err5 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a5,d_a5,&frac_ideal,&frac_rude,s5,where,details_flag);
    int err6 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a6,d_a6,&frac_ideal,&frac_rude,s6,where,details_flag);
    int err7 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a7,d_a7,&frac_ideal,&frac_rude,s7,where,details_flag);
    int err8 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a8,d_a8,&frac_ideal,&frac_rude,s8,where,details_flag);
    int err9 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a9,d_a9,&frac_ideal,&frac_rude,s9,where,details_flag);
    int err10 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    CUDA_WRAP_compare_device_array(a_size,a10,d_a10,&frac_ideal,&frac_rude,s10,where,details_flag);
    int err11 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;

    CUDA_WRAP_compare_device_array(a_size,a11,d_a11,&frac_ideal,&frac_rude,s11,where,details_flag);
    int err12 = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    
    CUDA_WRAP_compare_device_array(a_size,a12,d_a12,&frac_ideal,&frac_rude,s12,where,details_flag);
    int err12a = cudaGetLastError();
    total_ideal += frac_ideal;
    total_rude   += frac_rude;
    
    
    printf("AT: %30s TOTAL ideal %15.5f rude %15.5f \n",where,total_ideal/(double)num_params,total_rude/(double)num_params); 
    if(details_flag == DETAILS) puts("END VERIFY ==========================================================================================="); 
    
    return (total_ideal/(double)num_params);
}    


