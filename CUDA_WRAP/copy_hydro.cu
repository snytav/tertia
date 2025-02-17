#include "cuda_wrap_vector_list.h"
#include "diagnostic_print.h"
#include "../run_control.h"

#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
//#include <cutil.h>

int copyBeamFourierDataToP_TEMPORARILY_FROM_HOST(int size,double *b1,double *b2,double *a1,double *a2)
{
    int err = cudaGetLastError();
    
    //CUDA_DEBUG_printDdevice_array(size,a1,"before");
    cudaMemcpy(b1,a1,size*sizeof(double),cudaMemcpyHostToDevice);
    
    
    //CUDA_DEBUG_printDdevice_array(size,b1,"after");
    cudaMemcpy(b2,a2,size*sizeof(double),cudaMemcpyHostToDevice);
    
    err = cudaGetLastError();
    
    return 0;
}

int copyBeamFourierDataToP(int size,double *b1,double *b2,double *a1,double *a2)
{
    int err = cudaGetLastError();
    
    //CUDA_DEBUG_printDdevice_array(size,a1,"before");
    cudaMemcpy(b1,a1,size*sizeof(double),cudaMemcpyDeviceToDevice);
    
    
    //CUDA_DEBUG_printDdevice_array(size,b1,"after");
    cudaMemcpy(b2,a2,size*sizeof(double),cudaMemcpyDeviceToDevice);
    
    err = cudaGetLastError();
    
    return 0;
}

int CUDA_WRAP_copyArraysHost(
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
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9
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
   err[6] = cublasSetVector(a_size,sizeof(double),a7,1,d_a7,1); 
   err[7] = cublasSetVector(a_size,sizeof(double),a8,1,d_a8,1); 
   err[8] = cublasSetVector(a_size,sizeof(double),a9,1,d_a9,1); 
   
   return 0;
}



/*
int CUDA_WRAP_copyTwoLayersHostToDevice(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer **dl)
{
   for (k=0; k<l_Mz; k++)
   {
      if(k==l_Mz/3) {
         double checkk = 0.;
      }
      for (j=0; j<l_My; j++)
      {
         if(j==l_My/3) {
            double checkj = 0.;
         }
         if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
            double check1=0;
         };
         long n = j + ny*k;
         long lccc = GetNyz(j,k);
         Cell &c = p_CellLayerC[lccc];
         Cell &cp = p_CellLayerP[lccc];

         rRho[n] = cp.f_Dens; // - dens;

         rEx[n] = cp.f_Ex;
         rEy[n] = cp.f_Ey;
         rEz[n] = cp.f_Ez;

         rJx[n] = cp.f_Jx;
         rJy[n] = cp.f_Jy;
         rJz[n] = cp.f_Jz;

         if (rRho[n] > 0.) {
            double check = 0;
         };

         if (i < l_Mx-1 && fabs(fabs(rRho[n]) - dens) > maxRho) {
            maxRho = fabs(fabs(rRho[n]) - dens);
         }

         rJxBeam[n] = c.f_JxBeam;
         rRhoBeam[n] = c.f_RhoBeam;


}
*/
