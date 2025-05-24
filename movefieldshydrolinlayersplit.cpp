#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
//#include <complex>
#include "vcomplex.h"

using namespace std;

#include <fftw3.h>

#include "vlpl3d.h"

#include <float.h>

#include <sys/time.h>

#include "run_control.h"

#include "CUDA_WRAP/half_integer2D.h"
#include "CUDA_WRAP/variables.h"
#include "CUDA_WRAP/cuda_wrap_vector_list.h"
#include "CUDA_WRAP/linearized.h"
#include "CUDA_WRAP/cuda_wrap_control.h"
#include "CUDA_WRAP/normalization.h"
#include "CUDA_WRAP/fourierInit.h"
#include "CUDA_WRAP/mult.h"
#include "CUDA_WRAP/turn.h"
#include "CUDA_WRAP/cuParticles.h"
#include "CUDA_WRAP/diagnostic_print.h"
#include "run_control.h"
#include "CUDA_WRAP/beam_copy.h"
#include "CUDA_WRAP/cuLayers.h"
#include "CUDA_WRAP/paraCPUlayers.h"

#include "CUDA_WRAP/copy_hydro.h"
#include "CUDA_WRAP/profile.h"

#include "para.h"

double last_fourier,last_ordinary,last_hidden;


static int FirstCall;

//unsigned int fp_control_state = _controlfp(_EM_INEXACT, _MCW_EM);
static double *rEx, *rEy, *rEz, *rBx, *rBy, *rBz, *rJx, *rJy, *rJz, *rRho;
static double *rJxBeam, *rJyBeam, *rJzBeam, *rRhoBeam;
static double *rJxBeamP, *rJyBeamP, *rJzBeamP, *rRhoBeamP;
static double *fft_of_Ex, *fft_of_Ey, *fft_of_Ez, *fft_of_Bx, *fft_of_By, *fft_of_Bz;
static double *fft_of_ExP, *fft_of_EyP, *fft_of_EzP, *fft_of_BxP, *fft_of_ByP, *fft_of_BzP;
static double *fft_of_Jx, *fft_of_Jy, *fft_of_Jz, *fft_of_Rho;
static double *fft_of_JxP, *fft_of_JyP, *fft_of_JzP, *fft_of_RhoP;
static double *fft_of_JxBeam, *fft_of_JyBeam, *fft_of_JzBeam, *fft_of_RhoBeam;
static double *fft_of_JxBeamP, *fft_of_JyBeamP, *fft_of_JzBeamP, *fft_of_RhoBeamP;
static double *fftDensExpected;
static double *carray;
static double *fft_of_ExRho, *fft_of_EyRho, *fft_of_EzRho;
static fftw_plan planR2R_Ex, planR2R_Ey, planR2R_Ez;
static fftw_plan planR2R_Bx, planR2R_By, planR2R_Bz;
static fftw_plan planR2Rb_Ex, planR2Rb_Ey, planR2Rb_Ez;
static fftw_plan planR2Rb_Bx, planR2Rb_By, planR2Rb_Bz;
static fftw_plan planR2R_Jx, planR2R_Jy, planR2R_Jz, planR2R_Rho;
static fftw_plan planR2R_JxBeam, planR2R_JyBeam, planR2R_JzBeam, planR2R_RhoBeam;
static fftw_plan planR2R_JxBeamP, planR2R_JyBeamP, planR2R_JzBeamP, planR2R_RhoBeamP;
static fftw_plan planR2Rb_Jx, planR2Rb_Jy, planR2Rb_Jz, planR2Rb_Rho;
static fftw_plan planR2R_ExRho, planR2R_EyRho, planR2R_EzRho;

static double maxRho;

void CUDA_WRAP_getBeamFFT(double *jx,double *rho,int ncomplex)
{
   static int first = 1;



   for (int n=0; n<ncomplex; n++) 
   {
      jx[n]   = fft_of_JxBeam[n];
      rho[n]  = fft_of_RhoBeam[n];
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG      
      printf("getBeamFFT %5d %15.5e %15.5e  \n ",n,fft_of_RhoBeam[n],fft_of_JxBeam[n]);
#endif      
   }    
}


int CUDA_WRAP_COPY_INIT(int ny,int nz)
{
  
  
   CUDA_WRAP_copy_all_real_vectors_to_device(ny*nz,rEx,rEy,rEz,rBx,rBy,rBz,rJx,rJy,rJz,rRhoBeam,rJxBeam,
					           d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,d_rJx,d_rJy,d_rJz,d_rRhoBeam,d_rJxBeam); // 16 
   CUDA_WRAP_copy_all_vectors_to_device(ny*nz,fft_of_RhoP, fft_of_Rho, fft_of_JxP, fft_of_Jx,
                                        fft_of_JyP,fft_of_Jy,fft_of_JzP,fft_of_Jz,
					fft_of_Ex,fft_of_Ey,fft_of_Ez,fft_of_Bx,
					fft_of_By,fft_of_Bz,fft_of_JxBeam,fft_of_RhoBeam,rRho,
                                        ///////////////////////////
                                        d_fft_of_RhoP,d_fft_of_Rho,d_fft_of_JxP,d_fft_of_Jx,  // 4
                                        d_fft_of_JyP, d_fft_of_Jy, d_fft_of_JzP, d_fft_of_Jz,  // 8
                                        d_fft_of_Ex,d_fft_of_Ey,d_fft_of_Ez,d_fft_of_Bx,       // 12
                                        d_fft_of_By,d_fft_of_Bz,d_fft_of_JxBeam,d_fft_of_RhoBeam,d_rRho); // 16
   return 0;
}

int CUDA_WRAP_output_host_matrix(int n1,int n2,char *legend,int iLayer,double *h_m)
{
    double *h;
    char s[100];
    FILE *f;
    
    //h = (double *)malloc(n1*n2*sizeof(double));
    
    //cudaMemcpy(h,d_m,n1*n2*sizeof(double),cudaMemcpyDeviceToHost);

    sprintf(s,"host%s%03d.dat",legend,iLayer);
    if((f = fopen(s,"wt")) == NULL) return 1;
    
    for(int i = 0;i < n1;i++)
    {
       for(int j = 0;j < n2;j++)
       {
           fprintf(f,"%5d %5d %25.15e \n",i,j,h_m[i*n2+j]);
       }
       
       fprintf(f,"\n");
    }
    fclose(f);
    
    //free(h);
    
    return 0;

}

int CUDA_WRAP_CHECK(
  int ny,
  int nz,
  char *where,
  int details_flag,
  Mesh *mesh,
  int i_layer,
  Cell *p_CellArray
)
{
    double err_fft,err,err_hidden;
  
#ifndef CUDA_WRAP_CHECK_ALLOWED
    return 0;
#endif    
//    int err0 = cudaGetLastError();
    
    details_flag = CHECK_DETAILS;
    
    err_fft = CUDA_WRAP_verify_all_vectors_on_host(ny*nz,where,details_flag,
                                        fft_of_RhoP,d_fft_of_RhoP, "RhoP ",   // 1
                                        fft_of_Rho, d_fft_of_Rho, "Rho ",   // 2
                                        fft_of_JxP, d_fft_of_JxP, "JxP ",   // 3
                                        fft_of_Jx, d_fft_of_Jx,   "Jx  ",   // 4
                                        fft_of_JyP,d_fft_of_JyP,  "JyP ",   // 5
                                        fft_of_Jy, d_fft_of_Jy,   "Jy  ",   // 6
                                        fft_of_JzP,d_fft_of_JzP,  "JzP ",   // 7
                                        fft_of_Jz, d_fft_of_Jz,   "Jz  ",   // 8
                                        fft_of_Ex, d_fft_of_Ex,   "Ex  ",   // 9
                                        fft_of_Ey, d_fft_of_Ey,   "Ey  ",   // 10
                                        fft_of_Ez,d_fft_of_Ez,    "Ez  ",   // 11    
					fft_of_Bx, d_fft_of_Bx,   "Bx  ",    // 12
					fft_of_By, d_fft_of_By,   "By  ",   // 13
                                        fft_of_Bz, d_fft_of_Bz,   "Bz  ",    // 14
                                        fft_of_JxBeam,d_fft_of_JxBeam,   "JxBeam ", // 15
                                        fft_of_RhoBeam,d_fft_of_RhoBeam,  "RhoBeam ",// 16
                                        fft_of_ExP,d_fft_of_ExP,  "ExP ",   // 17
                                        fft_of_EyP,d_fft_of_EyP,  "EyP ",   // 18
                                        fft_of_EzP,d_fft_of_EzP,  "EzP ",   // 19
					fft_of_JxBeamP,d_fft_of_JxBeamP,"JxBeamP ", //20
					fft_of_RhoBeamP,d_fft_of_RhoBeamP,"RhoBeamP " //21
	                                );
					
//    int err1 = cudaGetLastError();
    err = CUDA_WRAP_verify_all_vectors_on_hostReal(ny*nz,where,details_flag,
                                        rJx,     d_rJx,       "rJx ",    // 1
                                        rRho,    d_rRho,      "rRho ",   // 2
                                        rJy,     d_rJy,       "rJy  ",   // 3
                                        rJz,     d_rJz,       "rJz  ",   // 4
                                        rEx,     d_rEx,       "rEx  ",   // 5
                                        rEy,     d_rEy,       "rEy  ",   // 6
                                        rEz,     d_rEz,       "rEz  ",   // 7    
					rBx,     d_rBx,       "rBx  ",   // 8
					rBy,     d_rBy,       "rBy  ",   // 9
                                        rBz,     d_rBz,       "rBz  ",   // 10
                                        rJxBeam, d_rJxBeam,   "rJxBeam ", // 11
                                        rRhoBeam,d_rRhoBeam,  "rRhoBeam " // 12
                                       );  
//   int err2 = cudaGetLastError(); 
   err_hidden = 1.0; //CUDA_WRAP_getArraysToCompare(where,mesh,i_layer,ny,nz,p_CellArray);
   
   printf("%s ======================== FourierSpace %15.5f OrdinarySpace %15.5f HiddenCurrents %15.5f ========================================\n",where,err_fft,err,err_hidden);
   
//   int err3 = cudaGetLastError();
   
   last_fourier  = err_fft;
   last_ordinary = err;
   last_hidden   = err_hidden;
    
   return 0; 
}

void printSummary(char *where,int iLayer)
{
     //puts("pp1");
     printf("%20s %d == wrong values %10.4f max delta %15.5e for value %5d fourier %10.4f ordinary %10.4f hidden %10.4f =================================\n",
	                      where,iLayer,last_wrong,last_delta,last_max_delta_value,last_fourier,last_ordinary,last_hidden);
     
     //puts("pp");
     if(last_delta > DELTA_TOLERANCE)
     {
       //  puts("qq");
         printf("LAYER %d delta %e too big, EXITING\n",iLayer,last_delta); 
         exit(0);
     }
}


//--- Mesh:: ----------------------.
void Mesh::GuessFieldsHydroLinLayerSplit(int iLayer,int iSplit)
{
//   cout << "Layer=" << iLayer <<endl;

   int i, j, k;
   double max_dJy = 0.;
   double max_Jy = 0.;
   double max_dEy = 0.;
   double max_Ey = 0.;
   double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;
   struct timeval tv[CUDA_WRAP_TN];
   printf("in GuessFieldsHydroLinLayerSplit \n");
   
//   int err0 = cudaGetLastError();
   gettimeofday(&tv[0],NULL);

   i = l_Mx-1;;
   j = l_My/2.;
   k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;;
   double zco = Z(k) - domain()->GetZlength()/2.;;

   double dens = 0.;
   int nsorts = domain()->GetNsorts();
   for (int isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += fabs(spec->Density(xco,yco,zco)*spec->GetQ2M());
   };

   if (dens == 0.) dens = 1.;

   double ts = Ts();
   double hx = HxSplit();
   double hy = Hy();
   double hz = Hz();

   VComplex I = VComplex(0.,1.);
   //   I.re = 0.;
   //   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int ncomplex = nz*ny;
//   int err01 = cudaGetLastError();

   if (FirstCall == 0) {
      FirstCall = -1;
      rEx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rEy = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rEz = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rBx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rBy = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rBz = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJy = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJz = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rRho= (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJxBeam = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJyBeam = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJzBeam = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rRhoBeam= (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJxBeamP = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJyBeamP = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJzBeamP = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rRhoBeamP= (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
//      int err03 = cudaGetLastError();
      

      fft_of_Ex = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Ey = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Ez = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Bx = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_By = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Bz = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Jx = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Jy = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Jz = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_Rho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_ExP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EyP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EzP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JxP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JyP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JzP = (double*)fftw_malloc(ncomplex*sizeof(double)); 
      fft_of_RhoP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JxBeam = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JyBeam = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JzBeam = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_RhoBeam = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JxBeamP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JyBeamP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JzBeamP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_RhoBeamP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fftDensExpected = (double*)fftw_malloc(ncomplex*sizeof(double));
      carray = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_ExRho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EyRho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EzRho = (double*)fftw_malloc(ncomplex*sizeof(double));

//      int err04 = cudaGetLastError();
      
      CUDA_WRAP_device_alloc(ny*nz,&d_fft_of_RhoP,&d_fft_of_Rho,&d_fft_of_JxP,&d_fft_of_Jx, // 4
                                &d_fft_of_JyP, &d_fft_of_Jy,&d_fft_of_JzP, &d_fft_of_Jz, // 8
                                &d_fft_of_Ex, &d_fft_of_Ey,&d_fft_of_Ez,&d_fft_of_Bx,    // 12
                                &d_fft_of_By,&d_fft_of_Bz,&d_fft_of_JxBeam,&d_fft_of_RhoBeam,  //16
			                                     &d_rRho,
                                &d_fft_of_ExP, &d_fft_of_EyP,&d_fft_of_EzP,
			        &d_fft_of_BxP, &d_fft_of_ByP,&d_fft_of_BzP,
			        &d_fft_of_JxBeamP,&d_fft_of_RhoBeamP);
			     							     
//      int err1 = cudaGetLastError();
      
      CUDA_WRAP_device_real_alloc(ny*nz,&d_rEx,&d_rEy,&d_rEz,&d_rBx,&d_rBy,&d_rBz,&d_rJx,&d_rJy,&d_rJz,&d_rRhoBeam,&d_rJxBeam);
      
#ifdef  CUDA_WRAP_VERIFICATION00_ALLOWED         
       CUDA_WRAP_verify_all_vectors_on_host(ny*nz,"verification0000000000000000000000000",DETAILS,
                                        fft_of_RhoP,d_fft_of_RhoP, "RhoP ",   // 1
                                        fft_of_Rho, d_fft_of_Rho, "Rho ",   // 2
                                        fft_of_JxP, d_fft_of_JxP, "JxP ",   // 3
                                        fft_of_Jx, d_fft_of_Jx,   "Jx  ",   // 4
                                        fft_of_JyP,d_fft_of_JxP,  "JxP ",   // 5
                                        fft_of_Jy, d_fft_of_Jy,   "Jy  ",   // 6
                                        fft_of_JzP,d_fft_of_JzP,  "JzP ",   // 7
                                        fft_of_Jz, d_fft_of_Jz,   "Jz  ",   // 8
                                        fft_of_Ex, d_fft_of_Ex,   "Ex  ",   // 9
                                        fft_of_Ey, d_fft_of_Ey,   "Ey  ",   // 10
                                        fft_of_Ez,d_fft_of_Ez,    "Ez  ",   // 11    
					fft_of_Bx, d_fft_of_Bx,   "Bx  ",    // 12
					fft_of_By, d_fft_of_By,   "By  ",   // 13
                                        fft_of_Bz, d_fft_of_Bz,   "Bz  ",    // 14
                                        fft_of_JxBeam,d_fft_of_JxBeam,   "JxBeam ", // 15
                                        fft_of_RhoBeam,d_fft_of_RhoBeam,  "RhoBeam ",// 16
                                        fft_of_ExP,d_fft_of_ExP,  "ExP ",   // 17
                                        fft_of_EyP,d_fft_of_EyP,  "EyP ",   // 18
                                        fft_of_EzP,d_fft_of_EzP,  "EzP ",   // 19
					fft_of_JxBeamP,d_fft_of_JxBeamP,"JxBeamP ", //20
					fft_of_RhoBeamP,d_fft_of_RhoBeamP,"RhoBeamP " //21					
                                       );
#endif       
       
//       int err2 = cudaGetLastError();
#ifndef CUDA_WRAP_FFTW_ALLOWED  
       
      CUDA_WRAP_FourierInit(ny,nz);
      CUDA_WRAP_buffer_init(ny,nz);
      
#endif 
      
     // CUDA_WRAP_COPY_INIT(ny,nz);
#ifdef COPY_BEAM_FROM_HOST
      CUDA_WRAP_copyBeamToArray(this,l_Mx,l_My,l_Mz,p_CellArray,&d_RhoBeam3D,&d_JxBeam3D);
#endif      
      //CUDA_WRAP_3Dto2D(iLayer-1,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);
      //CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJxBeam,"Jx0");
     
      

      int gotthreads = fftw_init_threads();
      int nthreads = 2;
     if (gotthreads == 0) {
         cout << "Could not init threads! \n";
         nthreads = 1;
      };

      fftw_plan_with_nthreads(nthreads);  
      /**/

      planR2Rb_Ex = fftw_plan_r2r_2d(nz, ny, carray, rEx, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2Rb_Ey = fftw_plan_r2r_2d(nz, ny, carray, rEy, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2Rb_Ez = fftw_plan_r2r_2d(nz, ny, carray, rEz, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2Rb_Bx = fftw_plan_r2r_2d(nz, ny, carray, rBx, FFTW_RODFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2Rb_By = fftw_plan_r2r_2d(nz, ny, carray, rBy, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2Rb_Bz = fftw_plan_r2r_2d(nz, ny, carray, rBz, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);

      planR2R_Ex = fftw_plan_r2r_2d(nz, ny, rEx, fft_of_Ex, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_Ey = fftw_plan_r2r_2d(nz, ny, rEy, fft_of_Ey, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_Ez = fftw_plan_r2r_2d(nz, ny, rEz, fft_of_Ez, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2R_Bx = fftw_plan_r2r_2d(nz, ny, rBx, fft_of_Bx, FFTW_RODFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_By = fftw_plan_r2r_2d(nz, ny, rBy, fft_of_By, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_Bz = fftw_plan_r2r_2d(nz, ny, rBz, fft_of_Bz, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);

      planR2R_ExRho = fftw_plan_r2r_2d(nz, ny, rEx, fft_of_ExRho, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_EyRho = fftw_plan_r2r_2d(nz, ny, rEy, fft_of_EyRho, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_EzRho = fftw_plan_r2r_2d(nz, ny, rEz, fft_of_EzRho, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2R_Jx  = fftw_plan_r2r_2d(nz, ny, rJx, fft_of_Jx, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_Jy  = fftw_plan_r2r_2d(nz, ny, rJy, fft_of_Jy, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_Jz  = fftw_plan_r2r_2d(nz, ny, rJz, fft_of_Jz, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_Rho = fftw_plan_r2r_2d(nz, ny, rRho, fft_of_Rho, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2R_JxBeam  = fftw_plan_r2r_2d(nz, ny, rJxBeam, fft_of_JxBeam, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_JyBeam  = fftw_plan_r2r_2d(nz, ny, rJyBeam, fft_of_JyBeam, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_JzBeam  = fftw_plan_r2r_2d(nz, ny, rJzBeam, fft_of_JzBeam, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_RhoBeam = fftw_plan_r2r_2d(nz, ny, rRhoBeam, fft_of_RhoBeam, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2R_JxBeamP  = fftw_plan_r2r_2d(nz, ny, rJxBeamP, fft_of_JxBeamP, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_JyBeamP  = fftw_plan_r2r_2d(nz, ny, rJyBeamP, fft_of_JyBeamP, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2R_JzBeamP  = fftw_plan_r2r_2d(nz, ny, rJzBeamP, fft_of_JzBeamP, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2R_RhoBeamP = fftw_plan_r2r_2d(nz, ny, rRhoBeamP, fft_of_RhoBeamP, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);

      planR2Rb_Jx  = fftw_plan_r2r_2d(nz, ny, carray, rJx, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2Rb_Jy  = fftw_plan_r2r_2d(nz, ny, carray, rJy, FFTW_REDFT11,  FFTW_RODFT11, FFTW_ESTIMATE);
      planR2Rb_Jz  = fftw_plan_r2r_2d(nz, ny, carray, rJz, FFTW_RODFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
      planR2Rb_Rho = fftw_plan_r2r_2d(nz, ny, carray, rRho, FFTW_REDFT11,  FFTW_REDFT11, FFTW_ESTIMATE);
   }

   if ((GetRank() == GetSize() - 1) && (iLayer > l_Mx-2)) {
      maxRho = 0.;
      for (i=0; i<ncomplex; i++) {
         fft_of_Ex[i] = fft_of_Ey[i] = fft_of_Ez[i] 
         = fft_of_Bx[i] = fft_of_By[i] = fft_of_Bz[i] 
         = fft_of_Jx[i] = fft_of_Jy[i] = fft_of_Jz[i] = fft_of_Rho[i] 
         = fft_of_JxP[i] = fft_of_JyP[i] = fft_of_JzP[i] = fft_of_RhoP[i]
         = fft_of_ExP[i] = fft_of_EyP[i] = fft_of_EzP[i]
         = fft_of_JxBeam[i] = fft_of_JyBeam[i] = fft_of_JzBeam[i] = fft_of_RhoBeam[i]
         = fft_of_ExRho[i] = fft_of_EyRho[i] = fft_of_EzRho[i] 
         = fft_of_JxBeamP[i] = fft_of_JyBeamP[i] = fft_of_JzBeamP[i] = fft_of_RhoBeamP[i] = 0.;
         fftDensExpected[i] = dens;
      }
      

   }
   
 /*  CUDA_WRAP_deviceSetZero(ny*nz,d_fft_of_Ex,d_fft_of_Ey,d_fft_of_Ez,  // 3
                                    d_fft_of_Bx,d_fft_of_By,d_fft_of_Bz,  // 6
                                    d_fft_of_Jx,d_fft_of_Jy,d_fft_of_Jz,d_fft_of_Rho, // 10
                                    d_fft_of_JxP,d_fft_of_JyP,d_fft_of_JzP,d_fft_of_RhoP, // 14
                                    d_fft_of_ExP,d_fft_of_EyP,d_fft_of_EzP,                  // 17   
                                    d_fft_of_JxBeam,fft_of_JyBeam,d_fft_of_JzBeam,d_fft_of_RhoBeam, // 21
                                    d_fft_of_JxBeamP,d_fft_of_JyBeamP,d_fft_of_JzBeamP,d_fft_of_RhoBeamP, // 25
			            d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rRho                		       ); // 
*/
#ifdef CUDA_WRAP_VERIFICATION_ALLOWED
   CUDA_WRAP_verify_all_vectors_on_host(ny*nz,"verification in the beginning",DETAILS,
                                        fft_of_RhoP,d_fft_of_RhoP, "RhoP ",   // 1
                                        fft_of_Rho, d_fft_of_Rho, "Rho ",   // 2
                                        fft_of_JxP, d_fft_of_JxP, "JxP ",   // 3
                                        fft_of_Jx, d_fft_of_Jx,   "Jx  ",   // 4
                                        fft_of_JyP,d_fft_of_JxP,  "JxP ",   // 5
                                        fft_of_Jy, d_fft_of_Jy,   "Jy  ",   // 6
                                        fft_of_JzP,d_fft_of_JzP,  "JzP ",   // 7
                                        fft_of_Jz, d_fft_of_Jz,   "Jz  ",   // 8
                                        fft_of_Ex, d_fft_of_Ex,   "Ex  ",   // 9
                                        fft_of_Ey, d_fft_of_Ey,   "Ey  ",   // 10
                                        fft_of_Ez,d_fft_of_Ez,    "Ez  ",   // 11    
					fft_of_Bx, d_fft_of_Bx,   "Bx  ",    // 12
					fft_of_By, d_fft_of_By,   "By  ",   // 13
                                        fft_of_Bz, d_fft_of_Bz,   "Bz  ",    // 14
                                        fft_of_JxBeam,d_fft_of_JxBeam,   "JxBeam ", // 15
                                        fft_of_RhoBeam,d_fft_of_RhoBeam,  "RhoBeam ",// 16
                                        fft_of_ExP,d_fft_of_ExP,  "ExP ",   // 17
                                        fft_of_EyP,d_fft_of_EyP,  "EyP ",   // 18
                                        fft_of_EzP,d_fft_of_EzP,  "EzP ",   // 19
					fft_of_JxBeamP,d_fft_of_JxBeamP,"JxBeamP ", //20
					fft_of_RhoBeamP,d_fft_of_RhoBeamP,"RhoBeamP " //21
				       );
#endif

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;

   double maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;
   
   //printf("in Guess rank %d \n",GetRank());    
   
   
   gettimeofday(&tv[1],NULL);
   
   if(iLayer <= 119)
   {
      int i89 = 0;  
   }
#ifdef CUDA_WRAP_CHECK_IN_OUT   
   CUDA_WRAP_CHECK(ny,nz,"start",DETAILS,this,iLayer,p_CellArray);
#endif     
   //CUDA_WRAP_3Dto2D(191,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);
   //CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJxBeam,"A");
   
   
   
#ifdef CUDA_WRAP_FFTW_ALLOWED   
   i = iLayer;
   if (i == 154) {
      double check;
      check = 0;
   }

   if (i == l_Mx-1) maxRho = 0.;
 /*  
   if((GetRank() < GetSize() - 1) && (iLayer == l_Mx -1))
   {
      double *RhoBeam_forP,*JxBeam_forP;
      
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
           Cell &cp = p_CellLayerP[lccc];

           rRhoBeam[n] = cp.GetRhoBeam();
           rJxBeam[n]  = cp.GetJxBeam();
	  }
      }
      fftw_execute(planR2R_RhoBeam);
      fftw_execute(planR2R_JxBeam);
      
   }*/
   

      if((GetRank() < GetSize() - 1) && (iLayer == l_Mx -1) && (iSplit == 0))
      {
	 puts("calling CUDA_WRAP_setBeamFFT");
	 CUDA_WRAP_setBeamFFT(fft_of_JxBeam,fft_of_RhoBeam,l_My*l_Mz);
      }
      for (int n=0; n<ncomplex; n++) {
          fft_of_RhoBeamP[n] = fft_of_RhoBeam[n];
          fft_of_JxBeamP[n]  = fft_of_JxBeam[n];
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	  
          printf("toP %5d %15.5e %15.5e %15.5e %15.5e \n ",n,fft_of_RhoBeamP[n],fft_of_RhoBeam[n],fft_of_JxBeamP[n],fft_of_JxBeam[n]);
#endif	  
      }
      
   
#else
//   copyBeamFourierDataToP_TEMPORARILY_FROM_HOST(l_My*l_Mz,d_fft_of_JxBeamP,d_fft_of_RhoBeamP,
//			            fft_of_JxBeam, fft_of_RhoBeam); 
   copyBeamFourierDataToP(l_My*l_Mz,d_fft_of_JxBeamP,d_fft_of_RhoBeamP,
			            d_fft_of_JxBeam, d_fft_of_RhoBeam); 
#endif
   
#ifdef CUDA_WRAP_FFTW_ALLOWED  
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
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	 
	 if((iLayer <= 61) && (GetRank() == 0)) printf("ini-guess %3d %3d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",j,k,rEx[n],rEy[n],rEz[n],rJx[n],rJy[n],rJz[n],rRho[n]);
#endif	 

         if (rRho[n] > 0.) {
            double check = 0;
         };

         if (i < l_Mx-1 && fabs(fabs(rRho[n]) - dens) > maxRho) {
            maxRho = fabs(fabs(rRho[n]) - dens);
         }

         rJxBeam[n] = c.f_JxBeam;
         rRhoBeam[n] = c.f_RhoBeam;
	 
	 if(fabs(rRhoBeam[n]) > 1e-8)
	 {
	    int iz6 = 0;  
   	 }

         if (i < l_Mx-2 && fabs(cp.f_Dens + dens) > maxRho) maxRho = fabs(cp.f_Dens + dens);

         if (fabs(c.f_JxBeam) > 10) {
            cout << "Too large beam current c.f_JxBeam="<<c.f_JxBeam<<endl;
            exit(20);
         }
      }
   }
#endif   

   //printf("mid copy rank %d \n",GetRank());    


//CUDA_WRAP_copyArraysHost(l_My*l_Mz,rEx,rEy,rEz,rJx,rJy,rJz,rJxBeam,rRhoBeam,rRho,
//			              d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho);
#ifdef CUDA_WRAP_CHECK_IN_OUT  
   CUDA_WRAP_CHECK(ny,nz,"mid copy layer values",DETAILS,this,iLayer,p_CellArray);
#endif 
CUDA_WRAP_copyArraysDevice(l_My*l_Mz,d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho);


#ifdef CUDA_WRAP_CHECK_IN_OUT  
   CUDA_WRAP_CHECK(ny,nz,"after copy layer values",DETAILS,this,iLayer,p_CellArray);
#endif  
   
   if(iLayer == 119)
   {
      cudaLayer *h_cl,*h_pl;
      getLayersPC(&h_cl,&h_pl);
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);   
   }   
   
 //  CUDA_WRAP_setCurrentsToZero(l_My,l_Mz,d_rJx,d_rJy,d_rJz);
   //CUDA_WRAP_restoreLayerCurrents(iLayer+1,l_Mx,l_My,l_Mz,d_rRho,d_rEx,d_rEy,d_rEz);
   gettimeofday(&tv[2],NULL);
   
 //  CUDA_WRAP_3Dto2D(iLayer,l_My,l_Mz,d_RhoBeam3D,d_rRhoBeam);
 //  CUDA_WRAP_3Dto2D(iLayer,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);  
#ifdef CUDA_WRAP_CHECK_IN_OUT  
/*   if(iLayer <= 476)
   {
      int i0 = 0  ;
      CUDA_WRAP_getHiddenCurrents("after assignment",this,iLayer+1,l_My,l_Mz,p_CellArray);
   }*/
   CUDA_WRAP_CHECK(ny,nz,"after main init",DETAILS,this,iLayer,p_CellArray);
#endif   
   
   //CUDA_WRAP_3Dto2D(191,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);
   //CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJxBeam,"BBBB");
   
  // CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rRho,"before copy Rho");
  // CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJx,"before copy Jx");
 //  CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_partRho,"before copy Rho");
 //  CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_partJx,"before copy Jx");

//   CUDA_WRAP_3Dto2D(iLayer,l_My,l_Mz,d_RhoBeam3D,d_rRhoBeam);
//   CUDA_WRAP_3Dto2D(iLayer,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);      

  // CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJxBeam,"beam copied");
   
#ifdef CUDA_WRAP_CHECK_IN_OUT  
   CUDA_WRAP_CHECK(ny,nz,"before 1st fourier",DETAILS,this,iLayer,p_CellArray);
#endif  
   
  /* CUDA_WRAP_emergency_exit("101+0");
   CUDA_WRAP_prepareFFTfromDevice(l_My,l_Mz,9,d_rRhoBeam);
   CUDA_WRAP_emergency_exit("101+1");
   CUDA_WRAP_setGrid(l_My,l_Mz,9,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_emergency_exit("101+2");
   CUDA_WRAP_copyGlobalToSurfaceLayer(l_My,l_Mz,0,d_rRhoBeam,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_copyGlobalToSurfaceLayer(l_My,l_Mz,1,d_rJxBeam,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_copyGlobalToSurfaceLayer(l_My,l_Mz,2,d_rJx,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_copyGlobalToSurfaceLayer(l_My,l_Mz,3,d_rJy,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_copyGlobalToSurfaceLayer(l_My,l_Mz,4,d_rJz,fft_cudaBlock,fft_cudaGrid);
   CUDA_WRAP_emergency_exit("101+3");*/

   double frac_ideal, frac_rude;
   
   //CUDA_WRAP_3Dto2D(191,l_My,l_Mz,d_JxBeam3D,d_rJxBeam);
   //CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rJxBeam,"Jx0");
   gettimeofday(&tv[3],NULL);
   
   if (i == 0) {
      double check = 0.;
   };
   if (maxRho > 0.5) {
      maxRho = 0.5;
   };
   
   //printf("first fourier rank %d \n",GetRank());    
   

//   CUDA_WRAP_fourierHalfInteger2D_fromDevice(int n1,int n2,double *m,double* fres,int flagFFTW_dir1,int flagFFTW_dir2);
   struct timeval tvc1,tvc2,tv1,tv2;
   gettimeofday(&tvc1,NULL);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rRhoBeam,d_rRhoBeam,d_fft_of_RhoBeam,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   gettimeofday(&tvc2,NULL);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rJxBeam,d_rJxBeam,d_fft_of_JxBeam,FFTW_REDFT11,FFTW_REDFT11,iLayer);

#ifdef CUDA_WRAP_FFTW_ALLOWED   
   gettimeofday(&tv1,NULL);
   fftw_execute(planR2R_RhoBeam);
   fftw_execute(planR2R_JxBeam);
   gettimeofday(&tv2,NULL);
#endif   

#ifdef CUDA_WRAP_CHECK_ALL     
      CUDA_WRAP_CHECK(ny,nz,"after 1st fourier",DETAILS,this,iLayer,p_CellArray); 
#endif   
   
    
//   //printf("FFTime fftw %e cuda %e \n",tv2.tv_sec - tv1.tv_sec + 1e-6*(tv2.tv_usec - tv1.tv_usec),tvc2.tv_sec - tvc1.tv_sec + 1e-6*(tvc2.tv_usec - tvc1.tv_usec));

#ifdef CUDA_WRAP_CHECK_IN_OUT  
   CUDA_WRAP_CHECK(ny,nz,"after copy BeamP ",DETAILS,this,iLayer,p_CellArray);
#endif     
   
   
  CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_fft_of_JxBeam,"beam copied");

#ifdef CUDA_WRAP_FFTW_ALLOWED       
   fftw_execute(planR2R_Rho);
   fftw_execute(planR2R_Ex);
   fftw_execute(planR2R_Ey);
   fftw_execute(planR2R_Ez);
   fftw_execute(planR2R_Jx);
   fftw_execute(planR2R_Jy);
   fftw_execute(planR2R_Jz);
#endif   
   gettimeofday(&tv[4],NULL);
   printf("before Fourier block block 3=============================================\n");
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rRho,d_rRho,d_fft_of_Rho,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rEx,d_rEx,d_fft_of_Ex,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rEy,d_rEy,d_fft_of_Ey,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rEz,d_rEz,d_fft_of_Ez,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rJx,d_rJx,d_fft_of_Jx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rJy,d_rJy,d_fft_of_Jy,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rEx,d_rJz,d_fft_of_Jz,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   gettimeofday(&tv[5],NULL);
   printf("end    Fourier block block 3=============================================\n");
   

      
/*      fftw_execute(planR2R_Jx);
      fftw_execute(planR2R_Jy);
      fftw_execute(planR2R_Jz);*/
      
     // CUDA_WRAP_fourierHalfInteger2D(ny,nz,rJx,d_rJx,d_fft_of_Jx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
#ifdef CUDA_WRAP_CHECK_ALL     
      CUDA_WRAP_CHECK(ny,nz,"20nd fourier",DETAILS,this,iLayer,p_CellArray); 
#endif
     // CUDA_WRAP_fourierHalfInteger2D(ny,nz,rJy,d_rJy,d_fft_of_Jy,FFTW_REDFT11,FFTW_RODFT11,iLayer);

      
      
   //printf("20nd fourier rank %d \n",GetRank());    

   //------------------------ linearized E, B ----------------------------
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;

   int kZmax = nz/2;
   int kYmax = ny/2;
   double Ylength = domain()->GetYlength();
   double Zlength = domain()->GetZlength();

   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

   double maxfEx, maxfEy, maxfEz, maxfBx, maxfBy, maxfBz;

   maxfEx = maxfEy = maxfEz = maxfBx = maxfBy = maxfBz = 0.;

   double total_dens = 0.;
   double viscosity = 1e-3; 
   
#ifdef CUDA_WRAP_CHECK_ALL   
   CUDA_WRAP_CHECK(ny,nz,"verification before",DETAILS,this,iLayer,p_CellArray);
#endif   
   
#ifdef CUDA_WRAP_FFTW_ALLOWED
   for (k=0; k<nz; k++)
   {
      for (j=0; j<ny; j++)
      {
         double akz = M_PI/Zlength*(k+0.5);
         double aky = M_PI/Ylength*(j+0.5);
         double ak2 = aky*aky + akz*akz;
         double damp = 1.;

         long n1 = j + ny*k;

         if (ak2==0.) {
            fft_of_Ex[n1] = fft_of_Ey[n1] = fft_of_Ez[n1] = fft_of_Jx[n1] = fft_of_Jy[n1] = fft_of_Jz[n1] = fft_of_Rho[n1] = 0.;
            continue;
         };

         //         VComplex rp  = (1.-maxRho)*vcRhoP[n1] + maxRho*diff_rp*damp;
         double rp  = fft_of_RhoP[n1] = fft_of_Rho[n1]; // + diff_rp*damp;
//         rp = dens;
         double jxp = fft_of_JxP[n1] = fft_of_Jx[n1]; // + diff_jx*damp;
         double jyp = fft_of_JyP[n1] = fft_of_Jy[n1]; // + diff_jy*damp;
         double jzp = fft_of_JzP[n1] = fft_of_Jz[n1]; // + diff_jz*damp;  
         double exp = fft_of_ExP[n1] = fft_of_Ex[n1]; // + diff_jx*damp;
         double eyp = fft_of_EyP[n1] = fft_of_Ey[n1]; // + diff_jy*damp;
         double ezp = fft_of_EzP[n1] = fft_of_Ez[n1]; // + diff_jz*damp;  
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG
	 if((iLayer <= 61) && (GetRank() == 0)) printf("loop-guess %3d %3d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",j,k,rp,jxp,jyp,jzp,exp,eyp,ezp);
#endif	 

         double rb  = fft_of_RhoBeam[n1];
         double jxb = fft_of_JxBeam[n1];
         double rbp  = fft_of_RhoBeamP[n1];
         double jxbp = fft_of_JxBeamP[n1];
         double h = hx;

         double propagator = (4.-dens*hx*hx)/(4.+dens*hx*hx);
         double denominator = 4.+dens*hx*hx;
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	 
	 if((iLayer <= 61) && (GetRank() == 0)) printf("loop-guess1 %3d %3d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",j,k,rb,jxb,rbp,jxbp,h,propagator,denominator);
#endif	 

         fft_of_Rho[n1] = propagator*rp - (rb+rbp)*dens*hx*hx/denominator + 4.*exp*hx*(ak2+dens)/denominator;
         fft_of_Ex[n1] = propagator*exp - 2.*(2.*rp+rb+rbp)*dens*hx/((ak2+dens)*denominator);

//         fft_of_Ey[n1] = -(dens*eyp + eyp*ak2 - aky*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);
//         fft_of_Ez[n1] = -(dens*ezp + ezp*ak2 - akz*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);

         fft_of_Ey[n1] = -eyp + aky*(rb+rbp+rp+fft_of_Rho[n1])/(dens+ak2);
         fft_of_Ez[n1] = -ezp + akz*(rb+rbp+rp+fft_of_Rho[n1])/(dens+ak2);

         fft_of_Jy[n1] = jyp + hx*dens*(fft_of_Ey[n1] + eyp)/2.;
         fft_of_Jz[n1] = jzp + hx*dens*(fft_of_Ez[n1] + ezp)/2.;

         fft_of_Jx[n1] = jxp + hx*dens*(fft_of_Ex[n1] + exp)/2.;

         double newEy = 0.;
         double newEz = 0.;

         newEy = -eyp + (aky*(rb+rbp+fft_of_Rho[n1]+rp) + 2.*(jyp-fft_of_Jy[n1])/hx)/ak2;
         newEz = -ezp + (akz*(rb+rbp+fft_of_Rho[n1]+rp) + 2.*(jzp-fft_of_Jz[n1])/hx)/ak2;

         fft_of_Bx[n1] = -aky/ak2*fft_of_Jz[n1] + akz/ak2*fft_of_Jy[n1];
         fft_of_By[n1] = (-akz*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) + dens*fft_of_Ez[n1])/ak2;
	 ////printf("host   %5d %e %e %e \n",n1,fft_of_Jx[n1],fft_of_JxBeam[n1],fft_of_Ez[n1]);
         fft_of_Bz[n1] =  (aky*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) - dens*fft_of_Ey[n1])/ak2;

         if (fft_of_Ex[n1] != 0. && k==l_Mz/2 && j==l_My/2) {
            double dummy = 0.;
         }
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG         
         if((iLayer <= 61) && (GetRank() == 0)) printf("loop-res %3d %3d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",j,k,
	                                                                                                                           fft_of_Rho[n1],fft_of_Ex[n1],fft_of_Ey[n1],fft_of_Ez[n1],
                                                                                                                                                  fft_of_Bx[n1],fft_of_By[n1],fft_of_Bz[n1],	                                                                                                                            
                                                                                                                                                  fft_of_Jx[n1],fft_of_Jy[n1],fft_of_Jz[n1]						       
	);
#endif	 
//         fft_of_Bz[n1] =  aky*fft_of_JxBeam[n1]/ak2;

 
         if (fabs(fft_of_Ex[n1]) > maxfEx) maxfEx = fabs(fft_of_Ex[n1]);
         if (fabs(fft_of_Ey[n1]) > maxfEy) maxfEy = fabs(fft_of_Ey[n1]);
         if (fabs(fft_of_Ez[n1]) > maxfEz) maxfEz = fabs(fft_of_Ez[n1]);

         if (fabs(fft_of_Bx[n1]) > maxfBx) maxfBx = fabs(fft_of_Bx[n1]);
         if (fabs(fft_of_By[n1]) > maxfBy) maxfBy = fabs(fft_of_By[n1]);
         if (fabs(fft_of_Bz[n1]) > maxfBz) maxfBz = fabs(fft_of_Bz[n1]);

         if (maxfEx > 1e-5 || maxfEy > 1e-5 || maxfEz > 1e-5 || maxfBx > 1e-5 || maxfBy > 1e-5 || maxfBz > 1e-5) {
            double check = 0.;
         };

      }
   }
#endif   
   gettimeofday(&tv[6],NULL);
   //printf("before lin loop %d \n",GetRank());    
   
   
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(ny,nz,"before lin loop",DETAILS,this,iLayer,p_CellArray);
#endif   

//   if(iLayer == 471)
//   {
      timeBegin(1);  
      CUDA_WRAP_linearized_loop(ny,nz,hx,dens,Zlength,Ylength,
                                d_fft_of_Rho,
                                d_fft_of_RhoP,
                                d_fft_of_JxP,
                                d_fft_of_JyP,
                                d_fft_of_JzP,
                                d_fft_of_Ex,
                                d_fft_of_Ey,
                                d_fft_of_Ez,
                                d_fft_of_ExP,
                                d_fft_of_EyP,
                                d_fft_of_EzP,
                                d_fft_of_Jx,
                                d_fft_of_Jy,
                                d_fft_of_Jz,
                                d_fft_of_Bx,
                                d_fft_of_By,
                                d_fft_of_Bz,
                                d_fft_of_JxBeam,
                                d_fft_of_RhoBeam,
                                d_fft_of_JxBeamP,
                                d_fft_of_RhoBeamP);
      timeEnd(1);
   //   puts("out lin list");
//   }

#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(ny,nz,"after lin loop",DETAILS,this,iLayer,p_CellArray);
#endif   
   gettimeofday(&tv[7],NULL);

   //------------------------ transform to configuration space E, B ----------------------------
#ifdef CUDA_WRAP_FFTW_ALLOWED
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Rho[n];
   fftw_execute(planR2Rb_Rho);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ex[n];
   fftw_execute(planR2Rb_Ex);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ey[n];
   fftw_execute(planR2Rb_Ey);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ez[n];
   fftw_execute(planR2Rb_Ez);

   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Jx[n];
   fftw_execute(planR2Rb_Jx);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Jy[n];
   fftw_execute(planR2Rb_Jy);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Jz[n];
   fftw_execute(planR2Rb_Jz);

   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Bx[n];
   fftw_execute(planR2Rb_Bx);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_By[n];
   fftw_execute(planR2Rb_By);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Bz[n];
   fftw_execute(planR2Rb_Bz);
#endif   
   gettimeofday(&tv[8],NULL);
   
#ifdef CUDA_WRAP_CHECK_ALL   
   CUDA_WRAP_CHECK(ny,nz,"inverse Fourier",DETAILS,this,iLayer,p_CellArray);
#endif
   //printf("inverse fourier rank %d \n",GetRank());    
     
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Rho,d_fft_of_Rho,d_rRho,FFTW_REDFT11,FFTW_REDFT11,iLayer); 
   
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Ex,d_fft_of_Ex,d_rEx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Ey,d_fft_of_Ey,d_rEy,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Ez,d_fft_of_Ez,d_rEz,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Bx,d_fft_of_Bx,d_rBx,FFTW_RODFT11,FFTW_RODFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_By,d_fft_of_By,d_rBy,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Bz,d_fft_of_Bz,d_rBz,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Jx,d_fft_of_Jx,d_rJx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Jy,d_fft_of_Jy,d_rJy,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   gettimeofday(&tv[9],NULL);
   CUDA_WRAP_fourierHalfInteger2D(ny,nz,fft_of_Jz,d_fft_of_Jz,d_rJz,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   
   
#ifdef CUDA_WRAP_CHECK_ALL   
   CUDA_WRAP_CHECK(ny,nz,"after inverse fourier",DETAILS,this,iLayer,p_CellArray);
#endif   
   gettimeofday(&tv[10],NULL);

   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;
   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 1.e-5;
   double difEx, difEy, difEz, difBx, difBy, difBz;
   difEx = difEy = difEz = difBx = difBy = difBz = 0.;
   
#ifdef CUDA_WRAP_FFTW_ALLOWED
   for (k=0; k<nz; k++)
   {
      for (j=0; j<ny; j++)
      {
         long n = j + l_My*k;

         rRho[n] = rRho[n]/(4.*ny*nz);

         rEx[n] = rEx[n]/(4.*ny*nz);
         rEy[n] = rEy[n]/(4.*ny*nz);
         rEz[n] = rEz[n]/(4.*ny*nz);

         rJx[n] = rJx[n]/(4.*ny*nz);
         rJy[n] = rJy[n]/(4.*ny*nz);
         rJz[n] = rJz[n]/(4.*ny*nz);

         rBx[n] = rBx[n]/(4.*ny*nz);
//         rBx[n] = 0.;
         rBy[n] = rBy[n]/(4.*ny*nz);
         rBz[n] = rBz[n]/(4.*ny*nz);

         sumEx += rEx[n]*rEx[n];
         sumEy += rEy[n]*rEy[n];
         sumEz += rEz[n]*rEz[n];
         sumBx += rBx[n]*rBx[n];
         sumBy += rBy[n]*rBy[n];
         sumBz += rBz[n]*rBz[n];

         long lccc = GetNyz(j,k);

         Cell &ccc = p_CellLayerC[lccc];

         ccc.f_Dens = rRho[n]; // + dens;

         ccc.f_Jx = rJx[n];
         ccc.f_Jy = rJy[n];
         ccc.f_Jz = rJz[n];

         ccc.f_Ex = rEx[n];
         ccc.f_Ey = rEy[n];
         ccc.f_Ez = rEz[n];

         ccc.f_Bx = rBx[n];
         ccc.f_By = rBy[n];
         ccc.f_Bz = rBz[n];

         if (fabs(rEx[n]) > maxEx) {
            maxEx = fabs(rEx[n]);
         }
         if (fabs(rEy[n]) > maxEy) maxEy = fabs(rEy[n]);
         if (fabs(rEz[n]) > maxEz) maxEz = fabs(rEz[n]);

         if (fabs(rBx[n]) > maxBx) maxBx = fabs(rBx[n]);
         if (fabs(rBy[n]) > maxBy) maxBy = fabs(rBy[n]);
         if (fabs(rBz[n]) > maxBz) maxBz = fabs(rBz[n]);

      }
   }
#endif   
   
#ifdef CUDA_WRAP_CHECK_ALL   
   CUDA_WRAP_CHECK(ny,nz,"before norm loop",DETAILS,this,iLayer,p_CellArray);
#endif
//#endif
   
   timeBegin(4);
   normalizationLoop(ny,nz,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,d_rJx,d_rJy,d_rJz,d_rRho);
   timeEnd(4);
   
 //  CUDA_WRAP_copyLayerCurrents(iLayer,l_Mx,l_My,l_Mz,d_rRho,d_rEx,d_rEy,d_rEz);
 //  CUDA_WRAP_copyLayerFields(iLayer,l_Mx,l_My,l_Mz,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz);
   gettimeofday(&tv[11],NULL);

#ifdef CUDA_WRAP_CHECK_ALL   
   CUDA_WRAP_CHECK(ny,nz,"after norm loop",DETAILS,this,iLayer,p_CellArray);
#endif 
   if(iLayer == 119) CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,d_rEx,"R");
    cudaLayer *h_cl,*h_pl;
   getLayersPC(&h_cl,&h_pl);
   if(iLayer == 119)
   {
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);   
   }   
   CUDA_WRAP_storeArraysToDevice(l_My,l_Mz,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,d_rJx,d_rJy,d_rJz,d_rRho);
  
   if(iLayer == 119)
   {
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);   
   }


//   cout << "Guess: maxEx =" << maxEx <<"maxEy =" << maxEy <<"maxEz =" << maxEz << endl;
   if (maxEx > 100 || maxEy > 100 || maxEz > 100) {
      //      cout << "Large real fields! \n";
   }

   if (maxEx > 1e-5 || maxEy > 1e-5 || maxEz > 1e-5 || maxBx > 1e-5 || maxBy > 1e-5 || maxBz > 1e-5) {
      double check = 0.;
   };

   if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 1e4) {
      double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
      sum += 0;
   };


   if (i == 300) {
      double check;
      check = 0.;
   }
   //printf("end-guess rank %d \n",GetRank());    

   /*
   cout << "Max Rho="<<maxRho<<endl;
   cout << "Max Jy="<<max_Jy<<" Max dJy="<<max_dJy<<endl;
   cout << "Max Ey="<<max_Ey<<" Max dEy="<<max_dEy<<endl;
   */
   //   domain()->Exchange(SPACK_F);
   /*
   fftw_free(in);
   fftw_free(in1);
   fftw_free(in2);
   fftw_free(in3);
   fftw_free(vcOut);
   fftw_free(vcOut1);
   fftw_free(vcAc);
   fftw_free(vcAm);
   fftw_free(vcFp);
   fftw_free(vcFc);
   fftw_free(vcFm);
   fftw_free(vcFnext);
   fftw_destroy_plan(planR2C);
   fftw_destroy_plan(planC2R);
   fftw_destroy_plan(planR2C1);
   fftw_destroy_plan(planC2R1);
   fftw_destroy_plan(planR2C2);
   fftw_destroy_plan(planC2R2);
   fftw_destroy_plan(planR2C3);
   fftw_destroy_plan(planC2R3);
   fftw_destroy_plan(planR2C11);
   fftw_destroy_plan(planC2R11);
   */
   
 //  CUDA_WRAP_output_host_matrix(l_My,l_Mz,"EY",iLayer,rEy);
   
//#ifdef CUDA_WRAP_CHECK_IN_OUT   
   CUDA_WRAP_CHECK(ny,nz,"end",DETAILS,this,iLayer,p_CellArray);
//#endif
      for (int n=0; n<ncomplex; n++) {
          //fft_of_RhoBeamP[n] = fft_of_RhoBeam[n];
          //fft_of_JxBeamP[n]  = fft_of_JxBeam[n];
          //printf("ENDtoP %5d %15.5e %15.5e %15.5e %15.5e \n ",n,fft_of_RhoBeamP[n],fft_of_RhoBeam[n],fft_of_JxBeamP[n],fft_of_JxBeam[n]);
      }
     //if((GetRank() == 0) ) exit(0);
   
#ifdef GUESS_TIME_PRINT   
 //  CUDA_WRAP_output_fields(l_My,l_Mz,iLayer,d_rEx,d_rEy,d_rEz);
   double sum = 0.0;
   for(int i = 1;i < CUDA_WRAP_TN;i++)
   {
       double t = tv[i].tv_sec - tv[i-1].tv_sec + 1e-6*(tv[i].tv_usec - tv[i-1].tv_usec);
       sum += t;
       //printf("              stage %2d time %15.5e total %15.5e \n",i,t,sum);  
   }
#endif   
}


//--- Mesh:: ----------------------.
double Mesh::IterateFieldsHydroLinLayerSplit(int iLayer,int iSplit,int N_iter)
{
   int i, j, k;
   double max_dJy = 0.;
   double max_Jy = 0.;
   double max_dEy = 0.;
   double max_Ey = 0.;
   double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;
   
   if(iLayer <= 119 && iSplit >= 0 && N_iter >= 0)
   {
      int z = 0;  
   }
   
   double dens = 0.;
#ifdef CUDA_WRAP_FFTW_ALLOWED      
   i = l_Mx-1;;
   j = l_My/2.;
   k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;
   double zco = Z(k) - domain()->GetZlength()/2.;

   
   int nsorts = domain()->GetNsorts();
   for (int isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += fabs(spec->Density(xco,yco,zco)*spec->GetQ2M());
   };
#endif 
   

   if (dens <= 0.) dens = 1.;

   double ts = Ts();
   double hx = HxSplit();
   double h = hx;
   double hy = Hy();
   double hz = Hz();
  
   
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate start",DETAILS,this,iLayer,p_CellArray);
#endif      

   VComplex I = VComplex(0.,1.);
   //   I.re = 0.;
   //   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int ncomplex = nz*ny;

   if (FirstCall == 0) {
      cout << "Error from IterateFieldsHydroLinLayer: data not initialized! \n";
      exit (-1);
   };

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;

   double maxJx, maxJy, maxJz, difJx, difJy, difJz;
   maxJx = maxJy = maxJz = difJx = difJy = difJz = 1e-15;
   
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate before copy",DETAILS,this,iLayer,p_CellArray);
#endif      
   
#ifdef CUDA_WRAP_FFTW_ALLOWED   
   i = iLayer;
   if (i == 154) {
      double check;
      check = 0;
   }

   if (i == l_Mx-1) maxRho = 0.;

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

         if (fabs(c.f_Jx - rJx[n]) > difJx) difJx = fabs(c.f_Jx - rJx[n]);
         if (fabs(c.f_Jy - rJy[n]) > difJy) difJy = fabs(c.f_Jy - rJy[n]);
         if (fabs(c.f_Jz - rJz[n]) > difJz) difJz = fabs(c.f_Jz - rJz[n]);

         if (fabs(rJx[n]) > maxJx) maxJx = fabs(rJx[n]);
         if (fabs(rJy[n]) > maxJy) maxJy = fabs(rJy[n]);
         if (fabs(rJz[n]) > maxJz) maxJz = fabs(rJz[n]);

         rRho[n] = c.f_Dens+1;
         rJx[n] = c.f_Jx;
         rJy[n] = c.f_Jy;
         rJz[n] = c.f_Jz;

         double diffJx_exp = -dens*(c.f_Ex + cp.f_Ex)/2.;
         double diffJy_exp = -dens*(c.f_Ey + cp.f_Ey)/2.;
         double diffJz_exp = -dens*(c.f_Ez + cp.f_Ez)/2.;

         double diffJx_r = (cp.f_Jx - c.f_Jx)/hx;
         double diffJy_r = (cp.f_Jy - c.f_Jy)/hx;
         double diffJz_r = (cp.f_Jz - c.f_Jz)/hx;

         if (diffJx_r != 0. && k==l_Mz/2 && j==l_My/2) {
            double dummy = 0.;
         }

         rJxBeam[n] = c.f_JxBeam;
         rRhoBeam[n] = c.f_RhoBeam;

         if (i < l_Mx-2 && fabs(cp.f_Dens + dens) > maxRho) maxRho = fabs(cp.f_Dens + dens);

         if (fabs(c.f_JxBeam) > 10) {
            cout << "Too large beam current c.f_JxBeam="<<c.f_JxBeam<<endl;
            exit(20);
         }
      }
   }
#endif   
//   CUDA_WRAP_copyArraysHost(l_My*l_Mz,rEx,rEy,rEz,rJx,rJy,rJz,rJxBeam,rRhoBeam,rRho,
//			              d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho);
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate mid copy",DETAILS,this,iLayer,p_CellArray);
#endif 
   
#ifndef CUDA_WRAP_FFTW_ALLOWED   
   CUDA_WRAP_copyArraysDeviceIterate(l_My,l_Mz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho); 
#endif   
   
   CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rJx,"a copy");
//    CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rJy,"a copy");
//    CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rJz,"a copy");
//    CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rJxBeam,"a copy");
//    CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rRhoBeam,"a copy");
//    CUDA_DEBUG_printDdevice_matrixCentre(l_My,l_Mz,d_rRho,"a copy");

   
   if(iLayer == 119)
   {
     int z = 0;  
   }
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate after copy",DETAILS,this,iLayer,p_CellArray);
#endif      


   if (i == 0) {
      double check = 0.;
   };

   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rJx,d_rJx,d_fft_of_Jx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rJy,d_rJy,d_fft_of_Jy,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,rEx,d_rJz,d_fft_of_Jz,FFTW_RODFT11,FFTW_REDFT11,iLayer);

#ifdef CUDA_WRAP_FFTW_ALLOWED   
   fftw_execute(planR2R_Jx);
   fftw_execute(planR2R_Jy);
   fftw_execute(planR2R_Jz);
#endif   
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate after first fourier",DETAILS,this,iLayer,p_CellArray);
#endif      

   //------------------------ linearized E, B ----------------------------
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;

   int kZmax = nz/2;
   int kYmax = ny/2;
   double Ylength = domain()->GetYlength();
   double Zlength = domain()->GetZlength();

   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

   double maxfEx, maxfEy, maxfEz, maxfBx, maxfBy, maxfBz;

   maxfEx = maxfEy = maxfEz = maxfBx = maxfBy = maxfBz = 0.;
   double errorEx = 0.;

   double total_dens = 0.;
   double viscosity = 1e-3; 
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate before lin loop",DETAILS,this,iLayer,p_CellArray);
#endif      

#ifdef CUDA_WRAP_FFTW_ALLOWED      
   for (k=0; k<nz; k++)
   {
      for (j=0; j<ny; j++)
      {
         double akz = M_PI/Zlength*(k+0.5);
         double aky = M_PI/Ylength*(j+0.5);
         double ak2 = aky*aky + akz*akz;
         double damp = 1.;

         long n1 = j + ny*k;

         if (ak2==0.) {
            fft_of_Ex[n1] = fft_of_Ey[n1] = fft_of_Ez[n1] = fft_of_Jx[n1] = fft_of_Jy[n1] = fft_of_Jz[n1] = fft_of_Rho[n1] = 0.;
            continue;
         };

         double diff_jx = (fft_of_JxP[n1] - fft_of_Jx[n1])/hx;
         double diff_jy = (fft_of_JyP[n1] - fft_of_Jy[n1])/hx;
         double diff_jz = (fft_of_JzP[n1] - fft_of_Jz[n1])/hx;

         if (diff_jx != 0.) {
            fftDensExpected[n1] = -2.*diff_jx/(fft_of_ExP[n1] + fft_of_Ex[n1]);
            fftDensExpected[n1] = max(fabs(dens)/2.,fftDensExpected[n1]);
            fftDensExpected[n1] = min(fabs(dens)*2.,fftDensExpected[n1]);
         };


/*
         if (fabs(fft_of_ExP[n1] + fft_of_Ex[n1]) > 1e-5) {
            dens = -2.*diff_jx/(fft_of_ExP[n1] + fft_of_Ex[n1]);
         } else {
            dens = dens;
         };
*/

         //         VComplex rp  = (1.-maxRho)*vcRhoP[n1] + maxRho*diff_rp*damp;
         double rp  = fft_of_RhoP[n1]; // 
//         rp = dens;
         double jx = fft_of_Jx[n1]; //
         double jy = fft_of_Jy[n1]; //
         double jz = fft_of_Jz[n1]; //  
         double jxp = fft_of_JxP[n1]; //
         double jyp = fft_of_JyP[n1]; //
         double jzp = fft_of_JzP[n1]; //  

         double rho  = rp + fft_of_Jx[n1] - fft_of_JxP[n1] 
            - hx*(aky*(jy+fft_of_JyP[n1]) + akz*(jz+fft_of_JzP[n1]))/2.;
/*
            rho  = fft_of_Rho[n1];

         jx = fft_of_Jx[n1] = jxp + fft_of_Rho[n1] - fft_of_RhoP[n1] 
            + hx*(aky*(jy+fft_of_JyP[n1]) + akz*(jz+fft_of_JzP[n1]))/2.;;
            */
         double diffRho = fft_of_Rho[n1] - rho;


         double rb  = fft_of_RhoBeam[n1];
         double rbp = fft_of_RhoBeamP[n1];
         double jxb  = fft_of_JxBeam[n1];
         double jxbp = fft_of_JxBeamP[n1];
         double eyp = fft_of_EyP[n1];
         double ezp = fft_of_EzP[n1];
         double ey = fft_of_Ey[n1];
         double ez = fft_of_Ez[n1];

         double newEx = -(aky*jy + akz*jz)/ak2;
         double newEy = aky*(rb+rho)/(dens+ak2);
         double newEz = akz*(rb+rho)/(dens+ak2);

         newEy = -eyp + aky*(rb+rbp+rp+fft_of_Rho[n1])/(dens+ak2);
         newEz = -ezp + akz*(rb+rbp+rp+fft_of_Rho[n1])/(dens+ak2);

//         double newEy = -eyp + (aky*(rb+rbp+rho+rp) + 2.*(jyp-jy)/hx)/ak2;
//         double newEz = -ezp + (akz*(rb+rbp+rho+rp) + 2.*(jzp-jz)/hx)/ak2;

         if (newEx != 0. && k==l_Mz/2 && j==l_My/2) {
            double dummy = 0.;
         }
         fft_of_Ey[n1] = newEy;
         fft_of_Ez[n1] = newEz;

         fft_of_Rho[n1] = rho;

//         fft_of_Ey[n1] = -(dens*eyp + eyp*ak2 - aky*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);
//         fft_of_Ez[n1] = -(dens*ezp + ezp*ak2 - akz*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);

         errorEx += (newEx - fft_of_Ex[n1])*(newEx - fft_of_Ex[n1]);
         fft_of_Ex[n1] = newEx;
//         fft_of_Ex[n1] = 0.5*(newEx + fft_of_Ex[n1]);

         fft_of_Bx[n1] = -aky/ak2*fft_of_Jz[n1] + akz/ak2*fft_of_Jy[n1];
         fft_of_By[n1] = (-akz*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) + dens*newEz)/ak2;
         fft_of_Bz[n1] =  (aky*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) - dens*newEy)/ak2;

         if (fabs(fft_of_Ex[n1]) > maxfEx) maxfEx = fabs(fft_of_Ex[n1]);
         if (fabs(fft_of_Ey[n1]) > maxfEy) maxfEy = fabs(fft_of_Ey[n1]);
         if (fabs(fft_of_Ez[n1]) > maxfEz) maxfEz = fabs(fft_of_Ez[n1]);

         if (maxfEx > 1e-5 || maxfEy > 1e-5 || maxfEz > 1e-5 || maxfBx > 1e-5 || maxfBy > 1e-5 || maxfBz > 1e-5) {
            double check = 0.;
         };

      }
   }
#endif

   CUDA_WRAP_linearizedIterateloop(l_My,l_Mz,hx,dens,Zlength,Ylength,
                              d_fft_of_Rho,
                              d_fft_of_RhoP,
                              d_fft_of_JxP,
                              d_fft_of_JyP,
                              d_fft_of_JzP,
                              d_fft_of_Ex,
                              d_fft_of_Ey,
                              d_fft_of_Ez,
                              d_fft_of_EyP,
                              d_fft_of_EzP,
                              d_fft_of_Jx,
                              d_fft_of_Jy,
                              d_fft_of_Jz,
                              d_fft_of_Bx,
                              d_fft_of_By,
                              d_fft_of_Bz,
                              d_fft_of_JxBeam,
                              d_fft_of_RhoBeam,
                              d_fft_of_JxBeamP,
                              d_fft_of_RhoBeamP);
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate after lin loop",DETAILS,this,iLayer,p_CellArray);
#endif      
   

   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Rho,d_fft_of_Rho,d_rRho,FFTW_REDFT11,FFTW_REDFT11,iLayer); 
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Ex,d_fft_of_Ex,d_rEx,FFTW_REDFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Ey,d_fft_of_Ey,d_rEy,FFTW_REDFT11,FFTW_RODFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Ez,d_fft_of_Ez,d_rEz,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Bx,d_fft_of_Bx,d_rBx,FFTW_RODFT11,FFTW_RODFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_By,d_fft_of_By,d_rBy,FFTW_RODFT11,FFTW_REDFT11,iLayer);
   CUDA_WRAP_fourierHalfInteger2D(l_My,l_Mz,fft_of_Bz,d_fft_of_Bz,d_rBz,FFTW_REDFT11,FFTW_RODFT11,iLayer);   
   
   //------------------------ transform to configuration space E, B ----------------------------
#ifdef CUDA_WRAP_FFTW_ALLOWED
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Rho[n];
   fftw_execute(planR2Rb_Rho);

   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ex[n];
   fftw_execute(planR2Rb_Ex);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ey[n];
   fftw_execute(planR2Rb_Ey);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ez[n];
   fftw_execute(planR2Rb_Ez);
  
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Bx[n];
   fftw_execute(planR2Rb_Bx);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_By[n];
   fftw_execute(planR2Rb_By);
   for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Bz[n];
   fftw_execute(planR2Rb_Bz);
#endif   
#ifdef CUDA_WRAP_CHECK_ALL
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate after 2nd fourier",DETAILS,this,iLayer,p_CellArray);
#endif      

   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;
   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 1.e-5;
   double difEx, difEy, difEz, difBx, difBy, difBz;
   difEx = difEy = difEz = difBx = difBy = difBz = 0.;

#ifdef CUDA_WRAP_FFTW_ALLOWED      
   for (k=0; k<nz; k++)
   {
      for (j=0; j<ny; j++)
      {
         long n = j + l_My*k;

         rRho[n] = rRho[n]/(4.*ny*nz);

         rEx[n] = rEx[n]/(4.*ny*nz);
         rEy[n] = rEy[n]/(4.*ny*nz);
         rEz[n] = rEz[n]/(4.*ny*nz);

         rBx[n] = rBx[n]/(4.*ny*nz);
//         rBx[n] = 0.;
         rBy[n] = rBy[n]/(4.*ny*nz);
         rBz[n] = rBz[n]/(4.*ny*nz);

         sumEx += rEx[n]*rEx[n];
         sumEy += rEy[n]*rEy[n];
         sumEz += rEz[n]*rEz[n];
         sumBx += rBx[n]*rBx[n];
         sumBy += rBy[n]*rBy[n];
         sumBz += rBz[n]*rBz[n];

         if (fabs(rEx[n]) > maxEx) {
            maxEx = fabs(rEx[n]);
         }
         if (fabs(rEy[n]) > maxEy) maxEy = fabs(rEy[n]);
         if (fabs(rEz[n]) > maxEz) maxEz = fabs(rEz[n]);

         if (fabs(rBx[n]) > maxBx) maxBx = fabs(rBx[n]);
         if (fabs(rBy[n]) > maxBy) maxBy = fabs(rBy[n]);
         if (fabs(rBz[n]) > maxBz) maxBz = fabs(rBz[n]);

         long lccc = GetNyz(j,k);

         Cell &ccc = p_CellLayerC[lccc];

         if (fabs(ccc.f_Ex - rEx[n]) > difEx) difEx = fabs(ccc.f_Ex - rEx[n]);
         if (fabs(ccc.f_Ey - rEy[n]) > difEy) difEy = fabs(ccc.f_Ey - rEy[n]);
         if (fabs(ccc.f_Ez - rEz[n]) > difEz) difEz = fabs(ccc.f_Ez - rEz[n]);

         if (fabs(ccc.f_Bx - rBx[n]) > difBx) difBx = fabs(ccc.f_Bx - rBx[n]);
         if (fabs(ccc.f_By - rBy[n]) > difBy) difBy = fabs(ccc.f_By - rBy[n]);
         if (fabs(ccc.f_Bz - rBz[n]) > difBz) difBz = fabs(ccc.f_Bz - rBz[n]);

         ccc.f_Dens = rRho[n];

         ccc.f_Ex = rEx[n];
         ccc.f_Ey = rEy[n];
         ccc.f_Ez = rEz[n];

         ccc.f_Bx = rBx[n];
         ccc.f_By = rBy[n];
         ccc.f_Bz = rBz[n];
      }
   }
#endif   
   if(iLayer == 119)
   {
     int z = 0;  
   }
   normalizationIterateLoop(l_My,l_Mz,d_rRho,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz);
#ifdef CUDA_WRAP_CHECK_ALL      
   CUDA_WRAP_CHECK(l_My,l_Mz,"Iterate after norm loop",DETAILS,this,iLayer,p_CellArray);
#endif      
   
   CUDA_WRAP_storeArraysToDeviceC(l_My,l_Mz,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,d_rRho);
//   cout << "Iterate: maxEx =" << maxEx <<"maxEy =" << maxEy <<"maxEz =" << maxEz << endl;

   if (maxEx > 1e-5 || maxEy > 1e-5 || maxEz > 1e-5 || maxBx > 1e-5 || maxBy > 1e-5 || maxBz > 1e-5) {
      double check = 0.;
   };

   double Eerr = max(difEx/maxEx, difEy/maxEy); 
   Eerr = max(Eerr, difEz/maxEz); 

   if (maxEx > 100 || maxEy > 100 || maxEz > 100) {
      //      cout << "Large real fields! \n";
   }

   if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 1e4) {
      double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
      sum += 0;
   };


   if (i == 300) {
      double check;
      check = 0.;
   }

   /*
   cout << "Max Rho="<<maxRho<<endl;
   cout << "Max Jy="<<max_Jy<<" Max dJy="<<max_dJy<<endl;
   cout << "Max Ey="<<max_Ey<<" Max dEy="<<max_dEy<<endl;
   */
   //   domain()->Exchange(SPACK_F);
   /*
   fftw_free(in);
   fftw_free(in1);
   fftw_free(in2);
   fftw_free(in3);
   fftw_free(vcOut);
   fftw_free(vcOut1);
   fftw_free(vcAc);
   fftw_free(vcAm);
   fftw_free(vcFp);
   fftw_free(vcFc);
   fftw_free(vcFm);
   fftw_free(vcFnext);
   fftw_destroy_plan(planR2C);
   fftw_destroy_plan(planC2R);
   fftw_destroy_plan(planR2C1);
   fftw_destroy_plan(planC2R1);
   fftw_destroy_plan(planR2C2);
   fftw_destroy_plan(planC2R2);
   fftw_destroy_plan(planR2C3);
   fftw_destroy_plan(planC2R3);
   fftw_destroy_plan(planR2C11);
   fftw_destroy_plan(planC2R11);
   */

//   cout <<"Layer=" << iLayer <<" Eerr="<<Eerr << endl;
   return Eerr;
}

