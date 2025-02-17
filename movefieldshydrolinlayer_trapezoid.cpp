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

//--- Mesh:: ----------------------.
void Mesh::GuessFieldsHydroLinLayer(int iLayer)
{
//   cout << "Layer=" << iLayer <<endl;

   int i, j, k;
   double max_dJy = 0.;
   double max_Jy = 0.;
   double max_dEy = 0.;
   double max_Ey = 0.;
   double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;

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
   double hx = Hx();
   double hy = Hy();
   double hz = Hz();

   VComplex I = VComplex(0.,1.);
   //   I.re = 0.;
   //   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int ncomplex = nz*ny;

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
      carray = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_ExRho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EyRho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_EzRho = (double*)fftw_malloc(ncomplex*sizeof(double));
      fftDensExpected = (double*)fftw_malloc(ncomplex*sizeof(double));;
/*
      int gotthreads = fftw_init_threads();
      int nthreads = 2;
     if (gotthreads == 0) {
         cout << "Could not init threads! \n";
         nthreads = 1;
      };

      fftw_plan_with_nthreads(nthreads);  
      */

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

   if (iLayer > l_Mx-2) {
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

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;


   double maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;

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
         long lccc = GetN(i,j,k);
         Cell &c = p_CellArray[lccc];
         Cell &cp = p_CellArray[lccc+1];

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

         if (i < l_Mx-2 && fabs(cp.f_Dens + dens) > maxRho) maxRho = fabs(cp.f_Dens + dens);

         if (fabs(c.f_JxBeam) > 10) {
            cout << "Too large beam current c.f_JxBeam="<<c.f_JxBeam<<endl;
            exit(20);
         }
      }
   }

   if (i == 0) {
      double check = 0.;
   };
   if (maxRho > 0.5) {
      maxRho = 0.5;
   };

   fftw_execute(planR2R_RhoBeam);
   fftw_execute(planR2R_JxBeam);

   fftw_execute(planR2R_Rho);
   fftw_execute(planR2R_Ex);
   fftw_execute(planR2R_Ey);
   fftw_execute(planR2R_Ez);
   fftw_execute(planR2R_Jx);
   fftw_execute(planR2R_Jy);
   fftw_execute(planR2R_Jz);

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

         double rb  = fft_of_RhoBeam[n1];
         double jxb = fft_of_JxBeam[n1];
         double h = hx;

         double propagator = (1.-fftDensExpected[n1]*hx*hx/4.)/(1.+fftDensExpected[n1]*hx*hx/4.);
         double denominator = 1.+fftDensExpected[n1]*hx*hx/4.;

         fft_of_Rho[n1] = propagator*rp - rb*fftDensExpected[n1]*hx*hx/(2.*denominator) + exp*hx*(ak2+fftDensExpected[n1])/denominator;
         fft_of_Ex[n1] = propagator*exp - (rp+rb)*fftDensExpected[n1]*hx/((ak2+fftDensExpected[n1])*denominator);


         fft_of_Ey[n1] = -(fftDensExpected[n1]*eyp + eyp*ak2 - aky*(rp+2.*rb+fft_of_Rho[n1]))/(fftDensExpected[n1]+ak2);
         fft_of_Ez[n1] = -(fftDensExpected[n1]*ezp + ezp*ak2 - akz*(rp+2.*rb+fft_of_Rho[n1]))/(fftDensExpected[n1]+ak2);

         fft_of_Jy[n1] = jyp + hx*fftDensExpected[n1]*(fft_of_Ey[n1] + eyp)/2.;
         fft_of_Jz[n1] = jzp + hx*fftDensExpected[n1]*(fft_of_Ez[n1] + ezp)/2.;

         fft_of_Jx[n1] = jxp + hx*fftDensExpected[n1]*(fft_of_Ex[n1] + exp)/2.;

         fft_of_Bx[n1] = -aky/ak2*fft_of_Jz[n1] + akz/ak2*fft_of_Jy[n1];
         fft_of_By[n1] = (-akz*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) + dens*fft_of_Ez[n1])/ak2;
         fft_of_Bz[n1] =  (aky*(fft_of_Jx[n1] + fft_of_JxBeam[n1]) - dens*fft_of_Ey[n1])/ak2;
 
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

   //------------------------ transform to configuration space E, B ----------------------------

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

   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;
   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 1.e-5;
   double difEx, difEy, difEz, difBx, difBy, difBz;
   difEx = difEy = difEz = difBx = difBy = difBz = 0.;

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

         long lccc = GetN(i,j,k);

         Cell &ccc = p_CellArray[lccc];

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
}


//--- Mesh:: ----------------------.
double Mesh::IterateFieldsHydroLinLayer(int iLayer)
{
   int i, j, k;
   double max_dJy = 0.;
   double max_Jy = 0.;
   double max_dEy = 0.;
   double max_Ey = 0.;
   double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;

   i = l_Mx-1;;
   j = l_My/2.;
   k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;
   double zco = Z(k) - domain()->GetZlength()/2.;

   double dens = 0.;
   int nsorts = domain()->GetNsorts();
   for (int isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += fabs(spec->Density(xco,yco,zco)*spec->GetQ2M());
   };

   if (dens <= 0.) dens = 1.;

   double ts = Ts();
   double h  = Hx();
   double hx = Hx();
   double hy = Hy();
   double hz = Hz();

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
         long lccc = GetN(i,j,k);
         Cell &c = p_CellArray[lccc];
         Cell &cp = p_CellArray[lccc+1];

         if (fabs(c.f_Jx - rJx[n]) > difJx) difJx = fabs(c.f_Jx - rJx[n]);
         if (fabs(c.f_Jy - rJy[n]) > difJy) difJy = fabs(c.f_Jy - rJy[n]);
         if (fabs(c.f_Jz - rJz[n]) > difJz) difJz = fabs(c.f_Jz - rJz[n]);

         if (fabs(rJx[n]) > maxJx) maxJx = fabs(rJx[n]);
         if (fabs(rJy[n]) > maxJy) maxJy = fabs(rJy[n]);
         if (fabs(rJz[n]) > maxJz) maxJz = fabs(rJz[n]);

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

   if (i == 0) {
      double check = 0.;
   };

//   fftw_execute(planR2R_RhoBeam);
//   fftw_execute(planR2R_JxBeam);

   fftw_execute(planR2R_Jx);
   fftw_execute(planR2R_Jy);
   fftw_execute(planR2R_Jz);

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

         if (fabs(fft_of_ExP[n1] + fft_of_Ex[n1]) > 1e-5) {
            fftDensExpected[n1] = -2.*diff_jx/(fft_of_ExP[n1] + fft_of_Ex[n1]);
         } else {
            fftDensExpected[n1] = dens;
         };

         //         VComplex rp  = (1.-maxRho)*vcRhoP[n1] + maxRho*diff_rp*damp;
         double rp  = fft_of_RhoP[n1]; // 
//         rp = dens;
         double jx = fft_of_Jx[n1]; //
         double jy = fft_of_Jy[n1]; //
         double jz = fft_of_Jz[n1]; //  

         double rho  = rp + fft_of_Jx[n1] - fft_of_JxP[n1] 
            - hx*(aky*(jy+fft_of_JyP[n1]) + akz*(jz+fft_of_JzP[n1]))/2.;

         fft_of_Rho[n1] = rho;

         double rb  = fft_of_RhoBeam[n1];
         double eyp = fft_of_EyP[n1];
         double ezp = fft_of_EzP[n1];

         double newEx = -(aky*jy + akz*jz)/ak2;
         double newEy = ((-ak2)*eyp + aky*(rp+2.*rb+rho) + 2*diff_jy)/(ak2);
         double newEz = ((-ak2)*ezp + akz*(rp+2.*rb+rho) + 2*diff_jz)/(ak2);

         if (newEx != 0. && k==l_Mz/2 && j==l_My/2) {
            double dummy = 0.;
         }

//         double newEy = 2.*(aky*(dens+(rp+rho)/2.+rb) + diff_jy)/ak2 - fft_of_Ey[n1];
         fft_of_Ey[n1] = newEy;
//         fft_of_Ey[n1] = 0.5*(newEy + fft_of_Ey[n1]);
//         double newEz = 2.*(akz*(dens+(rp+rho)/2.+rb) + diff_jz)/ak2 - fft_of_Ez[n1];
         fft_of_Ez[n1] = newEz;
//         fft_of_Ez[n1] = 0.5*(newEz + fft_of_Ez[n1]);

         errorEx += (newEx - fft_of_Ex[n1])*(newEx - fft_of_Ex[n1]);
         fft_of_Ex[n1] = newEx;
//         fft_of_Ex[n1] = 0.5*(newEx + fft_of_Ex[n1]);

         fft_of_Bx[n1] = fft_of_By[n1] = fft_of_Bz[n1] = 0.;

         if (fabs(fft_of_Ex[n1]) > maxfEx) maxfEx = fabs(fft_of_Ex[n1]);
         if (fabs(fft_of_Ey[n1]) > maxfEy) maxfEy = fabs(fft_of_Ey[n1]);
         if (fabs(fft_of_Ez[n1]) > maxfEz) maxfEz = fabs(fft_of_Ez[n1]);

         if (maxfEx > 1e-5 || maxfEy > 1e-5 || maxfEz > 1e-5 || maxfBx > 1e-5 || maxfBy > 1e-5 || maxfBz > 1e-5) {
            double check = 0.;
         };

      }
   }

   //------------------------ transform to configuration space E, B ----------------------------

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

   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;
   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 1.e-5;
   double difEx, difEy, difEz, difBx, difBy, difBz;
   difEx = difEy = difEz = difBx = difBy = difBz = 0.;

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

         long lccc = GetN(i,j,k);

         Cell &ccc = p_CellArray[lccc];

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

