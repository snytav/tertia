#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>

using namespace std;

#include <fftw3.h>

#include "vlpl3d.h"

static int FirstCall = 0;

static double *rEx, *rEy, *rEz, *rBx, *rBy, *rBz, *rJx, *rJy, *rJz, *rRho;
static double *rJxBeam, *rJyBeam, *rJzBeam, *rRhoBeam;
static double *rJxBeamP, *rJyBeamP, *rJzBeamP, *rRhoBeamP;
static double *fft_of_Ex, *fft_of_Ey, *fft_of_Ez, *fft_of_Bx, *fft_of_By, *fft_of_Bz;
static double *fft_of_Jx, *fft_of_Jy, *fft_of_Jz, *fft_of_JxP, *fft_of_JyP, *fft_of_JzP, *fft_of_Rho; 
static double *fft_of_JxBeam, *fft_of_JyBeam, *fft_of_JzBeam, *fft_of_RhoBeam;
static double *fft_of_JxBeamP, *fft_of_JyBeamP, *fft_of_JzBeamP, *fft_of_RhoBeamP;
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


//--- Mesh:: ----------------------.
void Mesh::MoveFieldsHydro(void)
{
   int i, j, k;
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

   double ts = Ts();
   double h  = Hx();
   double hy = Hy();
   double hz = Hz();

//   I.re = 0.;
//   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int nyc = ny;
   int ncomplex = nz*nyc;

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
      fft_of_JxP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JyP = (double*)fftw_malloc(ncomplex*sizeof(double));
      fft_of_JzP = (double*)fftw_malloc(ncomplex*sizeof(double)); 
      fft_of_Rho = (double*)fftw_malloc(ncomplex*sizeof(double));
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
      
      int gotthreads = fftw_init_threads();
      int nthreads = 2;
      if (gotthreads == 0) {
         cout << "Could not init threads! \n";
         nthreads = 1;
      };

      fftw_plan_with_nthreads(nthreads);  
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

   for (int i=0; i<ncomplex; i++) {
      fft_of_Ex[i] = fft_of_Ey[i] = fft_of_Ez[i] 
      = fft_of_Bx[i] = fft_of_By[i] = fft_of_Bz[i] 
      = fft_of_Jx[i] = fft_of_Jy[i] = fft_of_Jz[i] = fft_of_Rho[i]
      = fft_of_JxBeam[i] = fft_of_JyBeam[i] = fft_of_JzBeam[i] = fft_of_RhoBeam[i]
      = fft_of_ExRho[i] = fft_of_EyRho[i] = fft_of_EzRho[i] 
      = fft_of_JxBeamP[i] = fft_of_JyBeamP[i] = fft_of_JzBeamP[i] = fft_of_RhoBeamP[i] = 0.;
//      fft_of_Rho[i] = .01;
   }

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;


   double maxRho, maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxRho = maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;

   for (i=l_Mx-1; i>=0; i--) {
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

            rRhoBeam[n] = (c.f_RhoBeam + cp.f_RhoBeam)/2.;

            rJxBeam[n] = cp.f_JxBeam;

            rRho[n] = cp.f_Dens;
            rJx[n] = cp.f_Jx;
            rJy[n] = cp.f_Jy;
            rJz[n] = cp.f_Jz;

            if (fabs(rRho[n]) > maxRho) maxRho = rRho[n];
            if (fabs(rJx[n]) > maxJx) maxJx = rJx[n];
            if (fabs(rJy[n]) > maxJy) maxJy = rJy[n];
            if (fabs(rJz[n]) > maxJz) maxJz = rJz[n];
         }
      }

      fftw_execute(planR2R_RhoBeam);
      fftw_execute(planR2R_JxBeam);

      //------------------------ linearized E, B ----------------------------
      sumEx = sumEy = sumEz = 0.;
      sumBx = sumBy = sumBz = 0.;

      if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 0.) {
         double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
         sum += 0;
         i+=0;
      };

      double Ylength = domain()->GetYlength();
      double Zlength = domain()->GetZlength();

      double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;
      maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

      double total_dens = 0.;
      for (k=0; k<nz; k++)
      {
         for (j=0; j<ny; j++)
         {
            double akz = PI/Zlength*(k+0.5);
            double aky = PI/Ylength*(j+0.5);
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*k;

            if (ak2==0.) {
               fft_of_Ex[n1] = fft_of_Ey[n1] = fft_of_Ez[n1] = fft_of_Jx[n1] = fft_of_Jy[n1] = fft_of_Jz[n1] = fft_of_Rho[n1] = 0.;
               continue;
            };

            double rp  = fft_of_Rho[n1];
            double rb  = fft_of_RhoBeam[n1];
            double jxp = fft_of_Jx[n1];
            double jyp = fft_of_Jy[n1];
            double jzp = fft_of_Jz[n1];

            fft_of_Ey[n1] = aky/(ak2+dens)*(-rp+rb);
            fft_of_Ez[n1] = akz/(ak2+dens)*(-rp+rb);

            fft_of_Jy[n1] = jyp + h*dens*fft_of_Ey[n1];
            fft_of_Jz[n1] = jzp + h*dens*fft_of_Ez[n1];
         }
      }

      //------------------------ transform to configuration space E, Rho ----------------------------

//      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Jx[n];

      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ey[n];
      fftw_execute(planR2Rb_Ey);
      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ez[n];
      fftw_execute(planR2Rb_Ez);

      //------------------------ get products rho*E ----------------------------
      for (k=0; k<l_Mz; k++)
      {
         for (j=0; j<l_My; j++)
         {
            long n = j + ny*k;
            Cell &ccc = GetCell(i,j,k);
            Cell &pcc = GetCell(i+1,j,k);
            Cell &ppc = GetCell(i+1,j+1,k);
            Cell &pmc = GetCell(i+1,j-1,k);
            Cell &pcp = GetCell(i+1,j,k+1);
            Cell &pcm = GetCell(i+1,j,k-1);

            rRho[n] = pcc.f_Dens - dens; // density perturbation

            rJx[n] = pcc.f_Jx;
            rJy[n] = pcc.f_Jy;
            rJz[n] = pcc.f_Jz;
            rEx[n] = pcc.f_Ex;

            rEy[n]  /= (4.*ny*nz);
            rEz[n]  /= (4.*ny*nz);

            if (rRho[n] > 2*dens) rRho[n] = 2*dens;
            if (rRho[n] < -.75*dens) rRho[n] = -0.75*dens;

            double ux = -rJx[n]/(rRho[n]+dens);
            double uy = -rJy[n]/(rRho[n]+dens);
            double uz = -rJz[n]/(rRho[n]+dens);

            double u2 = ux*ux + uy*uy + uz*uz;
            if (u2 > 0.5) {
               u2 = 0.5;
            };

            double gamma_inv = sqrt(1. - u2);
//            gamma_inv = 1.;

            rRho[n] = rRho[n]*gamma_inv + (gamma_inv-1.)*dens;

            rEz[n] *= rRho[n];
            rEy[n] *= rRho[n];

            if (ppc.f_Dens < 0.5*dens) ppc.f_Dens = 0.5*dens;
            if (pmc.f_Dens < 0.5*dens) pmc.f_Dens = 0.5*dens;

            if (pcp.f_Dens < 0.5*dens) pcp.f_Dens = 0.5*dens;
            if (pcm.f_Dens < 0.5*dens) pcm.f_Dens = 0.5*dens;

            double uxppc = -ppc.f_Jx/ppc.f_Dens;
            double uyppc = -ppc.f_Jy/ppc.f_Dens;
            double uzppc = -ppc.f_Jz/ppc.f_Dens;

            double uxpmc = -pmc.f_Jx/pmc.f_Dens;
            double uypmc = -pmc.f_Jy/pmc.f_Dens;
            double uzpmc = -pmc.f_Jz/pmc.f_Dens;

            double uxpcp = -pcp.f_Jx/pcp.f_Dens;
            double uypcp = -pcp.f_Jy/pcp.f_Dens;
            double uzpcp = -pcp.f_Jz/pcp.f_Dens;

            double uxpcm = -pcm.f_Jx/pcm.f_Dens;
            double uypcm = -pcm.f_Jy/pcm.f_Dens;
            double uzpcm = -pcm.f_Jz/pcm.f_Dens;

            rEx[n] += (uyppc*ppc.f_Jx - uypmc*pmc.f_Jy)/(2.*hy) + (uzpcp*pcp.f_Jx - uzpcm*pcm.f_Jx)/(2.*hz);
            rEy[n] += (uyppc*ppc.f_Jy - uypmc*pmc.f_Jy)/(2.*hy) + (uzpcp*pcp.f_Jy - uzpcm*pcm.f_Jy)/(2.*hz);
            rEz[n] += (uyppc*ppc.f_Jz - uypmc*pmc.f_Jz)/(2.*hy) + (uzpcp*pcp.f_Jz - uzpcm*pcm.f_Jz)/(2.*hz);

         }
      }

      fftw_execute(planR2R_ExRho);
      fftw_execute(planR2R_EyRho);
      fftw_execute(planR2R_EzRho);

      //------------------------ add perturbations to E, j ----------------------------
      
      total_dens = 0.;
      for (k=0; k<nz; k++)
      {
         for (j=0; j<ny; j++)
         {
            double akz = PI/Zlength*(k+0.5);
            double aky = PI/Ylength*(j+0.5);
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*k;


            if (ak2==0.) {
               fft_of_Ex[n1] = fft_of_Ey[n1] = fft_of_Ez[n1] = fft_of_Jx[n1] = fft_of_Jy[n1] = fft_of_Jz[n1] = fft_of_Rho[n1] = 0.;
               continue;
            };

            double rp  = fft_of_Rho[n1];
            double rb  = fft_of_RhoBeam[n1];
            double jxp = fft_of_Jx[n1];
            double damping = 1./(1.+1e-0*h*ak2/sqrt(dens+1.));

            fft_of_Jy[n1] = fft_of_Jy[n1] + h*fft_of_EyRho[n1]*damping;
            fft_of_Jz[n1] = fft_of_Jz[n1] + h*fft_of_EzRho[n1]*damping;

            fft_of_Ey[n1] = fft_of_Ey[n1] - fft_of_EyRho[n1]/(ak2 + dens)*damping;
            fft_of_Ez[n1] = fft_of_Ez[n1] - fft_of_EzRho[n1]/(ak2 + dens)*damping;
 
            fft_of_Bx[n1] = akz/ak2*fft_of_Jy[n1] - aky/ak2*fft_of_Jz[n1];
            
            fft_of_By[n1] = dens/ak2*fft_of_Ez[n1] 
               - akz/ak2*(fft_of_Jx[n1] + fft_of_JxBeam[n1]);
            fft_of_Bz[n1] =  -dens/ak2*fft_of_Ey[n1] 
               + aky/ak2*(fft_of_Jx[n1] + fft_of_JxBeam[n1]);
      
            fft_of_Ex[n1] = -aky/ak2*fft_of_Jy[n1] -akz/ak2*fft_of_Jz[n1];
//            fft_of_Jx[n1] = jxp + h*fft_of_ExRho[n1];
            fft_of_Jx[n1] = jxp + h*dens*fft_of_Ex[n1];

            fft_of_Rho[n1] = rp - h*dens*fft_of_Ex[n1] + aky*h*fft_of_Jy[n1] + akz*h*fft_of_Jz[n1];

            fft_of_JxP[n1] = fft_of_Jx[n1];
            fft_of_JyP[n1] = fft_of_Jy[n1];
            fft_of_JzP[n1] = fft_of_Jz[n1];        

         }
      }

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

      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ex[n];
      fftw_execute(planR2Rb_Ex);
      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ey[n];
      fftw_execute(planR2Rb_Ey);
      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Ez[n];
      fftw_execute(planR2Rb_Ez);

      for (int n=0; n<ncomplex; n++) carray[n] = fft_of_Rho[n];
      fftw_execute(planR2Rb_Rho);

      sumEx = sumEy = sumEz = 0.;
      sumBx = sumBy = sumBz = 0.;
      maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

      for (k=0; k<nz; k++)
      {
         for (j=0; j<ny; j++)
         {
            long n = j + l_My*k;
            long lccc = GetN(i,j,k);

            Cell &ccc = p_CellArray[lccc];

            ccc.f_Ex = rEx[n]/(4.*ny*nz);
            ccc.f_Ey = rEy[n]/(4.*ny*nz);
            ccc.f_Ez = rEz[n]/(4.*ny*nz);

            ccc.f_Bx = rBx[n]/(4.*ny*nz);
            ccc.f_By = rBy[n]/(4.*ny*nz);
            ccc.f_Bz = rBz[n]/(4.*ny*nz);


            ccc.f_Jx = rJx[n]/(4.*ny*nz);
            ccc.f_Jy = rJy[n]/(4.*ny*nz);
            ccc.f_Jz = rJz[n]/(4.*ny*nz);
            ccc.f_Dens = rRho[n]/(4.*ny*nz) + dens;

            sumEx += rEx[n]*rEx[n];
            sumEy += rEy[n]*rEy[n];
            sumEz += rEz[n]*rEz[n];
            sumBx += rBx[n]*rBx[n];
            sumBy += rBy[n]*rBy[n];
            sumBz += rBz[n]*rBz[n];

            total_dens += ccc.f_Dens;

            if (fabs(ccc.f_Ex) > maxEx) {
               maxEx = fabs(ccc.f_Ex);
            }
            if (fabs(ccc.f_Ey) > maxEy) maxEy = fabs(ccc.f_Ey);
            if (fabs(ccc.f_Ez) > maxEz) maxEz = fabs(ccc.f_Ez);

            if (fabs(ccc.f_Bx) > maxBx) maxBx = fabs(ccc.f_Bx);
            if (fabs(ccc.f_By) > maxBy) maxBy = fabs(ccc.f_By);
            if (fabs(ccc.f_Bz) > maxBz) maxBz = fabs(ccc.f_Bz);
         }
      }

      if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 0.) {
         double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
         sum += 0;
      };

//      cout <<"Density at i="<<i<< " is " << total_dens << endl;
      };

   //   domain()->Exchange(SPACK_F);
   /*
   fftw_free(in);
   fftw_free(in1);
   fftw_free(in2);
   fftw_free(in3);
   fftw_free(fft_of_Out);
   fftw_free(fft_of_Out1);
   fftw_free(fft_of_Ac);
   fftw_free(fft_of_Am);
   fftw_free(fft_of_Fp);
   fftw_free(fft_of_Fc);
   fftw_free(fft_of_Fm);
   fftw_free(fft_of_Fnext);
   fftw_destroy_plan(planR2R);
   fftw_destroy_plan(planR2Rb);
   fftw_destroy_plan(planR2R1);
   fftw_destroy_plan(planR2Rb1);
   fftw_destroy_plan(planR2R2);
   fftw_destroy_plan(planR2Rb2);
   fftw_destroy_plan(planR2R3);
   fftw_destroy_plan(planR2Rb3);
   fftw_destroy_plan(planR2R11);
   fftw_destroy_plan(planR2Rb11);
   */
}

