#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <complex>
#include "vcomplex.h"

using namespace std;

#include <fftw3.h>

#include "vlpl3d.h"

static int FirstCall = 0;

static double *rEx, *rEy, *rEz, *rBx, *rBy, *rBz, *rJx, *rJy, *rJz, *rRho;
static double *rJxBeam, *rJyBeam, *rJzBeam, *rRhoBeam;
static double *rJxBeamP, *rJyBeamP, *rJzBeamP, *rRhoBeamP;
static complex <double> *vcEx, *vcEy, *vcEz, *vcBx, *vcBy, *vcBz;
static complex <double> *vcJx, *vcJy, *vcJz, *vcRho;
static complex <double> *vcJxP, *vcJyP, *vcJzP, *vcRhoP;
static complex <double> *vcJxBeam, *vcJyBeam, *vcJzBeam, *vcRhoBeam;
static complex <double> *vcJxBeamP, *vcJyBeamP, *vcJzBeamP, *vcRhoBeamP;
static complex <double> *carray;
static complex <double> *vcExRho, *vcEyRho, *vcEzRho;
static fftw_plan planR2C_Ex, planR2C_Ey, planR2C_Ez;
static fftw_plan planR2C_Bx, planR2C_By, planR2C_Bz;
static fftw_plan planC2R_Ex, planC2R_Ey, planC2R_Ez;
static fftw_plan planC2R_Bx, planC2R_By, planC2R_Bz;
static fftw_plan planR2C_Jx, planR2C_Jy, planR2C_Jz, planR2C_Rho;
static fftw_plan planR2C_JxBeam, planR2C_JyBeam, planR2C_JzBeam, planR2C_RhoBeam;
static fftw_plan planR2C_JxBeamP, planR2C_JyBeamP, planR2C_JzBeamP, planR2C_RhoBeamP;
static fftw_plan planC2R_Jx, planC2R_Jy, planC2R_Jz, planC2R_Rho;
static fftw_plan planR2C_ExRho, planR2C_EyRho, planR2C_EzRho;


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

   complex <double> I = complex <double>(0.,1.);
//   I.re = 0.;
//   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int nyc = ny/2 + 1;
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

      vcEx = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEy = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEz = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBx = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBy = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBz = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJx = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJy = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJz = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRho = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxBeam = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyBeam = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzBeam = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoBeam = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxBeamP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyBeamP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzBeamP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoBeamP = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      carray = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcExRho = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEyRho = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEzRho = (complex <double>*)fftw_malloc(ncomplex*sizeof(fftw_complex));

      planC2R_Ex = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEx, FFTW_ESTIMATE);
      planC2R_Ey = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEy, FFTW_ESTIMATE);
      planC2R_Ez = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEz, FFTW_ESTIMATE);

      planC2R_Bx = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBx, FFTW_ESTIMATE);
      planC2R_By = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBy, FFTW_ESTIMATE);
      planC2R_Bz = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBz, FFTW_ESTIMATE);

      planR2C_Ex = fftw_plan_dft_r2c_2d(nz, ny, rEx, (fftw_complex*)vcEx, FFTW_ESTIMATE);
      planR2C_Ey = fftw_plan_dft_r2c_2d(nz, ny, rEy, (fftw_complex*)vcEy, FFTW_ESTIMATE);
      planR2C_Ez = fftw_plan_dft_r2c_2d(nz, ny, rEz, (fftw_complex*)vcEz, FFTW_ESTIMATE);

      planR2C_ExRho = fftw_plan_dft_r2c_2d(nz, ny, rEx, (fftw_complex*)vcExRho, FFTW_ESTIMATE);
      planR2C_EyRho = fftw_plan_dft_r2c_2d(nz, ny, rEy, (fftw_complex*)vcEyRho, FFTW_ESTIMATE);
      planR2C_EzRho = fftw_plan_dft_r2c_2d(nz, ny, rEz, (fftw_complex*)vcEzRho, FFTW_ESTIMATE);

      planR2C_Bx = fftw_plan_dft_r2c_2d(nz, ny, rBx, (fftw_complex*)vcBx, FFTW_ESTIMATE);
      planR2C_By = fftw_plan_dft_r2c_2d(nz, ny, rBy, (fftw_complex*)vcBy, FFTW_ESTIMATE);
      planR2C_Bz = fftw_plan_dft_r2c_2d(nz, ny, rBz, (fftw_complex*)vcBz, FFTW_ESTIMATE);

      planR2C_Jx = fftw_plan_dft_r2c_2d(nz, ny, rJx, (fftw_complex*)vcJx, FFTW_ESTIMATE);
      planR2C_Jy = fftw_plan_dft_r2c_2d(nz, ny, rJy, (fftw_complex*)vcJy, FFTW_ESTIMATE);
      planR2C_Jz = fftw_plan_dft_r2c_2d(nz, ny, rJz, (fftw_complex*)vcJz, FFTW_ESTIMATE);
      planR2C_Rho  = fftw_plan_dft_r2c_2d(nz, ny, rRho, (fftw_complex*)vcRho, FFTW_ESTIMATE);

      planR2C_JxBeam = fftw_plan_dft_r2c_2d(nz, ny, rJxBeam, (fftw_complex*)vcJxBeam, FFTW_ESTIMATE);
      planR2C_JyBeam = fftw_plan_dft_r2c_2d(nz, ny, rJyBeam, (fftw_complex*)vcJyBeam, FFTW_ESTIMATE);
      planR2C_JzBeam = fftw_plan_dft_r2c_2d(nz, ny, rJzBeam, (fftw_complex*)vcJzBeam, FFTW_ESTIMATE);
      planR2C_RhoBeam  = fftw_plan_dft_r2c_2d(nz, ny, rRhoBeam, (fftw_complex*)vcRhoBeam, FFTW_ESTIMATE);

      planR2C_JxBeamP = fftw_plan_dft_r2c_2d(nz, ny, rJxBeamP, (fftw_complex*)vcJxBeamP, FFTW_ESTIMATE);
      planR2C_JyBeamP = fftw_plan_dft_r2c_2d(nz, ny, rJyBeamP, (fftw_complex*)vcJyBeamP, FFTW_ESTIMATE);
      planR2C_JzBeamP = fftw_plan_dft_r2c_2d(nz, ny, rJzBeamP, (fftw_complex*)vcJzBeamP, FFTW_ESTIMATE);
      planR2C_RhoBeamP  = fftw_plan_dft_r2c_2d(nz, ny, rRhoBeamP, (fftw_complex*)vcRhoBeamP, FFTW_ESTIMATE);

      planC2R_Jx = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rJx, FFTW_ESTIMATE);
      planC2R_Jy = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rJy, FFTW_ESTIMATE);
      planC2R_Jz = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rJz, FFTW_ESTIMATE);
      planC2R_Rho  = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rRho, FFTW_ESTIMATE);
   }

   for (int i=0; i<ncomplex; i++) {
      vcEx[i] = vcEy[i] = vcEz[i] 
      = vcBx[i] = vcBy[i] = vcBz[i] 
      = vcJx[i] = vcJy[i] = vcJz[i] = vcRho[i]
      = vcJxP[i] = vcJyP[i] = vcJzP[i] = vcRhoP[i]
      = vcJxBeam[i] = vcJyBeam[i] = vcJzBeam[i] = vcRhoBeam[i]
      = vcExRho[i] = vcEyRho[i] = vcEzRho[i] 
      = vcJxBeamP[i] = vcJyBeamP[i] = vcJzBeamP[i] = vcRhoBeamP[i] = complex <double>(0.,0.);
//      vcRho[i] = .01;
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

            rRho[n] = cp.f_Dens;

            rRhoBeam[n] = (c.f_RhoBeam + cp.f_RhoBeam)/2.;

            rJx[n] = cp.f_Jx;
            rJy[n] = cp.f_Jy;
            rJz[n] = cp.f_Jz;

            rJxBeam[n] = c.f_JxBeam;
            rJxBeamP[n] = cp.f_JxBeam;

            /*
            rRho[n] = c.f_RhoBeam;
            rJx[n] = c.f_JxBeam;
            rJy[n] = c.f_JyBeam;
            rJz[n] = c.f_JzBeam;
            rJxDx[n] = (cp.f_JxBeam - c.f_JxBeam)/hx;
            rJyDx[n] = (cp.f_JyBeam - c.f_JyBeam)/hx;
            rJzDx[n] = (cp.f_JzBeam - c.f_JzBeam)/hx;
            */

            if (fabs(rRho[n]) > maxRho) maxRho = rRho[n];
            if (fabs(rJx[n]) > maxJx) maxJx = rJx[n];
            if (fabs(rJy[n]) > maxJy) maxJy = rJy[n];
            if (fabs(rJz[n]) > maxJz) maxJz = rJz[n];
         }
      }

      fftw_execute(planR2C_RhoBeam);
      fftw_execute(planR2C_JxBeam);

      //------------------------ linearized E, B ----------------------------
      sumEx = sumEy = sumEz = 0.;
      sumBx = sumBy = sumBz = 0.;

      if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 0.) {
         double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
         sum += 0;
         i+=0;
      };

      int kZmax = nz/2;
      int kYmax = ny/2;
      double Ylength = domain()->GetYlength();
      double Zlength = domain()->GetZlength();

      double maxEx, maxEy, maxEz, maxBx, maxBy, maxBz;
      maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

      double total_dens = 0.;
      for (k=0; k<kZmax; k++)
      {
         for (j=0; j<kYmax; j++)
         {
            double akz = 2.*PI/Zlength*k;
            double aky = 2.*PI/Ylength*j;
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*k;

            if (ak2==0.) {
               vcEx[n1] = vcEy[n1] = vcEz[n1] = vcJx[n1] = vcJy[n1] = vcJz[n1] = vcRho[n1] = 0.;
               continue;
            };

            vcJxP[n1] = vcJx[n1];
            vcJyP[n1] = vcJy[n1];
            vcJzP[n1] = vcJz[n1];

            complex <double> rp  = vcRho[n1];
            complex <double> rb  = vcRhoBeam[n1];
            complex <double> jxp = vcJxP[n1];
            complex <double> jyp = vcJyP[n1];
            complex <double> jzp = vcJzP[n1];

            vcEy[n1] = -aky/(ak2+dens)*I*(-rp+rb);
            vcEz[n1] = -akz/(ak2+dens)*I*(-rp+rb);

            vcJy[n1] = jyp + h*dens*vcEy[n1];
            vcJz[n1] = jzp + h*dens*vcEz[n1];
         }
      }

      for (k=kZmax-1; k>0; k--)
      {
         for (j=0; j<kYmax; j++)
         {
            double akz = -2.*PI/Zlength*k;
            double aky =  2.*PI/Ylength*j;
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*(nz-k);

            if (ak2==0.) {
               vcEx[n1] = vcEy[n1] = vcEz[n1] = vcJx[n1] = vcJy[n1] = vcJz[n1] = vcRho[n1] = 0.;
               continue;
            };

            vcJxP[n1] = vcJx[n1];
            vcJyP[n1] = vcJy[n1];
            vcJzP[n1] = vcJz[n1];

            complex <double> rp  = vcRho[n1];
            complex <double> rb  = vcRhoBeam[n1];
            complex <double> jxp = vcJxP[n1];
            complex <double> jyp = vcJyP[n1];
            complex <double> jzp = vcJzP[n1];

            vcEy[n1] = -aky/(ak2+dens)*I*(-rp+rb);
            vcEz[n1] = -akz/(ak2+dens)*I*(-rp+rb);

            vcJy[n1] = jyp + h*dens*vcEy[n1];
            vcJz[n1] = jzp + h*dens*vcEz[n1];
         }
      }

      //------------------------ transform to configuration space E, Rho ----------------------------

      for (int n=0; n<ncomplex; n++) carray[n] = vcEx[n];
      fftw_execute(planC2R_Ex);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEy[n];
      fftw_execute(planC2R_Ey);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEz[n];
      fftw_execute(planC2R_Ez);
      for (int n=0; n<ncomplex; n++) carray[n] = vcRho[n];
      fftw_execute(planC2R_Rho);

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
            rEx[n] = pcc.f_Ex;

            rJy[n]  /= (4.*ny*nz);
            rJz[n]  /= (4.*ny*nz);
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

            rEy[n] += (uyppc*ppc.f_Jy - uypmc*pmc.f_Jy)/(2.*hy) + (uzpcp*pcp.f_Jy - uzpcm*pcm.f_Jy)/(2.*hz);
            rEz[n] += (uyppc*ppc.f_Jz - uypmc*pmc.f_Jz)/(2.*hy) + (uzpcp*pcp.f_Jz - uzpcm*pcm.f_Jz)/(2.*hz);
         }
      } 

      fftw_execute(planR2C_ExRho);
      fftw_execute(planR2C_EyRho);
      fftw_execute(planR2C_EzRho);

      //------------------------ add perturbations to E, j ----------------------------
      
      total_dens = 0.;
      for (k=0; k<kZmax; k++)
      {
         for (j=0; j<kYmax; j++)
         {
            double akz = 2.*PI/Zlength*k;
            double aky = 2.*PI/Ylength*j;
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*k;

            if (ak2==0.) {
               vcEx[n1] = vcEy[n1] = vcEz[n1] = vcJx[n1] = vcJy[n1] = vcJz[n1] = vcRho[n1] = 0.;
               continue;
            };

            complex <double> rp  = vcRho[n1];
            complex <double> rb  = vcRhoBeam[n1];
            complex <double> jxp = vcJxP[n1];
            complex <double> jyp = vcJyP[n1];
            complex <double> jzp = vcJzP[n1];
            double damping = 1./(1.+0.5*h*ak2);


            vcJy[n1] = vcJy[n1] + h*vcEyRho[n1]*damping;
            vcJz[n1] = vcJz[n1] + h*vcEzRho[n1]*damping;

            vcEy[n1] = vcEy[n1] - vcEyRho[n1]/(ak2 + dens)*damping;
            vcEz[n1] = vcEz[n1] - vcEzRho[n1]/(ak2 + dens)*damping;
      
            vcEx[n1] = -aky/ak2*I*vcJy[n1] -akz/ak2*I*vcJz[n1];
            
            vcBx[n1] = aky/ak2*I*vcJz[n1] -akz/(ak2 + dens)*I*vcJy[n1];
            vcBy[n1] =  akz/ak2*I*(vcJx[n1] + vcJxBeam[n1]) + dens*vcEz[n1]/ak2;
            vcBz[n1] = -aky/ak2*I*(vcJx[n1] + vcJxBeam[n1]) - dens*vcEy[n1]/ak2;

            vcJx[n1] = jxp + h*dens*vcEx[n1];
            vcRho[n1] = rp - h*dens*vcEx[n1] + aky*h*I*vcJy[n1] + akz*h*I*vcJz[n1];
         }
      }

      for (k=kZmax-1; k>0; k--)
      {
         for (j=0; j<kYmax; j++)
         {
            double akz = -2.*PI/Zlength*k;
            double aky =  2.*PI/Ylength*j;
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*(nz-k);

            if (ak2==0.) {
               vcEx[n1] = vcEy[n1] = vcEz[n1] = vcJx[n1] = vcJy[n1] = vcJz[n1] = vcRho[n1] = 0.;
               continue;
            };

            complex <double> rp  = vcRho[n1];
            complex <double> rb  = vcRhoBeam[n1];
            complex <double> jxp = vcJxP[n1];
            complex <double> jyp = vcJyP[n1];
            complex <double> jzp = vcJzP[n1];
            double damping = 1./(1.+0.5*h*ak2);


            vcJy[n1] = vcJy[n1] + h*vcEyRho[n1]*damping;
            vcJz[n1] = vcJz[n1] + h*vcEzRho[n1]*damping;

            vcEy[n1] = vcEy[n1] - vcEyRho[n1]/(ak2 + dens)*damping;
            vcEz[n1] = vcEz[n1] - vcEzRho[n1]/(ak2 + dens)*damping;

            vcEx[n1] = -aky/ak2*I*vcJy[n1] - akz/ak2*I*vcJz[n1];
            
            vcBx[n1] = aky/ak2*I*vcJz[n1] -akz/ak2*I*vcJy[n1];
            vcBy[n1] =  akz/ak2*I*(vcJx[n1] + vcJxBeam[n1]) + dens*vcEz[n1]/ak2;
            vcBz[n1] = -aky/ak2*I*(vcJx[n1] + vcJxBeam[n1]) - dens*vcEy[n1]/ak2;

            vcJx[n1] = jxp + h*dens*vcEx[n1];
            vcRho[n1] = rp - h*dens*vcEx[n1] + aky*h*I*vcJy[n1] + akz*h*I*vcJz[n1];
         }
      }

      for (int n=0; n<ncomplex; n++) carray[n] = vcBx[n];
      fftw_execute(planC2R_Bx);
      for (int n=0; n<ncomplex; n++) carray[n] = vcBy[n];
      fftw_execute(planC2R_By);
      for (int n=0; n<ncomplex; n++) carray[n] = vcBz[n];
      fftw_execute(planC2R_Bz);

      for (int n=0; n<ncomplex; n++) carray[n] = vcJx[n];
      fftw_execute(planC2R_Jx);
      for (int n=0; n<ncomplex; n++) carray[n] = vcJy[n];
      fftw_execute(planC2R_Jy);
      for (int n=0; n<ncomplex; n++) carray[n] = vcJz[n];
      fftw_execute(planC2R_Jz);

      for (int n=0; n<ncomplex; n++) carray[n] = vcEx[n];
      fftw_execute(planC2R_Ex);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEy[n];
      fftw_execute(planC2R_Ey);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEz[n];
      fftw_execute(planC2R_Ez);
      for (int n=0; n<ncomplex; n++) carray[n] = vcRho[n];
      fftw_execute(planC2R_Rho);

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

            ccc.f_Ex = rEx[n]/(ny*nz);
            ccc.f_Ey = rEy[n]/(ny*nz);
            ccc.f_Ez = rEz[n]/(ny*nz);

            ccc.f_Bx = rBx[n]/(ny*nz);
            ccc.f_By = rBy[n]/(ny*nz);
            ccc.f_Bz = rBz[n]/(ny*nz);


            ccc.f_Jx = rJx[n]/(ny*nz);
            ccc.f_Jy = rJy[n]/(ny*nz);
            ccc.f_Jz = rJz[n]/(ny*nz);
            ccc.f_Dens = rRho[n]/(ny*nz) + dens;

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

