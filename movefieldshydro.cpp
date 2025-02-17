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
//unsigned int fp_control_state = _controlfp(_EM_INEXACT, _MCW_EM);

static int FirstCall = 0;

static double *rEx, *rEy, *rEz, *rBx, *rBy, *rBz, *rJx, *rJy, *rJz, *rRho;
static double *rJxBeam, *rJyBeam, *rJzBeam, *rRhoBeam;
static double *rJxBeamP, *rJyBeamP, *rJzBeamP, *rRhoBeamP;
static VComplex *vcEx, *vcEy, *vcEz, *vcBx, *vcBy, *vcBz;
static VComplex *vcJx, *vcJy, *vcJz, *vcRho;
static VComplex *vcJxP, *vcJyP, *vcJzP, *vcRhoP;
static VComplex *vcJxBeam, *vcJyBeam, *vcJzBeam, *vcRhoBeam;
static VComplex *vcJxBeamP, *vcJyBeamP, *vcJzBeamP, *vcRhoBeamP;
static VComplex *carray;
static VComplex *vcExRho, *vcEyRho, *vcEzRho;
static fftw_plan planR2C_Ex, planR2C_Ey, planR2C_Ez;
static fftw_plan planR2C_Bx, planR2C_By, planR2C_Bz;
static fftw_plan planC2R_Ex, planC2R_Ey, planC2R_Ez;
static fftw_plan planC2R_Bx, planC2R_By, planC2R_Bz;
static fftw_plan planR2C_Jx, planR2C_Jy, planR2C_Jz, planR2C_Rho;
static fftw_plan planR2C_JxBeam, planR2C_JyBeam, planR2C_JzBeam, planR2C_RhoBeam;
static fftw_plan planR2C_JxBeamP, planR2C_JyBeamP, planR2C_JzBeamP, planR2C_RhoBeamP;
static fftw_plan planC2R_Jx, planC2R_Jy, planC2R_Jz, planC2R_Rho;
static fftw_plan planR2C_ExRho, planR2C_EyRho, planR2C_EzRho;

static double maxRho;

//--- Mesh:: ----------------------.
void Mesh::MoveFieldsHydro(void)
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
   int nsorts = domain()->GetNsorts();

   double ts = Ts();
   double h  = Hx();
   double hy = Hy();
   double hz = Hz();

   VComplex I = VComplex(0.,1.);
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

      vcEx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRho = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxBeam = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyBeam = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzBeam = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoBeam = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxBeamP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyBeamP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzBeamP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRhoBeamP = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      carray = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcExRho = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEyRho = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEzRho = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));

      int gotthreads = fftw_init_threads();
      int nthreads = 2;
      if (gotthreads == 0) {
         cout << "Could not init threads! \n";
         nthreads = 1;
      };

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
      = vcJxBeamP[i] = vcJyBeamP[i] = vcJzBeamP[i] = vcRhoBeamP[i] = VComplex(0.,0.);
      //      vcRho[i] = .01;
   }

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;


   double maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;

   for (int i=l_Mx-1; i>=0; i--) {
      j = l_My/2;
      k = l_Mz/2;
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
      if (i == 154) {
         double check;
         check = 0;
      }
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

            rRhoBeam[n] = c.f_RhoBeam;

            rJx[n] = cp.f_Jx;
            rJy[n] = cp.f_Jy;
            rJz[n] = cp.f_Jz;

            rJxBeam[n] = c.f_JxBeam;

            if (fabs(c.f_JxBeam) > 10) {
               cout << "Too large beam current c.f_JxBeam="<<c.f_JxBeam<<endl;
               exit(20);
            }
         }
      }

      fftw_execute(planR2C_RhoBeam);
      fftw_execute(planR2C_JxBeam);

      //------------------------ linearized E, B ----------------------------
      sumEx = sumEy = sumEz = 0.;
      sumBx = sumBy = sumBz = 0.;

      int kZmax = nz/2;
      int kYmax = ny/2;
      double Ylength = domain()->GetYlength();
      double Zlength = domain()->GetZlength();

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

            vcRhoP[n1] = vcRho[n1];
            vcJxP[n1] = vcJx[n1];
            vcJyP[n1] = vcJy[n1];
            vcJzP[n1] = vcJz[n1];

            VComplex rp  = vcRhoP[n1];
            VComplex rb  = vcRhoBeam[n1];
            VComplex jxp = vcJxP[n1];
            VComplex jyp = vcJyP[n1];
            VComplex jzp = vcJzP[n1];

            vcEy[n1] = (-aky/(ak2+dens))*I*(-rp+rb);
            vcEz[n1] = -akz/(ak2+dens)*I*(-rp+rb);

            vcJy[n1] = jyp + h*dens*vcEy[n1];
            vcJz[n1] = jzp + h*dens*vcEz[n1];

            vcEx[n1] = -aky/ak2*I*vcJy[n1] -akz/ak2*I*vcJz[n1];

            vcBx[n1] = aky/ak2*I*vcJz[n1] -akz/(ak2 + dens)*I*vcJy[n1];
            vcBy[n1] =  akz/ak2*I*(vcJx[n1] + vcJxBeam[n1]) + dens*vcEz[n1]/ak2;
            vcBz[n1] = -aky/ak2*I*(vcJx[n1] + vcJxBeam[n1]) - dens*vcEy[n1]/ak2;

            vcJx[n1] = jxp + h*dens*vcEx[n1];
            vcRho[n1] = rp - h*dens*vcEx[n1] + aky*h*I*vcJy[n1] + akz*h*I*vcJz[n1];

            if (abs(vcEx[n1]) > 100 || abs(vcEy[n1]) > 100 || abs(vcEz[n1]) > 100) {
               cout << "Large fields! \n";
            }
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

            vcRhoP[n1] = vcRho[n1];
            vcJxP[n1] = vcJx[n1];
            vcJyP[n1] = vcJy[n1];
            vcJzP[n1] = vcJz[n1];

            VComplex rp  = vcRhoP[n1];
            VComplex rb  = vcRhoBeam[n1];
            VComplex jxp = vcJxP[n1];
            VComplex jyp = vcJyP[n1];
            VComplex jzp = vcJzP[n1];

            vcEy[n1] = -aky/(ak2+dens)*I*(-rp+rb);
            vcEz[n1] = -akz/(ak2+dens)*I*(-rp+rb);

            vcJy[n1] = jyp + h*dens*vcEy[n1];
            vcJz[n1] = jzp + h*dens*vcEz[n1];

            vcEx[n1] = -aky/ak2*I*vcJy[n1] -akz/ak2*I*vcJz[n1];

            vcBx[n1] =  aky/ak2*I*vcJz[n1] -akz/(ak2 + dens)*I*vcJy[n1];
            vcBy[n1] =  akz/ak2*I*(vcJx[n1] + vcJxBeam[n1]) + dens*vcEz[n1]/ak2;
            vcBz[n1] = -aky/ak2*I*(vcJx[n1] + vcJxBeam[n1]) - dens*vcEy[n1]/ak2;

            vcJx[n1] = jxp + h*dens*vcEx[n1];
            vcRho[n1] = rp - h*dens*vcEx[n1] + aky*h*I*vcJy[n1] + akz*h*I*vcJz[n1];
            if (abs(vcEx[n1]) > 100 || abs(vcEy[n1]) > 100 || abs(vcEz[n1]) > 100 || abs(vcRho[n1]) > 100) {
               cout << "Large fields! \n";
            }
         }
      }

      //------------------------ transform to configuration space E, Rho ----------------------------

      for (int n=0; n<ncomplex; n++) carray[n] = vcRho[n];
      fftw_execute(planC2R_Rho);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEx[n];
      fftw_execute(planC2R_Ex);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEy[n];
      fftw_execute(planC2R_Ey);
      for (int n=0; n<ncomplex; n++) carray[n] = vcEz[n];
      fftw_execute(planC2R_Ez);

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

      maxEx = maxEy = maxEz = maxRho = 0.;
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

            rRho[n] /= (ny*nz);
            rJx[n]  /= (ny*nz);
            rJy[n]  /= (ny*nz);
            rJz[n]  /= (ny*nz);
            rEx[n]  /= (ny*nz);
            rEy[n]  /= (ny*nz);
            rEz[n]  /= (ny*nz);
            rBx[n]  /= (ny*nz);
            rBy[n]  /= (ny*nz);
            rBz[n]  /= (ny*nz);

            double rhop = pcc.f_Dens - dens; // density perturbation
            if (rhop > 2*dens) rhop = 2*dens;
            if (rhop < -.75*dens) rhop = -0.75*dens;

            double rho = rRho[n]; // density perturbation
            if (rho > 2*dens) rho = 2*dens;
            if (rho < -.75*dens) rho = -0.75*dens;

            double ux = -rJx[n]/(rho+dens);
            double uy = -rJy[n]/(rho+dens);
            double uz = -rJz[n]/(rho+dens);

            if (ux >  0.9) ux =  0.9;
            if (ux < -0.9) ux = -0.9;
            if (uy >  0.9) uy =  0.9;
            if (uy < -0.9) uy = -0.9;
            if (uz >  0.9) uz =  0.9;
            if (uz < -0.9) uz = -0.9;

            double u2 = ux*ux + uy*uy + uz*uz;
            double arg = 1.-u2;
            if (arg < 0.1) arg = 0.1;
            double gamma_inv = sqrt(arg);
            //            gamma_inv = 1.;

            rho = rho*gamma_inv + (gamma_inv-1.)*dens;

            rEx[n] *= rho;
            rEy[n] *= rho;
            rEz[n] *= rho;

            if (pcc.f_Dens < 0.5*dens) pcc.f_Dens = 0.5*dens;
            if (ppc.f_Dens < 0.5*dens) ppc.f_Dens = 0.5*dens;
            if (pmc.f_Dens < 0.5*dens) pmc.f_Dens = 0.5*dens;

            if (pcp.f_Dens < 0.5*dens) pcp.f_Dens = 0.5*dens;
            if (pcm.f_Dens < 0.5*dens) pcm.f_Dens = 0.5*dens;

            double uyppc = -ppc.f_Jy/ppc.f_Dens;
            double uypmc = -pmc.f_Jy/pmc.f_Dens;

            double uzpcp = -pcp.f_Jz/pcp.f_Dens;
            double uzpcm = -pcm.f_Jz/pcm.f_Dens;

            double uxpcc = -pcc.f_Jx/pcc.f_Dens;

            // Convection ->>>>>>>
            rEx[n] -= (uxpcc*pcc.f_Jx - ux*rJx[n])/(h) + (uyppc*ppc.f_Jx - uypmc*pmc.f_Jx)/(2.*hy) 
               + (uzpcp*pcp.f_Jx - uzpcm*pcm.f_Jx)/(2.*hz);
            rEy[n] -= (uxpcc*pcc.f_Jy - ux*rJy[n])/(h) + (uyppc*ppc.f_Jy - uypmc*pmc.f_Jy)/(2.*hy) 
               + (uzpcp*pcp.f_Jy - uzpcm*pcm.f_Jy)/(2.*hz);
            rEz[n] -= (uxpcc*pcc.f_Jz - ux*rJz[n])/(h) + (uyppc*ppc.f_Jz - uypmc*pmc.f_Jz)/(2.*hy) 
               + (uzpcp*pcp.f_Jz - uzpcm*pcm.f_Jz)/(2.*hz);

            //            rEx[n] -= (uyppc*ppc.f_Jx - uypmc*pmc.f_Jx)/(2.*hy) 
            //              + (uzpcp*pcp.f_Jx - uzpcm*pcm.f_Jx)/(2.*hz);
            //            rEy[n] -= (uyppc*ppc.f_Jy - uypmc*pmc.f_Jy)/(2.*hy) 
            //               + (uzpcp*pcp.f_Jy - uzpcm*pcm.f_Jy)/(2.*hz);
            //            rEz[n] -= (uyppc*ppc.f_Jz - uypmc*pmc.f_Jz)/(2.*hy) 
            //               + (uzpcp*pcp.f_Jz - uzpcm*pcm.f_Jz)/(2.*hz);


            // Magnetic force ->>>>>>>

            rEx[n] += dens*(rJy[n]*rBz[n] - rJz[n]*rBy[n]);
            rEy[n] += dens*(rJz[n]*rBx[n] - rJx[n]*rBz[n]);
            rEz[n] += dens*(rJx[n]*rBy[n] - rJy[n]*rBx[n]);

            if (fabs(rRho[n]) > maxRho) maxRho = rRho[n];
            if (fabs(rEx[n]) > maxEx) maxEx = rEx[n];
            if (fabs(rEy[n]) > maxEy) maxEy = rEy[n];
            if (fabs(rEz[n]) > maxEz) maxEz = rEz[n]; 
         }
      } 

      if (maxEx > 10 || maxEy > 10 || maxEz > 10 || maxRho > 10) {
         cout << "Large real fields! \n";
      }

      fftw_execute(planR2C_ExRho);
      fftw_execute(planR2C_EyRho);
      fftw_execute(planR2C_EzRho);


      //------------------------ add perturbations to E, j ----------------------------

      total_dens = 0.;
      double eta = maxRho - dens;
      if (eta < 1) eta = 1;
      eta = 1.;

      VComplex dEx, dEy, dEz, dJx, dJy, dJz;


      for (k=0; k<kZmax; k++)
      {
         for (j=0; j<kYmax; j++)
         {
            double akz = 2.*PI/Zlength*k;
            double aky = 2.*PI/Ylength*j;
            double ak2 = aky*aky + akz*akz;

            long n1 = j + nyc*k;

            if (ak2==0.) {
               //               vcEx[n1] = vcEy[n1] = vcEz[n1] = vcJx[n1] = vcJy[n1] = vcJz[n1] = vcRho[n1] = 0.;
               continue;
            };

            VComplex rp  = vcRhoP[n1];
            VComplex rb  = vcRhoBeam[n1];
            VComplex jxp = vcJxP[n1];
            VComplex jyp = vcJyP[n1];
            VComplex jzp = vcJzP[n1];
            double damping = 1./(1.+eta*h*ak2);

            dJx = h*vcExRho[n1]*damping;
            dJy = h*vcEyRho[n1]*damping;
            dJz = h*vcEzRho[n1]*damping;

            if (abs(dJy) > max_dJy) {
               max_dJy = abs(dJy);
            };
            if (abs(vcJy[n1]) > max_Jy) {
               max_Jy = abs(vcJy[n1]);
            };

            vcJx[n1] = vcJx[n1] + dJx;
            vcJy[n1] = vcJy[n1] + dJy;
            vcJz[n1] = vcJz[n1] + dJz;

            dEx = - aky/ak2*I*dJy - akz/ak2*I*dJz;
            dEy = -dJy/(h*(ak2 + dens));
            dEz = -dJz/(h*(ak2 + dens));

            if (abs(dEy) > max_dEy) {
               max_dEy = abs(dEy);
            };
            if (abs(vcEy[n1]) > max_Ey) {
               max_Ey = abs(vcEy[n1]);
            };

            vcEx[n1] = vcEx[n1] + dEx;
            vcEy[n1] = vcEy[n1] + dEy;
            vcEz[n1] = vcEz[n1] + dEz;

            vcRho[n1] = vcRho[n1] - h*dens*dEx + aky*h*I*dJy + akz*h*I*dJz;
            if (abs(vcEx[n1]) > 100 || abs(vcEy[n1]) > 100 || abs(vcEz[n1]) > 100 || abs(vcRho[n1]) > 100) {
               cout << "Large fields! \n";
            }
         }
      }

      for (k=kZmax-1; k>0; k--)
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

            VComplex rp  = vcRhoP[n1];
            VComplex rb  = vcRhoBeam[n1];
            VComplex jxp = vcJxP[n1];
            VComplex jyp = vcJyP[n1];
            VComplex jzp = vcJzP[n1];
            double damping = 1./(1.+eta*h*ak2);

            dJx = h*vcExRho[n1]*damping;
            dJy = h*vcEyRho[n1]*damping;
            dJz = h*vcEzRho[n1]*damping;

            if (abs(dJy) > max_dJy) {
               max_dJy = abs(dJy);
            };
            if (abs(vcJy[n1]) > max_Jy) {
               max_Jy = abs(vcJy[n1]);
            };

            vcJx[n1] = vcJx[n1] + dJx;
            vcJy[n1] = vcJy[n1] + dJy;
            vcJz[n1] = vcJz[n1] + dJz;

            dEx = -aky/ak2*I*dJy - akz/ak2*I*dJz;
            dEy = -dJy/(h*(ak2 + dens));
            dEz = -dJz/(h*(ak2 + dens));

            if (abs(dEy) > max_dEy) {
               max_dEy = abs(dEy);
            };
            if (abs(vcEy[n1]) > max_Ey) {
               max_Ey = abs(vcEy[n1]);
            };

            vcEx[n1] = vcEx[n1] + dEx;
            vcEy[n1] = vcEy[n1] + dEy;
            vcEz[n1] = vcEz[n1] + dEz;

            vcRho[n1] = vcRho[n1] - h*dens*dEx + aky*h*I*dJy + akz*h*I*dJz;
            if (abs(vcEx[n1]) > 100 || abs(vcEy[n1]) > 100 || abs(vcEz[n1]) > 100 || abs(vcRho[n1]) > 100) {
               cout << "Large fields! \n";
            }
         }
      }

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

            rRho[n] /= (ny*nz);
            if (rRho[n] < -0.5*dens) rRho[n] = -0.5*dens;
            if (rRho[n] > 0.75*dens) rRho[n] = 0.75*dens;

            ccc.f_Dens = rRho[n] + dens;
            if (ccc.f_Dens < 0.5*dens) ccc.f_Dens = 0.5*dens;

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

            if (fabs(rRho[n]) > maxRho) maxRho = rRho[n];

         }
      }

      fftw_execute(planR2C_Rho);

      if (maxEx > 100 || maxEy > 100 || maxEz > 100) {
         cout << "Large real fields! \n";
      }

      if (sumEx + sumEy + sumEz + sumBx + sumBy + sumBz > 1e4) {
         double sum = sumEx + sumEy + sumEz + sumBx + sumBy + sumBz;
         sum += 0;
      };


      if (i == 300) {
         double check;
         check = 0.;
      }

      //      cout <<"Density at i="<<i<< " is " << total_dens << endl;
   };
   /*
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

