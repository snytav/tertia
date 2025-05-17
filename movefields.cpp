#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "vcomplex.h"

using namespace std;

#include <fftw3.h>

#include "vlpl3d.h"

static int FirstCall = 0;

VComplex xx;

static double *rEx, *rEy, *rEz, *rBx, *rBy, *rBz, *rJx, *rJy, *rJz, *rRho;
static double *rEyp, *rEzp, *rByp, *rBzp;
static double *rJxDx, *rJyDx, *rJzDx;
static VComplex *vcEx, *vcEy, *vcEz, *vcBx, *vcBy, *vcBz, *vcJx, *vcJy, *vcJz, *vcRho;
static VComplex *vcExp, *vcEyp, *vcEzp, *vcBxp, *vcByp, *vcBzp;
static VComplex *vcJxDx, *vcJyDx, *vcJzDx;
static VComplex *carray;
static fftw_plan planR2C_Ex, planR2C_Ey, planR2C_Ez;
static fftw_plan planR2C_Bx, planR2C_By, planR2C_Bz;
static fftw_plan planC2R_Ex, planC2R_Ey, planC2R_Ez;
static fftw_plan planC2R_Bx, planC2R_By, planC2R_Bz;
static fftw_plan planR2C_Jx, planR2C_Jy, planR2C_Jz, planR2C_Rho;
static fftw_plan planR2C_JxDx, planR2C_JyDx, planR2C_JzDx;


//--- Mesh:: ----------------------.
void Mesh::MoveFields(void)
{
   return;
}

//--- Mesh:: ----------------------->
void Mesh::MoveFieldsLayer(int iIn, int iInitStep, double part)
{
   int i, j, k;
   i = iIn;
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

   if (i<0 || i>=l_Mx) {
      cout << "Wrong i=" << i << " in Mesh::MoveEfield(long i) " << endl;
      exit(i*10);
   };

   if (i == l_Mx/2) {
      double half = 0.;
   };
   if (i == l_Mx/3) {
      double third = 0.;
   };
   double ts = Ts();
   double hx = Hx();
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
      rJxDx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJyDx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rJzDx = (double*)fftw_malloc(nz*(ny+2)*sizeof(double));
      rRho= (double*)fftw_malloc(nz*(ny+2)*sizeof(double));

      vcEx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJy = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJz = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJxDx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJyDx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcJzDx = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcRho = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcExp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEyp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcEzp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBxp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcByp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      vcBzp = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
      carray = (VComplex*)fftw_malloc(ncomplex*sizeof(fftw_complex));
 
      planC2R_Ex = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEx, FFTW_ESTIMATE);
      planC2R_Ey = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEy, FFTW_ESTIMATE);
      planC2R_Ez = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rEz, FFTW_ESTIMATE);

      planC2R_Bx = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBx, FFTW_ESTIMATE);
      planC2R_By = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBy, FFTW_ESTIMATE);
      planC2R_Bz = fftw_plan_dft_c2r_2d(nz, ny, (fftw_complex*)carray, rBz, FFTW_ESTIMATE);

      planR2C_Jx = fftw_plan_dft_r2c_2d(nz, ny, rJx, (fftw_complex*)vcJx, FFTW_ESTIMATE);
      planR2C_Jy = fftw_plan_dft_r2c_2d(nz, ny, rJy, (fftw_complex*)vcJy, FFTW_ESTIMATE);
      planR2C_Jz = fftw_plan_dft_r2c_2d(nz, ny, rJz, (fftw_complex*)vcJz, FFTW_ESTIMATE);
      planR2C_JxDx = fftw_plan_dft_r2c_2d(nz, ny, rJxDx, (fftw_complex*)vcJxDx, FFTW_ESTIMATE);
      planR2C_JyDx = fftw_plan_dft_r2c_2d(nz, ny, rJyDx, (fftw_complex*)vcJyDx, FFTW_ESTIMATE);
      planR2C_JzDx = fftw_plan_dft_r2c_2d(nz, ny, rJzDx, (fftw_complex*)vcJzDx, FFTW_ESTIMATE);
      planR2C_Rho  = fftw_plan_dft_r2c_2d(nz, ny, rRho, (fftw_complex*)vcRho, FFTW_ESTIMATE);
   }

   if (iInitStep) {
      for (int n=0; n<ncomplex; n++) {
         vcEx[n] = vcEy[n] = vcEz[n] 
         = vcBx[n] = vcBy[n] = vcBz[n] 
         = vcExp[n] = vcBxp[n] 
         = vcEyp[n] = vcEzp[n] = vcByp[n] = vcBzp[n] 
         = vcJx[n] = vcJy[n] = vcJz[n] = vcRho[n]
         = vcJxDx[n] = vcJyDx[n] = vcJzDx[n] = carray[n] = VComplex(0.,0.);
      }
   };

   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;


   double maxRho, maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxRho = maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;
   i = iIn;

   for (k=0; k<nz; k++)
   {
      if(k==l_Mz/3) {
         double checkk = 0.;
      }
      for (j=0; j<ny; j++)
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

         rEy[n] = cp.f_Ey;
         rEz[n] = cp.f_Ez;

         rBy[n] = cp.f_By;
         rBz[n] = cp.f_Bz;

         rRho[n] = (c.f_Dens + c.f_RhoBeam + cp.f_Dens + cp.f_RhoBeam)/2.;
         rJx[n] = (cp.f_Jx + cp.f_JxBeam + c.f_Jx + c.f_JxBeam)/2.;
         rJy[n] = (cp.f_Jy + cp.f_JyBeam + c.f_Jy + c.f_JyBeam)/2.;
         rJz[n] = (cp.f_Jz + cp.f_JzBeam + c.f_Jz + c.f_JzBeam)/2.;
         rJxDx[n] = (cp.f_Jx + cp.f_JxBeam - c.f_Jx - c.f_JxBeam)/hx;
         rJyDx[n] = (cp.f_Jy + cp.f_JyBeam - c.f_Jy - c.f_JyBeam)/hx;
         rJzDx[n] = (cp.f_Jz + cp.f_JzBeam - c.f_Jz - c.f_JzBeam)/hx;

         rRho[n] = c.f_RhoBeam;
         rJx[n] = c.f_JxBeam;
         rJy[n] = c.f_JyBeam;
         rJz[n] = c.f_JzBeam;
         rJxDx[n] = (cp.f_JxBeam - c.f_JxBeam)/hx;
         rJyDx[n] = (cp.f_JyBeam - c.f_JyBeam)/hx;
         rJzDx[n] = (cp.f_JzBeam - c.f_JzBeam)/hx;

         if (fabs(rRho[n]) > maxRho) maxRho = rRho[n];
         if (fabs(rJx[n]) > maxJx) maxJx = rJx[n];
         if (fabs(rJy[n]) > maxJy) maxJy = rJy[n];
         if (fabs(rJz[n]) > maxJz) maxJz = rJz[n];
         if (fabs(rJxDx[n]) > maxJxDx) maxJxDx = rJxDx[n];
         if (fabs(rJyDx[n]) > maxJyDx) maxJyDx = rJyDx[n];
         if (fabs(rJzDx[n]) > maxJzDx) maxJzDx = rJzDx[n];
      }
   }

   fftw_execute(planR2C_Rho);
   fftw_execute(planR2C_Jx);
   fftw_execute(planR2C_Jy);
   fftw_execute(planR2C_Jz);
   fftw_execute(planR2C_JxDx);
   fftw_execute(planR2C_JyDx);
   fftw_execute(planR2C_JzDx);
//   fftw_execute(planR2C_Ey);
//   fftw_execute(planR2C_Ez);
//   fftw_execute(planR2C_By);
//   fftw_execute(planR2C_Bz);

   //------------------------ calculating Ex, Bx ----------------------------
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

   for (k=0; k<kZmax; k++)
   {
      for (j=0; j<kYmax; j++)
      {
         double aKz = 2.*PI/Zlength*k;
         double aKy = 2.*PI/Ylength*j;
         double aK2 = aKy*aKy + aKz*aKz;
         double aK2_inv = 0.;

         if (aK2 > 0.) {
            aK2_inv = 1./aK2;
         } else {
            aK2_inv = 0.;
         };

         long n1 = j + nyc*k;

         VComplex vcCoy = VComplex(-aKy*aK2_inv,0.)*I;
         VComplex vcCoz = VComplex(-aKz*aK2_inv,0.)*I;

         vcEx[n1] = vcCoy*vcJy[n1] + vcCoz*vcJz[n1];
         vcBx[n1] = vcCoz*vcJy[n1] - vcCoy*vcJz[n1];
         if (vcEx[n1].abs2() > maxEx) maxEx = vcEx[n1].abs2();
         if (vcBx[n1].abs2() > maxEx) maxEx = vcBx[n1].abs2();
      }
   }

   for (k=kZmax-1; k>0; k--)
   {
      for (j=0; j<kYmax; j++)
      {
         double aKz = -2.*PI/Zlength*k;
         double aKy =  2.*PI/Ylength*j;
         double aK2 = aKy*aKy + aKz*aKz;
         double aK2_inv = 1./aK2;

         long n1 = j + nyc*(nz-k);

         VComplex vcCoy = I*VComplex(-aKy*aK2_inv);
         VComplex vcCoz = I*VComplex(-aKz*aK2_inv);

         vcEx[n1] = vcCoy*vcJy[n1] + vcCoz*vcJz[n1];
         vcBx[n1] = vcCoz*vcJy[n1] - vcCoy*vcJz[n1];
         if (vcEx[n1].abs2() > maxEx) maxEx = vcEx[n1].abs2();
         if (vcBx[n1].abs2() > maxBx) maxBx = vcBx[n1].abs2();
      }
   }

   //------------------------ calculating Ey, Ez, By, Bz ----------------------------

   for (k=0; k<kZmax; k++)
   {
      for (j=0; j<kYmax; j++)
      {
         double aKz = 2.*PI/Zlength*k;
         double aKy = 2.*PI/Ylength*j;
         double aK2 = aKy*aKy + aKz*aKz;
         double aK2_inv = 0.;


         if (aK2 > 0.) {
            aK2_inv = 1./aK2;
         } else {
            aK2_inv = 0.;
         };

         long n1 = j + nyc*k;

         VComplex vcCoy = I*VComplex(-aKy*aK2_inv,0.);
         VComplex vcCoz = I*VComplex(-aKz*aK2_inv,0.);

         vcEy[n1] = vcCoy*vcRho[n1] - VComplex(-aK2_inv)*vcJyDx[n1];
         vcEz[n1] = vcCoz*vcRho[n1] - VComplex(-aK2_inv)*vcJzDx[n1];
         vcBy[n1] = -vcCoz*vcJx[n1] + VComplex(-aK2_inv)*vcJzDx[n1];
         vcBz[n1] =  vcCoy*vcJx[n1] - VComplex(-aK2_inv)*vcJyDx[n1];

         if (vcEy[n1].abs2() > maxEy) maxEy = vcEy[n1].abs2();
         if (vcBy[n1].abs2() > maxBy) maxBy = vcBy[n1].abs2();
         if (vcEz[n1].abs2() > maxEz) maxEz = vcEz[n1].abs2();
         if (vcBz[n1].abs2() > maxBz) maxBz = vcBz[n1].abs2();
      }
   }

   for (k=kZmax-1; k>0; k--)
   {
      for (j=0; j<kYmax; j++)
      {
         double aKz = -2.*PI/Zlength*k;
         double aKy = 2.*PI/Ylength*j;
         double aK2 = aKy*aKy + aKz*aKz;
         double aK2_inv = 0.;

         aK2_inv = 1./(aK2+dens);

         long n1 = j + nyc*(nz-k);

         VComplex vcCoy = I*VComplex(-aKy*aK2_inv, 0.);
         VComplex vcCoz = I*VComplex(-aKz*aK2_inv, 0.);
         VComplex vcDensK2 = VComplex(dens*aK2_inv,0.);


         vcEy[n1] = vcCoy*vcRho[n1] - VComplex(-aK2_inv)*vcJyDx[n1];
         vcEz[n1] = vcCoz*vcRho[n1] - VComplex(-aK2_inv)*vcJzDx[n1];
         vcBy[n1] = -vcCoz*vcJx[n1] + VComplex(-aK2_inv)*vcJzDx[n1];
         vcBz[n1] =  vcCoy*vcJx[n1] - VComplex(-aK2_inv)*vcJyDx[n1];

         if (vcEy[n1].abs2() > maxEy) maxEy = vcEy[n1].abs2();
         if (vcBy[n1].abs2() > maxBy) maxBy = vcBy[n1].abs2();
         if (vcEz[n1].abs2() > maxEz) maxEz = vcEz[n1].abs2();
         if (vcBz[n1].abs2() > maxBz) maxBz = vcBz[n1].abs2();
       }
   }

   if (maxEy > 0) {
      double check4 = 0.;
   }

   double wp = sqrt(dens);
   double arg = fabs(hx*wp);
   VComplex phi1;
   if (arg>1e-6) {
      phi1 = VComplex(sin(arg)/arg, (cos(arg)-1.)/arg);
   } else {
      phi1 = VComplex(1. - arg*arg/6., -arg*arg/2.);
   }

   VComplex phi_m, phi_p, phi_0;
   phi_p = VComplex(cos(arg),-sin(arg));
   phi_m = VComplex(cos(arg),sin(arg));
   phi_0 = VComplex(cos(arg),0.);

   for (int n=0; n<ncomplex; n++) {
      int n1 = n;

      vcEx[n1] = vcExp[n1] + hx*phi1*(vcEx[n1] - wp*I*vcExp[n1]);
      vcEy[n1] = vcEyp[n1] + hx*phi1*(vcEy[n1] - wp*I*vcEyp[n1]);
      vcEz[n1] = vcEzp[n1] + hx*phi1*(vcEz[n1] - wp*I*vcEzp[n1]);
      vcBx[n1] = vcBxp[n1] + hx*phi1*(vcBx[n1] - wp*I*vcBxp[n1]);
      vcBy[n1] = vcByp[n1] + hx*phi1*(vcBy[n1] - wp*I*vcByp[n1]);
      vcBz[n1] = vcBzp[n1] + hx*phi1*(vcBz[n1] - wp*I*vcBzp[n1]);

      vcExp[n] = vcEx[n];
      vcEyp[n] = vcEy[n];
      vcEzp[n] = vcEz[n];
      vcBxp[n] = vcBx[n];
      vcByp[n] = vcBy[n];
      vcBzp[n] = vcBz[n];
   };

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

   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;
   maxEx = maxEy = maxEz = maxBx = maxBy = maxBz = 0.;

   i = iIn;
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

         sumEx += rEx[n]*rEx[n];
         sumEy += rEy[n]*rEy[n];
         sumEz += rEz[n]*rEz[n];
         sumBx += rBx[n]*rBx[n];
         sumBy += rBy[n]*rBy[n];
         sumBz += rBz[n]*rBz[n];

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

//--- Mesh:: ----------------------->
//------------------------- maintain the constant wake wavelength ---------------------------
double Mesh::WakeControl(void) {
   if (domain()->GetCntrl()->GetWakeControlPeriod() == 0) return 0.;
   if (domain()->GetCntrl()->GetWakeControlFlag() == 0) return 0.;
   long i, j, k;
   double a, b, c;
   double ts = Ts();
   double xFirstZero = 0;
   int nPEs = domain()->nPEs();
   int nPE = domain()->nPE();
   int iPE = domain()->iPE();
   int jPE = domain()->jPE();
   int kPE = domain()->kPE();
   int Xpartition = domain()->Xpartition();
   int Ypartition = domain()->Ypartition();
   int Zpartition = domain()->Zpartition();

   int odd_k = Zpartition % 2;
   int odd_j = Ypartition % 2;

   int k_p = 0;
   int j_p = 0;
   //  int i_p = 1;

   if(!odd_k)
   {
      // even number
      k_p = (Zpartition / 2) - 1;
      k     = l_Mz-1;
   } else {
      k_p = (Zpartition - 1)/2;
      k     = l_Mz/2;
   }

   if(!odd_j)
   {
      // even number
      j_p = (Ypartition / 2) - 1;
      j = l_My-1;
   } else {
      j_p = (Ypartition - 1)/2;
      j = l_My/2;
   }

   if(kPE != k_p && jPE != j_p && nPE != 0) {
      return 0.;
   }

   b = .25;
   c = -b/4.;
   a = 1. - 2.*b - 2.*c; 

   i = 0;
   double* ExOnAxis = new double[l_Mx*Xpartition];
   long lc = GetN(i,j,k);
   long lm = GetN(i-1,j,k);
   long lmm = GetN(i-2,j,k);
   long lp = GetN(i+1,j,k);
   long lpp = GetN(i+2,j,k);
   for (int i=0; i<l_Mx; i++) {
      Cell &cc = p_CellArray[lc++];
      Cell &m = p_CellArray[lm++];
      Cell &mm = p_CellArray[lmm++];
      Cell &p = p_CellArray[lp++];
      Cell &pp = p_CellArray[lpp++];

      ExOnAxis[i] = c*(mm.f_Ex + pp.f_Ex) + b*(m.f_Ex + p.f_Ex) +a*cc.f_Ex;
   };

   int ip = 0;
   int masterPE = ip + Xpartition*(jPE + Ypartition*kPE);

#ifdef V_MPI
   MPI_Status mstatus;
   int msgtag = 177;
   int ierr = 0;

   if (nPE != 0) {
      int ierr = MPI_Send(ExOnAxis, l_Mx, 
         MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
   } else { // masterPE
      for (ip=0; ip<Xpartition; ip++) {
         int fromN = masterPE + ip;
         if (fromN == 0) continue;
         int ierr = MPI_Recv(ExOnAxis+l_Mx*ip, l_Mx, 
            MPI_DOUBLE, fromN, msgtag, MPI_COMM_WORLD, &mstatus);
      }
   }
#endif

   double wake_change = 0.;

   if (nPE == 0) {
      for (i=0; i<l_Mx*Xpartition; i++) {
         if (i>1 && ExOnAxis[i]>0 && ExOnAxis[i-2]<0) {
            xFirstZero = (X(i-2)*ExOnAxis[i] - X(i)*ExOnAxis[i-2])/
               (ExOnAxis[i] - ExOnAxis[i-2]);
            if (i_OptimalWakeRecorded == 0) {
               i_OptimalWakeRecorded = 1;
               f_WakeZeroPosition = xFirstZero 
                  + (domain()->GetCntrl()->GetShift()*Hx() - domain()->GetPhase());
               wake_change = 0.;
               break;
            } else {
               f_RecentWakeZeroPosition = xFirstZero;
               double deltaX = xFirstZero - f_WakeZeroPosition;
               double deltaXoverL = deltaX / domain()->GetXlength();
               double deltaNrelative = -deltaXoverL;
               wake_change = deltaNrelative;
               break;
            }
         }
      }
   }
   delete[] ExOnAxis;
   return wake_change;
};

void Mesh::AddWakeCorrection(double dDensityChange)   {
   f_WakeCorrection *= (1. + dDensityChange);
   if (domain()->nPE() == 0) {
      fprintf(pf_FileWakeControl,"%g %g %g %g \n",
         domain()->GetPhase(),f_WakeCorrection,f_WakeZeroPosition,f_RecentWakeZeroPosition);
      fflush(pf_FileWakeControl);
      domain()->Getout_Flog() << "Wake correction is now " << f_WakeCorrection << 
         " the first wake zero is located at " << f_RecentWakeZeroPosition << 
         " it should be at " << f_WakeZeroPosition << endl;
   };
}; 
