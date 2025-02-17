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
   double hx  = Hx();
   double hy = Hy();
   double hz = Hz();

   complex <double> I = complex <double>(0.,1.);
//   I.re = 0.;
//   I.im = 1.;
   int ny = l_My;
   int nz = l_Mz;
   int nyc = ny/2 + 1;
   int ncomplex = nz*nyc;
   double sumEx, sumEy, sumEz;
   double sumBx, sumBy, sumBz;
   sumEx = sumEy = sumEz = 0.;
   sumBx = sumBy = sumBz = 0.;


   double maxRho, maxJx, maxJy, maxJz, maxJxDx, maxJyDx, maxJzDx;
   maxRho = maxJx = maxJy = maxJz = maxJxDx = maxJyDx = maxJzDx = 0;
   double total_dens = 0.;

   for (int i=-l_dMx; i<l_Mx+l_dMx; i++) {
      ClearCurrents(i);
   }

   for (k=-l_dMz; k<l_Mz+l_dMz; k++)
   {
      for (j=-l_dMy; j<l_My+l_dMy; j++)
      {
         Cell &pcc = GetCell(l_Mx,j,k);
         Cell &ccc = GetCell(l_Mx-1,j,k);
         pcc.f_Dens = ccc.f_Dens = dens;
      }
   }

   for (i=l_Mx-1; i>=0; i--) {
      total_dens = 0.;
      for (j=0; j<l_My; j++) {
         Cell &cup1 = GetCell(i,j,l_Mz);
         Cell &cup0 = GetCell(i,j,l_Mz-1);
         Cell &cdown1 = GetCell(i,j,-1);
         Cell &cdown0 = GetCell(i,j,0);
         cup1.f_Dens = dens;
         cup1.f_Jx = cdown0.f_Jx;
         cup1.f_Jy = cdown0.f_Jy;
         cup1.f_Jz = cdown0.f_Jz;
         cup1.f_Ex = cdown0.f_Ex;
         cup1.f_Ey = cdown0.f_Ey;
         cup1.f_Ez = cdown0.f_Ez;

         cdown1.f_Dens = dens;
         cdown1.f_Jx = cup0.f_Jx;
         cdown1.f_Jy = cup0.f_Jy;
         cdown1.f_Jz = cup0.f_Jz;
         cdown1.f_Ex = cup0.f_Ex;
         cdown1.f_Ey = cup0.f_Ey;
         cdown1.f_Ez = cup0.f_Ez;
         cup1.f_Jx = cup1.f_Jy = cup1.f_Jz = cup1.f_Ex = cup1.f_Ey = cup1.f_Ez = 0.;
         cdown1.f_Jx = cdown1.f_Jy = cdown1.f_Jz = cdown1.f_Ex = cdown1.f_Ey = cdown1.f_Ez = 0.;
      };
      for (k=0; k<l_Mz; k++) {
         Cell &cup1 = GetCell(i,l_My,k);
         Cell &cup0 = GetCell(i,l_My-1,k);
         Cell &cdown1 = GetCell(i,-1,k);
         Cell &cdown0 = GetCell(i,0,k);
         cup1.f_Dens = dens;
         cup1.f_Jx = cdown0.f_Jx;
         cup1.f_Jy = cdown0.f_Jy;
         cup1.f_Jz = cdown0.f_Jz;
         cup1.f_Ex = cdown0.f_Ex;
         cup1.f_Ey = cdown0.f_Ey;
         cup1.f_Ez = cdown0.f_Ez;
         cup1.f_Jx = cup1.f_Jy = cup1.f_Jz = cup1.f_Ex = cup1.f_Ey = cup1.f_Ez = 0.;

         cdown1.f_Dens = dens;
         cdown1.f_Jx = cup0.f_Jx;
         cdown1.f_Jy = cup0.f_Jy;
         cdown1.f_Jz = cup0.f_Jz;
         cdown1.f_Ex = cup0.f_Ex;
         cdown1.f_Ey = cup0.f_Ey;
         cdown1.f_Ez = cup0.f_Ez;
         cdown1.f_Jx = cdown1.f_Jy = cdown1.f_Jz = cdown1.f_Ex = cdown1.f_Ey = cdown1.f_Ez = 0.;
      };
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
            Cell &ccc = GetCell(i,j,k);
            Cell &pcc = GetCell(i+1,j,k);
            Cell &mcc = GetCell(i-1,j,k);;
            Cell &cmc = GetCell(i,j-1,k);
            Cell &cpc = GetCell(i,j+1,k);
            Cell &ccm = GetCell(i,j,k-1);
            Cell &ccp = GetCell(i,j,k+1);

            if (ccc.f_RhoBeam != 0.) {
               double rb = ccc.f_RhoBeam;
            }

            mcc.f_Ey = -pcc.f_Ey + (cmc.f_Ey + cpc.f_Ey)/(1.+ccc.f_Dens*hy*hy/2.) 
               + 0.5*hy*(cmc.f_Dens + cmc.f_RhoBeam - cpc.f_Dens - cpc.f_RhoBeam)/(1.+ccc.f_Dens*hy*hy/2.)
               + hy*hy/(2.*hx)*(cpc.f_JyBeam - cmc.f_JyBeam)/(1.+ccc.f_Dens*hy*hy/2.);
            mcc.f_Jy = pcc.f_Jy + hx*ccc.f_Dens*(mcc.f_Ey + pcc.f_Ey);

            mcc.f_Ex = -pcc.f_Ex + cmc.f_Ex + cpc.f_Ex + 0.5*hy*(cmc.f_Jy + cmc.f_JyBeam - cpc.f_Jy - cpc.f_JyBeam);
            mcc.f_Jx = pcc.f_Jx + hx*ccc.f_Dens*(mcc.f_Ex + pcc.f_Ex);

            mcc.f_Dens = pcc.f_Dens + mcc.f_Jx - pcc.f_Jx - hx/hy*(cpc.f_Jy - cmc.f_Jy);

            if (mcc.f_Dens < 0.) {
               mcc.f_Dens = 0.;
            };
            total_dens += mcc.f_Dens;
         }
      }
      cout <<"Density at i="<<i<< " is " << total_dens << endl;
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

