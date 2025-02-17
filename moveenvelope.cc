#include "vlpl3d.h"
#include <math.h>

//--- Mesh:: ----------------------->
void Mesh::MoveEnvelope(void)
{
   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();
   double ts = 2*PI*Ts();
   double hx = Hx();
   double hy = Hy();
   double hz = Hz(); 

   long i, j, k;

   for (k=0; k<l_Mz+1; k++)
   {
      for (j=0; j<l_My+1; j++)
      {
         i = -1;
         long lccc = GetN(i,j,k);
         long lcmc = GetN(i,j-1,k);
         long lccm = GetN(i,j,k-1);
         long lcmm = GetN(i,j-1,k-1);

         for (i=-1; i<l_Mx+1; i++)
         {
            Cell &ccc = p_CellArray[lccc];
            Cell &cmm = p_CellArray[lcmm];
            Cell &cmc = p_CellArray[lcmc];
            Cell &ccm = p_CellArray[lccm];

            ccc.f_Jy -= 0.5*ts2hz*(ccc.f_Bx + cmc.f_Bx - ccm.f_Bx - cmm.f_Bx);
            ccc.f_Jz += 0.5*ts2hy*(ccc.f_Bx + ccm.f_Bx - cmc.f_Bx - cmm.f_Bx);

            ccc.f_Jy0 = -0.5*ts2hy*(ccc.f_Ex + ccm.f_Ex - cmc.f_Ex - cmm.f_Ex);
            ccc.f_Jz0 =  0.5*ts2hz*(ccc.f_Ex + cmc.f_Ex - ccm.f_Ex - cmm.f_Ex);

            ccc.f_EyAv = ccc.f_Ey;
            ccc.f_ByAv = ccc.f_By;
            ccc.f_EzAv = ccc.f_Ez;
            ccc.f_BzAv = ccc.f_Bz;

            //	      ccc.f_BxAv = ccc.f_Jx;
            lccc++; lcmc++; lccm++; lcmm++;
         }
      }
   }


   for (k=0; k<l_Mz+1; k++)
   {
      for (j=0; j<l_My+1; j++)
      {
         i = 0;
         long lccc = GetN(i,j,k);
         long lpcc = GetN(i+1,j,k);
         long lmcc = GetN(i-1,j,k);

         double eym = 0.;
         double ezm = 0.;
         double bym = 0.;
         double bzm = 0.;
         double ey = 0.;
         double ez = 0.;
         double by = 0.;
         double bz = 0.;

         for (i=0; i<l_Mx; i++)
         {
            Cell &ccc = p_CellArray[lccc];
            Cell &pcc = p_CellArray[lpcc];
            Cell &mcc = p_CellArray[lmcc];

            ey = 0.5*(pcc.f_Ey + mcc.f_Ey + mcc.f_Bz - pcc.f_Bz) - 0.25*(pcc.f_Jy  + 2*ccc.f_Jy  + mcc.f_Jy  + mcc.f_Jy0 - pcc.f_Jy0);
            bz = 0.5*(pcc.f_Bz + mcc.f_Bz + mcc.f_Ey - pcc.f_Ey) - 0.25*(pcc.f_Jy0 + 2*ccc.f_Jy0 + mcc.f_Jy0 + mcc.f_Jy  - pcc.f_Jy );

            ez = 0.5*(pcc.f_Ez + mcc.f_Ez + pcc.f_By - mcc.f_By) - 0.25*(pcc.f_Jz  + 2*ccc.f_Jz  + mcc.f_Jz  + pcc.f_Jz0 - mcc.f_Jz0);
            by = 0.5*(pcc.f_By + mcc.f_By + pcc.f_Ez - mcc.f_Ez) - 0.25*(pcc.f_Jz0 + 2*ccc.f_Jz0 + mcc.f_Jz0 + pcc.f_Jz  - mcc.f_Jz );

            mcc.f_Ey = eym;
            mcc.f_Ez = ezm;
            mcc.f_By = bym;
            mcc.f_Bz = bzm;

            eym = ey;
            ezm = ez;
            bym = by;
            bzm = bz;


            lccc++; lpcc++; lmcc++;
         }
         p_CellArray[lmcc].f_Ey = ey;
         p_CellArray[lmcc].f_Ez = ez;
         p_CellArray[lmcc].f_By = by;
         p_CellArray[lmcc].f_Bz = bz;
      }
   }

   if (!HybridIncluded()) return;
   //------------------ hybrid part -------------
   double epsilon = 1.;
   int isort = 1; 

   for (k=-1; k<l_Mz+1; k++)
   {
      for (j=-1; j<l_My+1; j++)
      {
         i = -1;
         long lccc = GetN(i,j,k);
         double b = 0; 

         for (i=-1; i<l_Mx+1; i++)
         {
            Cell &ccc = p_CellArray[lccc];

            double EyOld = ccc.f_EyAv;
            double EzOld = ccc.f_EzAv;
            Particle *p = ccc.p_Particles;
            double dens = ccc.f_DensH + ccc.f_DeltaDensH;

            double PyOld = 0.;
            double PzOld = 0.;
            double nu = 0.;

            if (dens > 0.) {
               double gamma = sqrt(1. + ccc.f_PxH*ccc.f_PxH 
                  + ccc.f_PyH*ccc.f_PyH
                  + ccc.f_PzH*ccc.f_PzH); 
               dens = dens/gamma;
               PyOld += ccc.f_PyH*dens;
               PzOld += ccc.f_PzH*dens;
               nu = dens*b;
               nu = 0.;
            };

            double conu = 1./(1.+nu*ts);
            double coe = 1./(1+0.25*dens*ts*ts);

            double dEy = ccc.f_Ey - ccc.f_EyAv;
            double dEz = ccc.f_Ez - ccc.f_EzAv;

            double newEy = coe*((1-0.25*dens*ts*ts)*EyOld + dEy - ts*PyOld);
            double newEz = coe*((1-0.25*dens*ts*ts)*EzOld + dEz - ts*PzOld);

            ccc.f_Jy0 = newEy - EyOld - dEy;
            ccc.f_Jz0 = newEz - EzOld - dEz; // Here we calculate  -ts*j_h to use it in d_t n + div(j_h) = 0

            ccc.f_Ey = newEy;
            ccc.f_Ez = newEz;

            if (dens > 0.) {
               ccc.f_PyH = conu*(0.5*ts*(newEy+EyOld) + ccc.f_PyH);
               ccc.f_PzH = conu*(0.5*ts*(newEz+EzOld) + ccc.f_PzH);
            }

            lccc++;
         }
      }
   }
}

//--- Mesh:: ----------------------.
void Mesh::MoveBfieldQ1D(void)
{
   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();

   double ts = PI*Ts();

   long i, j, k;
   double epsilon = 1.;
   int isort = 1; 

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lccc = GetN(i,j,k);
         long lcpc = GetN(i,j+1,k);
         long lccp = GetN(i,j,k+1);
         long lcpp = GetN(i,j+1,k+1);
         double b = 0; 

         for (i=0; i<l_Mx; i++)
         {
            Cell &ccc = p_CellArray[lccc];
            Cell &cpp = p_CellArray[lcpp];
            Cell &cpc = p_CellArray[lcpc];
            Cell &ccp = p_CellArray[lccp];

            ccc.f_Bx -= 0.5*(
               0.5*((cpc.f_Ez + cpp.f_Ez) - (ccc.f_Ez + ccp.f_Ez))*ts2hy
               -
               0.5*((ccp.f_Ey + cpp.f_Ey) - (ccc.f_Ey + cpc.f_Ey))*ts2hz);

            double dEx = 0.5*(
               0.5*((cpc.f_Bz + cpp.f_Bz) - (ccc.f_Bz + ccp.f_Bz))*ts2hy
               -
               0.5*((ccp.f_By + cpp.f_By) - (ccc.f_By + cpc.f_By))*ts2hz
               - ccc.f_Jx);

            if (i==2) {
               int j1t = j;
               int k1t = k;
            }
            if (HybridIncluded()) {
               double ExOld = ccc.f_Ex;
               double dens = ccc.f_DensH + ccc.f_DeltaDensH;

               double PxOld = 0.;
               double nu = 0.;

               if (dens > 0.) {
                  double gamma = sqrt(1. + ccc.f_PxH*ccc.f_PxH 
                     + ccc.f_PyH*ccc.f_PyH
                     + ccc.f_PzH*ccc.f_PzH); 
                  dens = dens/gamma;
                  PxOld = ccc.f_PxH*dens;
                  nu = dens*b;
                  nu = 0.;

                  double conu = 1./(1.+nu*ts);
                  double coe = 1./(1+0.25*dens*ts*ts);

                  double newEx = coe*((1-0.25*dens*ts*ts)*ExOld + dEx - ts*PxOld);

                  ccc.f_Jx0 = 2.*(newEx - ExOld - dEx);

                  ccc.f_Ex = newEx;
                  ccc.f_PxH = conu*(0.5*ts*(newEx+ExOld) + ccc.f_PxH);
               } else {
                  ccc.f_Ex += dEx;
               }
            } else {
               ccc.f_Ex += dEx;
            }

            lccp++; lcpc++; lccc++; lcpp++;
         }
      }
   }

   for (k=-1; k<l_Mz+1; k++)
   {
      for (j=-1; j<l_My+1; j++)
      {
         i = -1;
         long lccc = GetN(i,j,k);
         double b = 0; 

         for (i=-1; i<l_Mx+1; i++)
         {
            Cell &ccc = p_CellArray[lccc];

            lccc++;
         }
      }
   }
   //	ClearCurrents();
}

//--- Mesh:: ----------------------->
void Mesh::MoveEfieldNDF(void)
{
   double ts = 2*PI*Ts();
   double hx = Hx();
   double hy = Hy();
   double hz = Hz(); 
   double epsilon = 1.;

   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();

   double s3 = sqrt(3.);
   double hy2hx = Hy()/Hx();
   double hz2hx = Hz()/Hx();
   double co = (s3-1.)/(2.*s3)*1.25;
   double beta_x = 1.-co;
   double beta_y = 1.-co/(hy2hx*hy2hx);
   double beta_z = 1.-co/(hz2hx*hz2hx);
   double alfa_x = (1. - beta_x)*0.5;
   double alfa_y = (1. - beta_y)*0.5;
   double alfa_z = (1. - beta_z)*0.5;

   /*
   double alfa_y = 0.125*Hx()/Hy();;
   double alfa_z = 0.125*Hx()/Hz();;
   double alfa_x = alfa_y + alfa_z;

   double beta_x = 1.-2*alfa_x;
   double beta_y = 1.-2*alfa_y;
   double beta_z = 1.-2*alfa_z;
   */
   long i, j, k;

   double b = 0; 

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lccc = GetN(i,j,k);
         long lmcc = GetN(i-1,j,k);
         long lcmc = GetN(i,j-1,k);
         long lmmc = GetN(i-1,j-1,k);
         long lccm = GetN(i,j,k-1);
         long lmcm = GetN(i-1,j,k-1);
         long lcmm = GetN(i,j-1,k-1);

         long lpcc = GetN(i+1,j,k);
         long lcpc = GetN(i,j+1,k);
         long lccp = GetN(i,j,k+1);
         long lcpp = GetN(i,j+1,k+1);
         long lpcp = GetN(i+1,j,k+1);
         long lppc = GetN(i+1,j+1,k);
         long lpmc = GetN(i+1,j-1,k);
         long lpcm = GetN(i+1,j,k-1);
         long lcmp = GetN(i,j-1,k+1);
         long lmcp = GetN(i-1,j,k+1);
         long lcpm = GetN(i,j+1,k-1);
         long lmpc = GetN(i-1,j+1,k);

         for (i=0; i<l_Mx; i++)
         {
            Cell &ccc = p_CellArray[lccc];
            Cell &mcc = p_CellArray[lmcc];
            Cell &cmc = p_CellArray[lcmc];
            Cell &mmc = p_CellArray[lmmc];
            Cell &ccm = p_CellArray[lccm];
            Cell &mcm = p_CellArray[lmcm];
            Cell &cmm = p_CellArray[lcmm];

            Cell &pcc = p_CellArray[lpcc];
            Cell &cpc = p_CellArray[lcpc];
            Cell &ccp = p_CellArray[lccp];
            Cell &cpp = p_CellArray[lcpp];
            Cell &pcp = p_CellArray[lpcp];
            Cell &ppc = p_CellArray[lppc];
            Cell &pmc = p_CellArray[lpmc];
            Cell &pcm = p_CellArray[lpcm];
            Cell &mcp = p_CellArray[lmcp];
            Cell &cmp = p_CellArray[lcmp];
            Cell &cpm = p_CellArray[lcpm];
            Cell &mpc = p_CellArray[lmpc];

            double ExOld = ccc.f_Ex;
            double EyOld = ccc.f_Ey;
            double EzOld = ccc.f_Ez;
            Particle *p = ccc.p_Particles;
            double dens = ccc.f_DensH + ccc.f_DeltaDensH;
            double densx = 0.5*(dens+mcc.f_DensH + mcc.f_DeltaDensH);
            double densy = 0.5*(dens+cmc.f_DensH + cmc.f_DeltaDensH);
            double densz = 0.5*(dens+ccm.f_DensH + ccm.f_DeltaDensH);

            double PxOld = 0.;
            double PyOld = 0.;
            double PzOld = 0.;
            double nu = 0.;
            double dEx, dEy, dEz;
            dEx = dEy = dEz = 0.;

            if (dens > 0.) {
               double gamma = sqrt(1. + ccc.f_PxH*ccc.f_PxH 
                  + ccc.f_PyH*ccc.f_PyH
                  + ccc.f_PzH*ccc.f_PzH); 
               dens = dens/gamma;
               densx = densx/gamma;
               densy = densy/gamma;
               densz = densz/gamma;
               PxOld += ccc.f_PxH*densx;
               PyOld += ccc.f_PyH*densy;
               PzOld += ccc.f_PzH*densz;
               nu = dens*b;
               nu = 0.;
            };

            double conu = 1./(1.+nu*ts);
            double coe = 1./(1+0.25*dens*ts*ts);
            double coex = 1./(1+0.25*densx*ts*ts);
            double coey = 1./(1+0.25*densy*ts*ts);
            double coez = 1./(1+0.25*densz*ts*ts);

            switch(MaxwellSolver()) {
         case 2:
            return;
         default:
         case 0: // Yee

            dEx =
               (
               (cpc.f_Bz - ccc.f_Bz)*ts2hy
               - (ccp.f_By - ccc.f_By)*ts2hz
               - ccc.f_Jx);

            dEy =
               (
               (ccp.f_Bx - ccc.f_Bx)*ts2hz
               - (pcc.f_Bz - ccc.f_Bz)*ts2hx
               - ccc.f_Jy);

            dEz =
               (
               (pcc.f_By - ccc.f_By)*ts2hx
               - (cpc.f_Bx - ccc.f_Bx)*ts2hy
               - ccc.f_Jz);
            break;
         case 1: // NDF

            dEx = (
               ( (cpp.f_Bz - ccp.f_Bz + cpm.f_Bz - ccm.f_Bz)*alfa_z
               + (cpc.f_Bz - ccc.f_Bz)*beta_z )*ts2hy
               -
               ( (cpp.f_By - cpc.f_By + cmp.f_By - cmc.f_By)*alfa_y
               + (ccp.f_By - ccc.f_By)*beta_y )*ts2hz
               - ccc.f_Jx ) / ccc.f_Epsilon;

            dEy = (
               ( (pcp.f_Bx - pcc.f_Bx + mcp.f_Bx - mcc.f_Bx)*alfa_x
               + (ccp.f_Bx - ccc.f_Bx)*beta_x )*ts2hz
               -
               ( (pcp.f_Bz - ccp.f_Bz + pcm.f_Bz - ccm.f_Bz)*alfa_z
               + (pcc.f_Bz - ccc.f_Bz)*beta_z )*ts2hx
               - ccc.f_Jy) / ccc.f_Epsilon;

            dEz = (
               ( (ppc.f_By - cpc.f_By + pmc.f_By - cmc.f_By)*alfa_y
               + (pcc.f_By - ccc.f_By)*beta_y )*ts2hx
               -
               ( (ppc.f_Bx - pcc.f_Bx + mpc.f_Bx - mcc.f_Bx)*alfa_x
               + (cpc.f_Bx - ccc.f_Bx)*beta_x )*ts2hy
               - ccc.f_Jz) / ccc.f_Epsilon;
            break;
            };

            if (HybridIncluded()) {
               double newEx = coex*((1-0.25*densx*ts*ts)*ExOld + dEx - ts*PxOld);
               double newEy = coey*((1-0.25*densy*ts*ts)*EyOld + dEy - ts*PyOld);
               double newEz = coez*((1-0.25*densz*ts*ts)*EzOld + dEz - ts*PzOld);

               ccc.f_Jx = newEx - ExOld - dEx;
               ccc.f_Jy = newEy - EyOld - dEy;
               ccc.f_Jz = newEz - EzOld - dEz; // Here we calculate  -ts*j_h to use it in d_t n + div(j_h) = 0

               ccc.f_Ex = newEx;
               ccc.f_Ey = newEy;
               ccc.f_Ez = newEz;

               if (dens > 0.) {
                  ccc.f_PxH = conu*(0.5*ts*(newEx+ExOld) + ccc.f_PxH);
                  ccc.f_PyH = conu*(0.5*ts*(newEy+EyOld) + ccc.f_PyH);
                  ccc.f_PzH = conu*(0.5*ts*(newEz+EzOld) + ccc.f_PzH);

                  double px = ccc.f_PxH;
                  double py = ccc.f_PyH;
                  double pz = ccc.f_PzH;
                  double gamma = sqrt(1. + px*px + py*py + pz*pz); 

                  double bx = ccc.f_Bx;
                  double by = ccc.f_By;
                  double bz = ccc.f_Bz;

                  bx = bx/gamma;
                  by = by/gamma;
                  bz = bz/gamma;

                  double co = 2./(1. + (bx*bx) + (by*by) + (bz*bz));

                  double p3x = py*bz - pz*by + px;
                  double p3y = pz*bx - px*bz + py;
                  double p3z = px*by - py*bx + pz;

                  p3x *= co;
                  p3y *= co;
                  p3z *= co;

                  double px_new = p3y*bz - p3z*by;
                  double py_new = p3z*bx - p3x*bz;
                  double pz_new = p3x*by - p3y*bx;

                  ccc.f_PxH += px_new;
                  ccc.f_PyH += py_new;
                  ccc.f_PzH += pz_new;

               } else {
                  ccc.f_PxH = 0.;
                  ccc.f_PyH = 0.;
                  ccc.f_PzH = 0.;
               };
            } else {
               ccc.f_Ex += dEx;
               ccc.f_Ey += dEy;
               ccc.f_Ez += dEz;
            };

            lccc++; lmcc++; lcmc++; lmmc++; lccm++; lmcm++; lcmm++;
            lpcc++; lcpc++; lccp++; lpmc++; lpcm++; lmcp++; lcmp++; lcpm++; lmpc++;
            lcpp++; lpcp++; lppc++;
         }
      }
   }
}

//--- Mesh:: ----------------------->
//------------------------- updating the hybrid density from Poisson Eq. The relevant fields stored in f_Jx, f_Jy, f_Jz ---------------------------
void Mesh::AdjustHybridDensity(void) {
   double ts = 2.*PI*Ts();
   double hx = 2.*PI*Hx();
   double hy = 2.*PI*Hy();
   double hz = 2.*PI*Hz(); 
   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();

   if (HybridIncluded()<2) return;

   long i, j, k;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lccc = GetN(i,j,k);
         long lpcc = GetN(i+1,j,k);
         long lcpc = GetN(i,j+1,k);
         long lccp = GetN(i,j,k+1);
         long lmcc = GetN(i-1,j,k);
         long lcmc = GetN(i,j-1,k);
         long lccm = GetN(i,j,k-1);
         for (i=0; i<l_Mx; i++)
         {
            Cell &ccc = p_CellArray[lccc];

            Cell &pcc = p_CellArray[lpcc];
            Cell &cpc = p_CellArray[lcpc];
            Cell &ccp = p_CellArray[lccp];

            Cell &mcc = p_CellArray[lmcc];
            Cell &cmc = p_CellArray[lcmc];
            Cell &ccm = p_CellArray[lccm];

            lccc++; lpcc++; lcpc++; lccp++; lmcc++; lcmc++; lccm++; 

            double divE = 0;
            if (MaxwellSolver() == 2) {
               divE = (pcc.f_Jx0 - ccc.f_Jx0)/hx + (cpc.f_Jy0 - ccc.f_Jy0)/hy + (ccp.f_Jz0 - ccc.f_Jz0)/hz;
            } else {
               divE = (pcc.f_Jx - ccc.f_Jx)/hx + (cpc.f_Jy - ccc.f_Jy)/hy + (ccp.f_Jz - ccc.f_Jz)/hz;
            };
            divE *=  Cell::sf_DimFields;

            ccc.f_DeltaDensH = divE;
            //if (ccc.f_DensH <= 0.) ccc.f_DeltaDensH = 0.;
            if (ccc.f_DeltaDensH < - ccc.f_DensH) {
               ccc.f_DeltaDensH = -ccc.f_DensH;
            };

            ccc.f_DensH -= ccc.f_DeltaDensH;
            ccc.f_DeltaDensH = 0.;

            if (ccc.f_DensH == 0. && ccc.f_DeltaDensH == 0.) continue;
            if (HybridIncluded()<3) continue;

            double gamma = sqrt(1. + ccc.f_PxH*ccc.f_PxH 
               + ccc.f_PyH*ccc.f_PyH
               + ccc.f_PzH*ccc.f_PzH); 

            if (ccc.f_PxH < 0.) {
               ccc.f_PxH -= ccc.f_PxH/gamma*(pcc.f_PxH - ccc.f_PxH)*ts2hx;
               ccc.f_PyH -= ccc.f_PxH/gamma*(pcc.f_PyH - ccc.f_PyH)*ts2hx;
               ccc.f_PzH -= ccc.f_PxH/gamma*(pcc.f_PzH - ccc.f_PzH)*ts2hx;
            } else {
               ccc.f_PxH -= ccc.f_PxH/gamma*(ccc.f_PxH - mcc.f_PxH)*ts2hx;
               ccc.f_PyH -= ccc.f_PxH/gamma*(ccc.f_PyH - mcc.f_PyH)*ts2hx;
               ccc.f_PzH -= ccc.f_PxH/gamma*(ccc.f_PzH - mcc.f_PzH)*ts2hx;
            }

            if (ccc.f_PyH < 0.) {
               ccc.f_PxH -= ccc.f_PyH/gamma*(cpc.f_PxH - ccc.f_PxH)*ts2hy;
               ccc.f_PyH -= ccc.f_PyH/gamma*(cpc.f_PyH - ccc.f_PyH)*ts2hy;
               ccc.f_PzH -= ccc.f_PyH/gamma*(cpc.f_PzH - ccc.f_PzH)*ts2hy;
            } else {
               ccc.f_PxH -= ccc.f_PyH/gamma*(ccc.f_PxH - cmc.f_PxH)*ts2hy;
               ccc.f_PyH -= ccc.f_PyH/gamma*(ccc.f_PyH - cmc.f_PyH)*ts2hy;
               ccc.f_PzH -= ccc.f_PyH/gamma*(ccc.f_PzH - cmc.f_PzH)*ts2hy;
            }

            if (ccc.f_PzH < 0.) {
               ccc.f_PxH -= ccc.f_PzH/gamma*(ccp.f_PxH - ccc.f_PxH)*ts2hz;
               ccc.f_PyH -= ccc.f_PzH/gamma*(ccp.f_PyH - ccc.f_PyH)*ts2hz;
               ccc.f_PzH -= ccc.f_PzH/gamma*(ccp.f_PzH - ccc.f_PzH)*ts2hz;
            } else {
               ccc.f_PxH -= ccc.f_PzH/gamma*(ccc.f_PxH - ccm.f_PxH)*ts2hz;
               ccc.f_PyH -= ccc.f_PzH/gamma*(ccc.f_PyH - ccm.f_PyH)*ts2hz;
               ccc.f_PzH -= ccc.f_PzH/gamma*(ccc.f_PzH - ccm.f_PzH)*ts2hz;
            }
         }
      }
   }
}

//--- Mesh:: ----------------------.
void Mesh::MoveBfieldNDF(void)
{
   double ts2hx = 0.5*Ts()/Hx();
   double ts2hy = 0.5*Ts()/Hy();
   double ts2hz = 0.5*Ts()/Hz();


   double s3 = sqrt(3.);
   double hy2hx = Hy()/Hx();
   double hz2hx = Hz()/Hx();
   double co = (s3-1.)/(2.*s3)*1.25;
   double beta_x = 1.-co;
   double beta_y = 1.-co/(hy2hx*hy2hx);
   double beta_z = 1.-co/(hz2hx*hz2hx);
   double alfa_x = (1. - beta_x)*0.5;
   double alfa_y = (1. - beta_y)*0.5;
   double alfa_z = (1. - beta_z)*0.5;
   /*

   double alfa_y = 0.125*Hx()/Hy();;
   double alfa_z = 0.125*Hx()/Hz();;
   double alfa_x = alfa_y + alfa_z;

   double beta_x = 1.-2*alfa_x;
   double beta_y = 1.-2*alfa_y;
   double beta_z = 1.-2*alfa_z;
   */
   long i, j, k;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lccc = GetN(i,j,k);
         long lmcc = GetN(i-1,j,k);
         long lcmc = GetN(i,j-1,k);
         long lppc = GetN(i+1,j+1,k);
         long lccm = GetN(i,j,k-1);
         long lpcp = GetN(i+1,j,k+1);
         long lcpp = GetN(i,j+1,k+1);

         long lpcc = GetN(i+1,j,k);
         long lcpc = GetN(i,j+1,k);
         long lccp = GetN(i,j,k+1);
         long lpmc = GetN(i+1,j-1,k);
         long lpcm = GetN(i+1,j,k-1);
         long lcmp = GetN(i,j-1,k+1);
         long lmcp = GetN(i-1,j,k+1);
         long lcpm = GetN(i,j+1,k-1);
         long lmpc = GetN(i-1,j+1,k);

         long lmmc = GetN(i-1,j-1,k);
         long lcmm = GetN(i,j-1,k-1);
         long lmcm = GetN(i-1,j,k-1);

         for (i=0; i<l_Mx; i++)
         {
            Cell &ccc = p_CellArray[lccc];
            Cell &mcc = p_CellArray[lmcc];
            Cell &cmc = p_CellArray[lcmc];
            Cell &ppc = p_CellArray[lppc];
            Cell &ccm = p_CellArray[lccm];
            Cell &pcp = p_CellArray[lpcp];
            Cell &cpp = p_CellArray[lcpp];

            Cell &pcc = p_CellArray[lpcc];
            Cell &cpc = p_CellArray[lcpc];
            Cell &ccp = p_CellArray[lccp];
            Cell &pmc = p_CellArray[lpmc];
            Cell &pcm = p_CellArray[lpcm];
            Cell &mcp = p_CellArray[lmcp];
            Cell &cmp = p_CellArray[lcmp];
            Cell &cpm = p_CellArray[lcpm];
            Cell &mpc = p_CellArray[lmpc];

            Cell &cmm = p_CellArray[lcmm];
            Cell &mcm = p_CellArray[lmcm];
            Cell &mmc = p_CellArray[lmmc];

            switch(MaxwellSolver()) {
         case 2:
            return;
         default:
         case 0: // Yee

            ccc.f_Bx -=
               (ccc.f_Ez - cmc.f_Ez)*ts2hy
               - (ccc.f_Ey - ccm.f_Ey)*ts2hz;

            ccc.f_By -=
               (ccc.f_Ex - ccm.f_Ex)*ts2hz
               - (ccc.f_Ez - mcc.f_Ez)*ts2hx;

            ccc.f_Bz -=
               (ccc.f_Ey - mcc.f_Ey)*ts2hx
               - (ccc.f_Ex - cmc.f_Ex)*ts2hy;
            break;

         case 1: // NDF
            ccc.f_Bx -=
               ( (ccp.f_Ez - cmp.f_Ez + ccm.f_Ez - cmm.f_Ez)*alfa_z
               + (ccc.f_Ez - cmc.f_Ez)*beta_z )*ts2hy
               -
               ( (cpc.f_Ey - cpm.f_Ey + cmc.f_Ey - cmm.f_Ey)*alfa_y
               + (ccc.f_Ey - ccm.f_Ey)*beta_y )*ts2hz;

            ccc.f_By -=
               ( (pcc.f_Ex - pcm.f_Ex + mcc.f_Ex - mcm.f_Ex)*alfa_x
               + (ccc.f_Ex - ccm.f_Ex)*beta_x )*ts2hz
               -
               ( (ccp.f_Ez - mcp.f_Ez + ccm.f_Ez - mcm.f_Ez)*alfa_z
               + (ccc.f_Ez - mcc.f_Ez)*beta_z )*ts2hx;

            ccc.f_Bz -=
               ( (cpc.f_Ey - mpc.f_Ey + cmc.f_Ey - mmc.f_Ey)*alfa_y
               + (ccc.f_Ey - mcc.f_Ey)*beta_y )*ts2hx
               -
               ( (pcc.f_Ex - pmc.f_Ex + mcc.f_Ex - mmc.f_Ex)*alfa_x
               + (ccc.f_Ex - cmc.f_Ex)*beta_x )*ts2hy;
            }

            lccc++; lmcc++; lcmc++; lppc++; lccm++; lpcp++; lcpp++;
            lpcc++; lcpc++; lccp++; lpmc++; lpcm++; lmcp++; lcmp++; lcpm++; lmpc++;
            lcmm++; lmcm++; lmmc++;
         }
      }
   }
}  
//--- Mesh:: ----------------------->
//------------------------- updating the hybrid density from Poisson Eq. The relevant fields stored in f_Jx, f_Jy, f_Jz ---------------------------
void Mesh::FilterFieldX(void) {
   if (domain()->GetCntrl()->GetFilterPeriod() == 0) return;
   if (domain()->GetCntrl()->GetFilterFlag() == 0) return;
   long i, j, k;
   double a, b, c;
   double ts = Ts();
   double fFilterPeriod = domain()->GetCntrl()->GetFilterPeriod();
   double partial_filtering = ts/fFilterPeriod;

   b = .25*partial_filtering;
   c = -b/4.;
   a = 1. - 2.*b - 2.*c; 

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lc = GetN(i,j,k);
         long lm = GetN(i-1,j,k);
         long lmm = GetN(i-2,j,k);
         long lp = GetN(i+1,j,k);
         long lpp = GetN(i+2,j,k);

         for (i=0; i<l_Mx; i++)
         {
            Cell &cc = p_CellArray[lc];
            Cell &m = p_CellArray[lm];
            Cell &mm = p_CellArray[lmm];
            Cell &p = p_CellArray[lp];
            Cell &pp = p_CellArray[lpp];

            cc.f_ExAv = c*(mm.f_Ex + pp.f_Ex) + b*(m.f_Ex + p.f_Ex) +a*cc.f_Ex;
            cc.f_EyAv = c*(mm.f_Ey + pp.f_Ey) + b*(m.f_Ey + p.f_Ey) +a*cc.f_Ey;
            cc.f_EzAv = c*(mm.f_Ez + pp.f_Ez) + b*(m.f_Ez + p.f_Ez) +a*cc.f_Ez;

            cc.f_BxAv = c*(mm.f_Bx + pp.f_Bx) + b*(m.f_Bx + p.f_Bx) +a*cc.f_Bx;
            cc.f_ByAv = c*(mm.f_By + pp.f_By) + b*(m.f_By + p.f_By) +a*cc.f_By;
            cc.f_BzAv = c*(mm.f_Bz + pp.f_Bz) + b*(m.f_Bz + p.f_Bz) +a*cc.f_Bz;

            lc++; lm++; lmm++; lp++; lpp++;
         };
      };
   };

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i = 0;
         long lc = GetN(i,j,k);
         for (i=0; i<l_Mx; i++)
         {
            Cell &cc = p_CellArray[lc];

            cc.f_Ex = cc.f_ExAv;
            cc.f_Ey = cc.f_EyAv;
            cc.f_Ez = cc.f_EzAv;

            cc.f_Bx = cc.f_BxAv;
            cc.f_By = cc.f_ByAv;
            cc.f_Bz = cc.f_BzAv;

            lc++;
         };
      };
   };
};
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
