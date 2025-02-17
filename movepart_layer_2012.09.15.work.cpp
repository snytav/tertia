#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include <complex>
#include <math.h>
#include "vlpl3d.h"

static double maxVx;

//---Mesh:: ---------------------------------------------->
void Mesh::MoveAllLayers() 
{
   for (int iLayer=-l_dMx; iLayer<l_Mx+l_dMx; iLayer++) {
      ClearCurrents( iLayer);
   }

   SeedFrontParticles();
   maxVx = 0.;
   for (int iLayer=l_Mx-1; iLayer>-1; iLayer--) {
      MoveLayer(iLayer);
   }
   cout << " maxVx = " << maxVx ;
};

//---Mesh:: ---------------------------------------------->
void Mesh::MoveLayer(int iLayer) 
{
   long i, j, k, n;
   double part = 1.;
   int iFullStep = 0;
   double maxEx =0.;
   double maxEy =0.;
   double maxEz =0.;
   double maxRho = 0.;
   double minRho = 0.;
   double totalJy, totalJz;

   GuessFieldsHydroLinLayer(iLayer);
   ExchangeFields(iLayer);

   i= iLayer;
   j = k = 0;
   n = GetN(iLayer,  j,  k);
   Cell &c = p_CellArray[n];
   Cell &cm = p_CellArray[n-1];
   double Exc = c.f_Ex;
   double Exm = cm.f_Ex;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long ncc = GetN(iLayer,  j,  k);
         Cell &ccc = p_CellArray[ncc];
         maxEx = max(maxEx, fabs(ccc.f_Ex));
      }
   }

   int niter = 1;
   for (int iter=0; iter<niter; iter++) {
      iFullStep = 0;
      ClearCurrents( iLayer);
      ClearRho( iLayer );
      MoveParticlesLayer(iLayer, iFullStep, part); 
      ExchangeCurrents(iLayer);
      //   ExchangeRho(iLayer);

      IterateFieldsHydroLinLayer(iLayer);
      ExchangeFields(iLayer);
   };

   maxRho = minRho = c.f_Dens;
   totalJy = totalJz = 0.;
   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long ncc = GetN(iLayer,  j,  k);
         Cell &ccc = p_CellArray[ncc];
         totalJy += ccc.f_Jy;
         totalJz += ccc.f_Jz;
         maxRho = max(maxRho, ccc.f_Dens);
         minRho = min(minRho, ccc.f_Dens);
      }
   }

   ClearCurrents( iLayer);
   ClearRho( iLayer );
   part = 1.;
   iFullStep = 1;
   MoveParticlesLayer(iLayer, iFullStep, part); 
   ExchangeCurrents(iLayer);
//   ExchangeRho(iLayer);
   double err = IterateFieldsHydroLinLayer(iLayer);
   ExchangeFields(iLayer);

   maxRho = minRho = c.f_Dens;
   totalJy = totalJz = 0.;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long ncc = GetN(iLayer,  j,  k);
         Cell &ccc = p_CellArray[ncc];
         totalJy += ccc.f_Jy;
         totalJz += ccc.f_Jz;
         maxRho = max(maxRho, ccc.f_Dens);
         minRho = min(minRho, ccc.f_Dens);
      }
   }

   return;

   while (IterateFieldsHydroLinLayer(iLayer) > 1e-5) {
      ExchangeFields(iLayer);
      ClearCurrents( iLayer);
      part = 1.;
      MoveParticlesLayer(iLayer, iFullStep, part); 
      ExchangeCurrents(iLayer);
   };

   iFullStep = 1;
   MoveParticlesLayer(iLayer, iFullStep, part); 
   //ExchangeCurrents(iLayer);
}

//---Mesh:: ---------------------------------------------->
void Mesh::MoveParticlesLayer(int iLayer, int iFullStep, double part) 
{
   double Vx = 0.;
   l_Processed = 0;
   f_GammaMax  = 1.;
   double ElaserPhoton = 1.2398e-4/domain()->GetWavelength();
   int isort;
   long i, j, k, n;
   double ts = Ts();
   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();
   double hx = Hx()*part;
   double hy = Hy();
   double hz = Hz();

   i = iLayer;
   j = l_My/2.;
   k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;
   double zco = Z(k) - domain()->GetZlength()/2.;

   double dens = 0.;
   int nsorts = domain()->GetNsorts();
   for (isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += spec->Density(xco,yco,zco)*spec->GetQ2M();
   };

/*
   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long nccc = GetN(iLayer,  j,  k);
         Cell &ccc = p_CellArray[nccc];
         long npcc = nccc + 1;
         Cell &pcc = p_CellArray[npcc];
         ccc.f_Jx = pcc.f_Jx + hx*fabs(dens)*(ccc.f_Ex+pcc.f_Ex)/2.;
         ccc.f_Jy = pcc.f_Jy + hx*fabs(dens)*(ccc.f_Ey+pcc.f_Ey)/2.;
         ccc.f_Jz = pcc.f_Jz + hx*fabs(dens)*(ccc.f_Ez+pcc.f_Ez)/2.;
         Vx = 0.5*(ccc.f_Jx + pcc.f_Jx);
         maxVx = max(maxVx,fabs(Vx));

         ccc.f_Jx *= 1./(1.-Vx);
         ccc.f_Jy *= 1./(1.-Vx);
         ccc.f_Jz *= 1./(1.-Vx);
      }
   }

   return;
   */

   double* djx0= new double[nsorts];
   double* djy0= new double[nsorts];
   double* djz0= new double[nsorts];
   double* drho0= new double[nsorts];

   Cell ctmp;

   double bXext = domain()->GetBxExternal()/ctmp.sf_DimFields;
   double bYext = domain()->GetByExternal()/ctmp.sf_DimFields;
   double bZext = domain()->GetBzExternal()/ctmp.sf_DimFields;
   int ifscatter = domain()->GetSpecie(0)->GetScatterFlag();
   int* iAtomTypeArray = new int[nsorts]; 

   for (isort=0; isort<nsorts; isort++)
   {
      Specie* spec = domain()->GetSpecie(isort);
      spec->GetdJ( djx0[isort], djy0[isort], djz0[isort], drho0[isort] );
      iAtomTypeArray[isort] = spec->GetAtomType(); 
   }

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         i=iLayer;
         int ip = i+1;
         long npcc = GetN(ip,  j,  k);
/*
         long nppc = GetN(ip,  j+1,k);
         long npcp = GetN(ip,  j,  k+1);
         long nppp = GetN(ip,  j+1,k+1);
         long npmc = GetN(ip,  j-1,k);
         long npcm = GetN(ip,  j,  k-1);
         long npmm = GetN(ip,  j-1,k-1);
         long npmp = GetN(ip,  j-1,k+1);
         long nppm = GetN(ip,  j+1,k-1);
*/
         long nppc = npcc + l_sizeX;
         long npcp = npcc + l_sizeXY;
         long nppp = npcp + l_sizeX;
         long npmc = npcc - l_sizeX;
         long npcm = npcc - l_sizeXY;
         long npmm = npcm - l_sizeX;
         long npmp = npcp - l_sizeX;
         long nppm = nppc - l_sizeXY;

         Particle *p = NULL;
         Cell &pcc = p_CellArray[npcc];
         Cell &ppc = p_CellArray[nppc];
         Cell &pcp = p_CellArray[npcp];
         Cell &ppp = p_CellArray[nppp];
         Cell &pmc = p_CellArray[npmc];
         Cell &pcm = p_CellArray[npcm];
         Cell &pmm = p_CellArray[npmm];
         Cell &pmp = p_CellArray[npmp];
         Cell &ppm = p_CellArray[nppm];
         double djx = 0., djy = 0., djz = 0.;

         p = pcc.p_Particles;

         if (p==NULL)
            continue;

         p_PrevPart = NULL;
         while(p)
         {
            Particle *p_next = p->p_Next;
            isort = p->GetSort();
            if (isort > 0) {
               int ttest = 0;
            }
            if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
               double check1=0;
            };

            l_Processed++;
            double weight = p->f_Weight;
            double xp  = p->f_X;
            double yp  = p->f_Y;
            double zp  = p->f_Z;

            double x = xp;
            double y = yp;
            double z = zp;

            if (xp<0||xp>1 || yp<0||yp>1 || zp<0||zp>1)
            {
               domain()->out_Flog << "Wrong MoveParticles: x="
                  << xp << " y=" << yp << " z=" << zp << "\n";
               domain()->out_Flog.flush();
               exit(-212);
            }

            double px = p->f_Px;
            double py = p->f_Py;
            double pz = p->f_Pz;
            double pxp = px;
            double pyp = py;
            double pzp = pz;
            double gammap = sqrt(1. + px*px + py*py + pz*pz);
            Vx = px / gammap;
            if (fabs(Vx)>maxVx) maxVx = Vx;
            double q2m = p->f_Q2m;

            double Vxp = Vx;
            double Vyp = py/gammap;
            double Vzp = pz/gammap;

            double y_est = j + yp + Vyp/(1.-Vxp)*hx/hy;
            double z_est = k + zp + Vzp/(1.-Vxp)*hx/hz;

            while (y_est > l_My) y_est -= l_My;
            while (y_est < 0)    y_est += l_My;
            while (z_est > l_Mz) z_est -= l_Mz;
            while (z_est < 0)    z_est += l_Mz;

            int j_est = y_est;
            int k_est = z_est;

            double ym = y_est - j_est;
            double zm = y_est - z_est;

            if (ym + yp != 0.) {
               double dummy = 0.;
            }
/*
            ym = yp;
            zm = zp;

            j_est = j;
            k_est = k;
*/
/*
            long nccc = npcc - 1;
            long ncpc = nppc - 1;
            long nccp = npcp - 1;
            long ncpp = nppp - 1;

            if (j_est != j || k_est !=k) {
               nccc = GetN(i,  j,  k);
               ncpc = GetN(i,  j+1,k);
               nccp = GetN(i,  j,  k+1);
               ncpp = GetN(i,  j+1,k+1);
            }
*/
            long nccc = npcc - 1;
            long ncpc = nppc - 1;
            long nccp = npcp - 1;
            long ncpp = nppp - 1;
            long ncmc = npmc - 1;
            long nccm = npcm - 1;
            long ncmm = npmm - 1;
            long ncmp = npmp - 1;
            long ncpm = nppm - 1;

            Cell &ccc = p_CellArray[nccc];
            Cell &cpc = p_CellArray[ncpc];
            Cell &ccp = p_CellArray[nccp];
            Cell &cpp = p_CellArray[ncpp];
            Cell &cmc = p_CellArray[ncmc];
            Cell &ccm = p_CellArray[nccm];
            Cell &cmm = p_CellArray[ncmm];
            Cell &cmp = p_CellArray[ncmp];
            Cell &cpm = p_CellArray[ncpm];
/*
            double yc = y;
            double zc = z;

            double ayc = 1.-yp;
            double ayp = yp;
            double azc = 1.-zp;
            double azp = zp;

            double myc = 1.-yc;
            double myp = yc;
            double mzc = 1.-zc;
            double mzp = zc;

            double apcc = ayc*azc;
            double appc = ayp*azc;
            double apcp = ayc*azp;
            double appp = ayp*azp;

            double accc = myc*mzc;
            double acpc = myp*mzc;
            double accp = myc*mzp;
            double acpp = myp*mzp;
*/
            double ys = yp - 0.5;
            double zs = zp - 0.5;  

            double yms = ym - 0.5;
            double zms = zm - 0.5;  

            double ayc = 1.-ys*ys;
            double aym = 0.5*(-ys + ys*ys);
            double ayp = 0.5*( ys + ys*ys);
            double azc = 1.-zs*zs;
            double azm = 0.5*(-zs + zs*zs);
            double azp = 0.5*( zs + zs*zs);

            double myc = 1.-yms*yms;
            double mym = 0.5*(-yms + yms*yms);
            double myp = 0.5*( yms + yms*yms);
            double mzc = 1.-zms*zms;
            double mzm = 0.5*(-zms + zms*zms);
            double mzp = 0.5*( zms + zms*zms);

            double apcc = ayc*azc;
            double appc = ayp*azc;
            double apcp = ayc*azp;
            double appp = ayp*azp;
            double appm = ayp*azm;
            double apmp = aym*azp;
            double apmc = aym*azc;
            double apcm = ayc*azm;
            double apmm = aym*azm;

            double accc = myc*mzc;
            double acpc = myp*mzc;
            double accp = myc*mzp;
            double acpp = myp*mzp;
            double acpm = myp*mzm;
            double acmp = mym*mzp;
            double acmc = mym*mzc;
            double accm = myc*mzm;
            double acmm = mym*mzm;

            double ex, ey, ez;
            double exp, eyp, ezp;
            double bxp, byp, bzp;
            double exm, eym, ezm;
            double bxm, bym, bzm;

            double bx=0.;
            double by=0.;
            double bz=0.;

            exp = pcc.f_Ex;
//               apcc*pcc.f_Ex + appc*ppc.f_Ex + apcp*pcp.f_Ex + appp*ppp.f_Ex;

            eyp = pcc.f_Ey;
//               apcc*pcc.f_Ey + appc*ppc.f_Ey + apcp*pcp.f_Ey + appp*ppp.f_Ey;

            ezp = pcc.f_Ez;
//               apcc*pcc.f_Ez + appc*ppc.f_Ez + apcp*pcp.f_Ez + appp*ppp.f_Ez;

            bxp = pcc.f_Bx;
//               apcc*pcc.f_Bx + appc*ppc.f_Bx + apcp*pcp.f_Bx + appp*ppp.f_Bx;

            byp = pcc.f_By;
//               apcc*pcc.f_By + appc*ppc.f_By + apcp*pcp.f_By + appp*ppp.f_By;

            bzp = pcc.f_Bz;
//               apcc*pcc.f_Bz + appc*ppc.f_Bz + apcp*pcp.f_Bz + appp*ppp.f_Bz;

            exm = ccc.f_Ex;
//               accc*ccc.f_Ex + acpc*cpc.f_Ex + accp*ccp.f_Ex + acpp*cpp.f_Ex;

            eym = ccc.f_Ey;
//               accc*ccc.f_Ey + acpc*cpc.f_Ey + accp*ccp.f_Ey + acpp*cpp.f_Ey;

            ezm = ccc.f_Ez;
//               accc*ccc.f_Ez + acpc*cpc.f_Ez + accp*ccp.f_Ez + acpp*cpp.f_Ez;

            bxm = ccc.f_Bx;
//               accc*ccc.f_Bx + acpc*cpc.f_Bx + accp*ccp.f_Bx + acpp*cpp.f_Bx;

            bym = ccc.f_By;
//               accc*ccc.f_By + acpc*cpc.f_By + accp*ccp.f_By + acpp*cpp.f_By;

            bzm = ccc.f_Bz;
//               accc*ccc.f_Bz + acpc*cpc.f_Bz + accp*ccp.f_Bz + acpp*cpp.f_Bz;

/*
            exp =
               apcc*pcc.f_Ex + appc*ppc.f_Ex + apcp*pcp.f_Ex + appp*ppp.f_Ex +
               appm*ppm.f_Ex + apmp*pmp.f_Ex + apmc*pmc.f_Ex + apcm*pcm.f_Ex + apmm*pmm.f_Ex;

            eyp =
               apcc*pcc.f_Ey + appc*ppc.f_Ey + apcp*pcp.f_Ey + appp*ppp.f_Ey +
               appm*ppm.f_Ey + apmp*pmp.f_Ey + apmc*pmc.f_Ey + apcm*pcm.f_Ey + apmm*pmm.f_Ey;

            ezp =
               apcc*pcc.f_Ez + appc*ppc.f_Ez + apcp*pcp.f_Ez + appp*ppp.f_Ez +
               appm*ppm.f_Ez + apmp*pmp.f_Ez + apmc*pmc.f_Ez + apcm*pcm.f_Ez + apmm*pmm.f_Ez;

            bxp =
               apcc*pcc.f_Bx + appc*ppc.f_Bx + apcp*pcp.f_Bx + appp*ppp.f_Bx +
               appm*ppm.f_Bx + apmp*pmp.f_Bx + apmc*pmc.f_Bx + apcm*pcm.f_Bx + apmm*pmm.f_Bx;

            byp =
               apcc*pcc.f_By + appc*ppc.f_By + apcp*pcp.f_By + appp*ppp.f_By +
               appm*ppm.f_By + apmp*pmp.f_By + apmc*pmc.f_By + apcm*pcm.f_By + apmm*pmm.f_By;

            bzp =
               apcc*pcc.f_Bz + appc*ppc.f_Bz + apcp*pcp.f_Bz + appp*ppp.f_Bz +
               appm*ppm.f_Bz + apmp*pmp.f_Bz + apmc*pmc.f_Bz + apcm*pcm.f_Bz + apmm*pmm.f_Bz;

            exm =
               accc*ccc.f_Ex + acpc*cpc.f_Ex + accp*ccp.f_Ex + acpp*cpp.f_Ex +
               acpm*cpm.f_Ex + acmp*cmp.f_Ex + acmc*cmc.f_Ex + accm*ccm.f_Ex + acmm*cmm.f_Ex;

            eym =
               accc*ccc.f_Ey + acpc*cpc.f_Ey + accp*ccp.f_Ey + acpp*cpp.f_Ey +
               acpm*cpm.f_Ey + acmp*cmp.f_Ey + acmc*cmc.f_Ey + accm*ccm.f_Ey + acmm*cmm.f_Ey;

            ezm =
               accc*ccc.f_Ez + acpc*cpc.f_Ez + accp*ccp.f_Ez + acpp*cpp.f_Ez +
               acpm*cpm.f_Ez + acmp*cmp.f_Ez + acmc*cmc.f_Ez + accm*ccm.f_Ez + acmm*cmm.f_Ez;

            bxm =
               accc*ccc.f_Bx + acpc*cpc.f_Bx + accp*ccp.f_Bx + acpp*cpp.f_Bx +
               acpm*cpm.f_Bx + acmp*cmp.f_Bx + acmc*cmc.f_Bx + accm*ccm.f_Bx + acmm*cmm.f_Bx;

            bym =
               accc*ccc.f_By + acpc*cpc.f_By + accp*ccp.f_By + acpp*cpp.f_By +
               acpm*cpm.f_By + acmp*cmp.f_By + acmc*cmc.f_By + accm*ccm.f_By + acmm*cmm.f_By;

            bzm =
               accc*ccc.f_Bz + acpc*cpc.f_Bz + accp*ccp.f_Bz + acpp*cpp.f_Bz +
               acpm*cpm.f_Bz + acmp*cmp.f_Bz + acmc*cmc.f_Bz + accm*ccm.f_Bz + acmm*cmm.f_Bz;

*/
            ex = 0.5*(exp+exm);
            ey = 0.5*(eyp+eym);
            ez = 0.5*(ezp+ezm);

            double ex1 = ex;
            double ey1 = ey;
            double ez1 = ez;


            if(isort > 0 && iAtomTypeArray[isort] > 0 && iFullStep) {//if and only if ionizable ions
               int iZ = p->GetZ();

               if (iZ < iAtomTypeArray[isort]) {
                  double field = sqrt(ex*ex + ey*ey + ez*ez);
                  p->Ionize(&ccc, field);
                  p_next = p->p_Next;
                  //	      if (iZ == 0) continue;
               };
               q2m *= iZ;
               weight *= iZ;
            }

            ex *= q2m*hx/2.;
            ey *= q2m*hx/2.;
            ez *= q2m*hx/2.;
/*
            px += 2*ex;
            py += 2*ey;
            pz += 2*ez;
*/

            px += ex/(1.-Vx);
            py += ey/(1.-Vx);
            pz += ez/(1.-Vx);

            double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!

            if (f_GammaMax < gamma)
               f_GammaMax = gamma;

            double gamma_r = 1./gamma;																	 //!!!!!!

            bx += bXext;
            by += bYext;
            bz += bZext;

            double bx1 = bx;
            double by1 = by;
            double bz1 = bz;

            bx = bx*gamma_r*q2m/(1.-Vx)*hx/2.;
            by = by*gamma_r*q2m/(1.-Vx)*hx/2.;
            bz = bz*gamma_r*q2m/(1.-Vx)*hx/2.;

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

            px += ex/(1.-Vx) + px_new;
            py += ey/(1.-Vx) + py_new;
            pz += ez/(1.-Vx) + pz_new;
/*
            px += ex + px_new;
            py += ey + py_new;
            pz += ez + pz_new;
*/
            gamma = sqrt(1. + px*px + py*py + pz*pz);
            double Vxm = px/gamma;
            double Vym = py/gamma;
            double Vzm = pz/gamma;

            Vx = 0.5*(Vxm+Vxp);
            if (fabs(Vx)>maxVx) maxVx = Vx;
            double Vy = 0.5*(Vym+Vyp);
            double Vz = 0.5*(Vzm+Vzp);

            isort = p->GetSort();

            djx = weight*djx0[isort]*Vxm;
            djy = weight*djy0[isort]*Vym;
            djz = weight*djz0[isort]*Vzm;
            double drho = weight*drho0[isort];

            double xtmp = 0.;
            double ytmp = yp;
            double ztmp = zp;

            double full = 1.;
            double part_step = 1.;

            int itmp = iLayer;
            int jtmp = j;
            int ktmp = k;

            djx *= 1./(1.-Vx);
            djy *= 1./(1.-Vx);
            djz *= 1./(1.-Vx);
            drho *= 1./(1.-Vx);

            double dy = Vy*hx/hy;
            double dz = Vz*hx/hz;

            dy = dy/(1.-Vx);
            dz = dz/(1.-Vx);

            double partdx = 0.;
            double step = 1.;
            // --- first half-step

            if (j==l_My/2 && k==l_Mz/2) {
               double dummy = 0;
            };

            // Particle pusher cel per cell//////////////////////////////////////


            int j_jump = j;
            int k_jump = k;
            xtmp = 0;
            ytmp = yp;
            ztmp = zp;
            if (fabs(dy)>1. || fabs(dz)>1.) {
               if (fabs(dy) > fabs(dz)) {
                  step = partdx = fabs(dy);
               } else {
                  step = partdx = fabs(dz);
               };
            }

            if (partdx < 1.) {
               partdx = step = 1.;
            }

            while (partdx>0.) {
               if (partdx > 1.) {
                  partdx -= 1.;
                  part_step = 1./step;
               } else {
                  part_step = partdx/step;
                  partdx = 0.;
               }
               xtmp = 0.;

               ytmp += dy*part_step + j_jump;
               ztmp += dz*part_step + k_jump;

               while (ytmp > l_My) ytmp -= l_My;
               while (ytmp < 0) ytmp += l_My;
               while (ztmp > l_Mz) ztmp -= l_Mz;
               while (ztmp < 0) ztmp += l_Mz;

               int j_jump = ytmp;
               int k_jump = ztmp;
               ytmp -= j_jump;
               ztmp -= k_jump;
               if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) {
                  double checkpoint21 = 0.;
               };

               xtmp = 0;

               int itmp = iLayer;
               int jtmp = j_jump;
               int ktmp = k_jump;

               int ntmp = GetN(itmp,jtmp,ktmp);

               if (fabs(djy) > 0.) {
                  int check = 0;
               };

               DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
                  djx*part_step, djy*part_step, djz*part_step, drho*part_step);
            }
            /*
/////////////////////////// particle pusher one cell ///////////////////////
            xtmp = 0;
            ytmp = yp + dy + j;
            ztmp = zp + dz + k;

            while (ytmp > l_My) ytmp -= l_My;
            while (ytmp < 0) ytmp += l_My;
            while (ztmp > l_Mz) ztmp -= l_Mz;
            while (ztmp < 0) ztmp += l_Mz;

            jtmp = ytmp;
            ktmp = ztmp;

            ytmp -= jtmp;
            ztmp -= ktmp;
            
            DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
               djx, djy, djz, drho);
/////////////////////// end of one cell pusher ///////////////////
*/
            if (iFullStep) {
               xtmp = 0.;
               p->SetP(px,py,pz);
               p->SetX(xtmp,ytmp,ztmp);
               long nnew = GetN(iLayer,jtmp,ktmp);
               Cell &cnew = p_CellArray[nnew];
               p->p_Next = cnew.p_Particles;
               cnew.p_Particles = p;
               p->l_Cell = nnew;
               pcc.p_Particles = p_next;
            }
            p = p_next;

            if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
               double check1=0;
            };
         }
      }
   }
   long totalNe = domain()->GetSpecie(0)->GetNp();
   //   cout << "We have " << totalNe << " electrons \n";
//   cout << "Max Vx = " << maxVx << endl;

}

//---Mesh::DepositCurrentsInCell ---------------------------------------------->
void Mesh::DepositCurrentsInCell(
                                 Particle *p, int isort, 
                                 int i, int j, int k, 
                                 double Vx, double Vy, double Vz, 
                                 double x, double y, double z, 
                                 double djx, double djy, double djz, double drho)
{
   long nccc = GetN(i,j,k);
   Cell &ccc = GetCell(nccc);
   long ncpc = nccc + l_sizeX;
   long nccp = nccc + l_sizeXY;
   long ncpp = nccp + l_sizeX;
   long ncmc = nccc - l_sizeX;
   long nccm = nccc - l_sizeXY;
   long ncmm = nccm - l_sizeX;
   long ncmp = nccp - l_sizeX;
   long ncpm = ncpc - l_sizeXY;
   Cell &cpc = GetCell(ncpc);
   Cell &ccp = GetCell(nccp);
   Cell &cpp = GetCell(ncpp);
   Cell &cmc = GetCell(ncmc);
   Cell &ccm = GetCell(nccm);
   Cell &cmm = GetCell(ncmm);
   Cell &cmp = GetCell(ncmp);
   Cell &cpm = GetCell(ncpm);

   x = 0.;
/*
   double axc = 1.-x;
   double axp = x;
   double ayc = 1.-y;
   double ayp = y;
   double azc = 1.-z;
   double azp = z;

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;
*/

   x = 0.;

   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.;
   double ayc = 1.-ys*ys;
   double aym = 0.5*(-ys + ys*ys);
   double ayp = 0.5*( ys + ys*ys);
   double azc = 1.-zs*zs;
   double azm = 0.5*(-zs + zs*zs);
   double azp = 0.5*( zs + zs*zs);

   double accc = ayc*azc;
   double acpc = ayp*azc;
   double accp = ayc*azp;
   double acpp = ayp*azp;
   double acpm = ayp*azm;
   double acmp = aym*azp;
   double acmc = aym*azc;
   double accm = ayc*azm;
   double acmm = aym*azm;

   double weight = fabs(drho);
/*
   ccc.f_Jx += djx;
   ccc.f_Jy += djy;
   ccc.f_Jz += djz;
   ccc.f_Dens += drho;
*/

   ccc.f_Jx += djx*accc;
   ccc.f_Jy += djy*accc;
   ccc.f_Jz += djz*accc;

   cmc.f_Jx += djx*acmc;
   cmc.f_Jy += djy*acmc;
   cmc.f_Jz += djz*acmc;

   cpc.f_Jx += djx*acpc;
   cpc.f_Jy += djy*acpc;
   cpc.f_Jz += djz*acpc;

   ccm.f_Jx += djx*accm;
   ccm.f_Jy += djy*accm;
   ccm.f_Jz += djz*accm;

   ccp.f_Jx += djx*accp;
   ccp.f_Jy += djy*accp;
   ccp.f_Jz += djz*accp;

   cmm.f_Jx += djx*acmm;
   cmm.f_Jy += djy*acmm;
   cmm.f_Jz += djz*acmm;

   cpp.f_Jx += djx*acpp;
   cpp.f_Jy += djy*acpp;
   cpp.f_Jz += djz*acpp;

   cpm.f_Jx += djx*acpm;
   cpm.f_Jy += djy*acpm;
   cpm.f_Jz += djz*acpm;

   cmp.f_Jx += djx*acmp;
   cmp.f_Jy += djy*acmp;
   cmp.f_Jz += djz*acmp;

//   ccc.f_Dens += drho*ayc*azc;
   ccc.f_DensArray[isort] += weight*accc;

//   cmc.f_Dens += drho*aym*azc;
   cmc.f_DensArray[isort] += weight*acmc;

//   cpc.f_Dens += drho*ayp*azc;
   cpc.f_DensArray[isort] += weight*acpc;

//   ccm.f_Dens += drho*ayc*azm;
   ccm.f_DensArray[isort] += weight*accm;

//   ccp.f_Dens += drho*ayc*azp;
   ccp.f_DensArray[isort] += weight*accp;

//   cmm.f_Dens += drho*aym*azm;
   cmm.f_DensArray[isort] += weight*acmm;

//   cpp.f_Dens += drho*ayp*azp;
   cpp.f_DensArray[isort] += weight*acpp;

//   cpm.f_Dens += drho*ayp*azm;
   cpm.f_DensArray[isort] += weight*acpm;

//   cmp.f_Dens += drho*aym*azp;
   cmp.f_DensArray[isort] += weight*acmp;  

}

//---Mesh::DepositCurrentsInCell ---------------------------------------------->
void Mesh::DepositRhoInCell(
                            Particle *p, int isort, 
                            int i, int j, int k, 
                            double Vx, double Vy, double Vz, 
                            double x, double y, double z, 
                            double djx, double djy, double djz, double drho)
{
   Cell &ccc = GetCell(i,j,k);
   Cell &cpc = GetCell(i,j+1,k);
   Cell &ccp = GetCell(i,j,k+1);
   Cell &cpp = GetCell(i,j+1,k+1);

   x = 0.;

   double axc = 1.-x;
   double axp = x;
   double ayc = 1.-y;
   double ayp = y;
   double azc = 1.-z;
   double azp = z;

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;

   double weight = fabs(drho);

   ccc.f_Dens += drho*ayc*azc;
   ccc.f_DensArray[isort] += weight*accc;

   cpc.f_Dens += drho*ayp*azc;
   cpc.f_DensArray[isort] += weight*acpc;

   ccp.f_Dens += drho*ayc*azp;
   ccp.f_DensArray[isort] += weight*accp;

   cpp.f_Dens += drho*ayp*azp;
   cpp.f_DensArray[isort] += weight*acpp;
}

//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeFields(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      Cell &cc0 = GetCell(iLayer,0,k);
      Cell &cp1 = GetCell(iLayer,l_My,k);
      Cell &cc1 = GetCell(iLayer,-1,k);
      Cell &cp0 = GetCell(iLayer,l_My-1,k);
      for (int idim=0; idim<FLD_DIM; idim++) {
         cc1.f_Fields[idim] = cp0.f_Fields[idim];
         cp1.f_Fields[idim] = cc0.f_Fields[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      Cell &cc0 = GetCell(iLayer,j,0);
      Cell &cp1 = GetCell(iLayer,j,l_Mz);
      Cell &cc1 = GetCell(iLayer,j,-1);
      Cell &cp0 = GetCell(iLayer,j,l_My-1);
      for (int idim=0; idim<FLD_DIM; idim++) {
         cc1.f_Fields[idim] = cp0.f_Fields[idim];
         cp1.f_Fields[idim] = cc0.f_Fields[idim];
      }
   }
};

//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeRho(int iLayer) 
{
   return;
   int nsorts = domain()->GetNsorts();
   long n0, n1;
   for (int k=-1; k<l_Mz+1; k++) {
      Cell &c0 = GetCell(iLayer,0,k);
      Cell &c1 = GetCell(iLayer,l_My,k);
      c0.f_Dens += c1.f_Dens;
      Cell &c2 = GetCell(iLayer,-1,k);
      Cell &c3 = GetCell(iLayer,l_My-1,k);
      c3.f_Dens += c2.f_Dens;
      for (int isort=0; isort<nsorts; isort++) {
         c0.f_DensArray[isort] += c1.f_DensArray[isort];
         c3.f_DensArray[isort] += c2.f_DensArray[isort];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      Cell &c0 = GetCell(iLayer,j,0);
      Cell &c1 = GetCell(iLayer,j,l_Mz);
      c0.f_Dens += c1.f_Dens;
      Cell &c2 = GetCell(iLayer,j,-1);
      Cell &c3 = GetCell(iLayer,j,l_Mz-1);
      c3.f_Dens += c2.f_Dens;
      for (int isort=0; isort<nsorts; isort++) {
         c0.f_DensArray[isort] += c1.f_DensArray[isort];
         c3.f_DensArray[isort] += c2.f_DensArray[isort];
      }
   }

   int i = iLayer;
   int j = l_My/2.;
   int k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;
   double zco = Z(k) - domain()->GetZlength()/2.;
   double dens = 0.;

   for (int isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += spec->Density(xco,yco,zco)*spec->GetQ2M();
   };

   for (int k=0; k<l_Mz; k++) {
      for (int j=0; j<l_My; j++) {
          Cell &c = GetCell(iLayer,j,k);
          c.f_Dens -= dens;
      };
   };
}
//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeCurrents(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      Cell &cc0 = GetCell(iLayer,0,k);
      Cell &cp1 = GetCell(iLayer,l_My,k);
      Cell &cc1 = GetCell(iLayer,-1,k);
      Cell &cp0 = GetCell(iLayer,l_My-1,k);
      for (int idim=0; idim<CURR_DIM; idim++) {
         cc0.f_Currents[idim] += cp1.f_Currents[idim];
         cp0.f_Currents[idim] += cc1.f_Currents[idim];
         cp1.f_Currents[idim] = cc1.f_Currents[idim] = 0.; 
//         cc0.f_Currents[idim] = cp0.f_Currents[idim] = 0.;
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      Cell &cc0 = GetCell(iLayer,j,0);
      Cell &cp1 = GetCell(iLayer,j,l_Mz);
      Cell &cc1 = GetCell(iLayer,j,-1);
      Cell &cp0 = GetCell(iLayer,j,l_My-1);
      for (int idim=0; idim<CURR_DIM; idim++) {
         cc0.f_Currents[idim] += cp1.f_Currents[idim];
         cp0.f_Currents[idim] += cc1.f_Currents[idim];
         cp1.f_Currents[idim] = cc1.f_Currents[idim] = 0.; 
//         cc0.f_Currents[idim] = cp0.f_Currents[idim] = 0.;
      }
   }
};