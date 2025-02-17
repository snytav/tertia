#include <stdlib.h>

#include <math.h>
#include "vlpl3d.h"

//---Mesh:: ---------------------------------------------->
void Mesh::MoveAllLayers() 
{
   for (int iLayer=-l_dMx; iLayer<l_Mx+l_dMx; iLayer++) {
      ClearCurrents( iLayer);
   }

   SeedFrontParticles();
   for (int iLayer=l_Mx-1; iLayer>-1; iLayer--) {
      MoveLayer(iLayer);
   }
};

//---Mesh:: ---------------------------------------------->
void Mesh::MoveLayer(int iLayer) 
{
   int iInitStep = 0;
//-------------------
   if (iLayer == l_Mx-1) {
      iInitStep = 1;
   } else {
      iInitStep = 0;
   }

   double part = 1.;
//   MoveFieldsLayer(iLayer, iInitStep, part);
   MoveFieldsHydroLinLayer(iLayer);
   for (int k=-1; k<l_Mz+1; k++) {
      for (int idim=0; idim<FLD_DIM; idim++) {
         GetCell(iLayer,-1,k).f_Fields[idim] = GetCell(iLayer,l_My-1,k).f_Fields[idim];
         GetCell(iLayer,l_My,k).f_Fields[idim] = GetCell(iLayer,0,k).f_Fields[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      for (int idim=0; idim<CURR_DIM; idim++) {
         GetCell(iLayer,j,-1).f_Fields[idim] = GetCell(iLayer,j,l_Mz-1).f_Fields[idim];
         GetCell(iLayer,j,l_Mz).f_Fields[idim] = GetCell(iLayer,j,0).f_Fields[idim];
      }
   }

   int iFullStep = 1;
   part = 1.;
   MoveParticlesLayer(iLayer, iFullStep, part); 
   for (int k=-1; k<l_Mz+1; k++) {
      for (int idim=0; idim<CURR_DIM; idim++) {
         GetCell(iLayer,0,k).f_Currents[idim] += GetCell(iLayer,l_My,k).f_Currents[idim];
         GetCell(iLayer,l_My-1,k).f_Currents[idim] += GetCell(iLayer,-1,k).f_Currents[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      for (int idim=0; idim<CURR_DIM; idim++) {
         GetCell(iLayer,j,0).f_Currents[idim] += GetCell(iLayer,j,l_Mz).f_Currents[idim];
         GetCell(iLayer,j,l_Mz-1).f_Currents[idim] += GetCell(iLayer,j,-1).f_Currents[idim];
      }
   }

}

//---Mesh:: ---------------------------------------------->
void Mesh::MoveParticlesLayer(int iLayer, int iFullStep, double part) 
{
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

   int nsorts = domain()->GetNsorts();
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
         long nccc = GetN(i,  j,  k);
         long ncpc = GetN(i,  j+1,k);
         long nccp = GetN(i,  j,  k+1);
         long ncpp = GetN(i,  j+1,k+1);
         long ncmc = GetN(i,  j-1,k);
         long nccm = GetN(i,  j,  k-1);
         long ncmm = GetN(i,  j-1,k-1);
         long ncmp = GetN(i,  j-1,k+1);
         long ncpm = GetN(i,  j+1,k-1);

         Particle *p = NULL;
         Cell &ccc = p_CellArray[nccc];
         Cell &cpc = p_CellArray[ncpc];
         Cell &ccp = p_CellArray[nccp];
         Cell &cpp = p_CellArray[ncpp];
         Cell &cmc = p_CellArray[ncmc];
         Cell &ccm = p_CellArray[nccm];
         Cell &cmm = p_CellArray[ncmm];
         Cell &cmp = p_CellArray[ncmp];
         Cell &cpm = p_CellArray[ncpm];

         double djx = 0., djy = 0., djz = 0.;

         p = ccc.p_Particles;

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
            double x  = p->f_X;
            double y  = p->f_Y;
            double z  = p->f_Z;
            double px = p->f_Px;
            double py = p->f_Py;
            double pz = p->f_Pz;
            double pxn = px;
            double pyn = py;
            double pzn = pz;
            double Vx = px / sqrt(1. + px*px + py*py + pz*pz);
            double q2m = p->f_Q2m*hx/2.;

            if (x<0||x>1 || y<0||y>1 || z<0||z>1)
            {
               domain()->out_Flog << "Wrong MoveParticles: x="
                  << x << " y=" << y << " z=" << z << "\n";
               domain()->out_Flog.flush();
               exit(-212);
            }
/*
            double aym = (-2.*y + 3*y*y - y*y*y)/6.;
            double ayc = 1.-0.5*y - y*y + 0.5*y*y*y;
            double ayp = y + 0.5*y*y -0.5*y*y*y;
            double ay2 = (-y + y*y*y)/6.;
            double azm = (-2.*z + 3*z*z - z*z*z)/6.;
            double azc = 1.-0.5*z - z*z + 0.5*z*z*z;
            double azp = z + 0.5*z*z -0.5*z*z*z;
            double az2 = (-z + z*z*z)/6.;
*/

            double ys = y - 0.5;
            double zs = z - 0.5;
/*
            double ayc = 1-ys*ys;
            double aym = 0.5*(-ys + ys*ys);
            double ayp = 0.5*( ys + ys*ys);
            double azc = 1-zs*zs;
            double azm = 0.5*(-zs + zs*zs);
            double azp = 0.5*( zs + zs*zs);
*/

            double ayc = 1.-ys*ys;
            double aym = 0.5*(-ys + ys*ys);
            double ayp = 0.5*( ys + ys*ys);
            double azc = 1.-zs*zs;
            double azm = 0.5*(-zs + zs*zs);
            double azp = 0.5*( zs + zs*zs);

            double acc = ayc*azc;
            double apc = ayp*azc;
            double acp = ayc*azp;
            double app = ayp*azp;
            double apm = ayp*azm;
            double amp = aym*azp;
            double amc = aym*azc;
            double acm = ayc*azm;
            double amm = aym*azm;

            double ex, ey, ez;

            double bx=0.;
            double by=0.;
            double bz=0.;

            ex =
               acc*ccc.f_Ex + apc*cpc.f_Ex + acp*ccp.f_Ex + app*cpp.f_Ex +
               apm*cpm.f_Ex + amp*cmp.f_Ex + amc*cmc.f_Ex + acm*ccm.f_Ex + amm*cmm.f_Ex;

            ey =
               acc*ccc.f_Ey + apc*cpc.f_Ey + acp*ccp.f_Ey + app*cpp.f_Ey +
               apm*cpm.f_Ey + amp*cmp.f_Ey + amc*cmc.f_Ey + acm*ccm.f_Ey + amm*cmm.f_Ey;

            ez =
               acc*ccc.f_Ez + apc*cpc.f_Ez + acp*ccp.f_Ez + app*cpp.f_Ez +
               apm*cpm.f_Ez + amp*cmp.f_Ez + amc*cmc.f_Ez + acm*ccm.f_Ez + amm*cmm.f_Ez;

            bx =
               acc*ccc.f_Bx + apc*cpc.f_Bx + acp*ccp.f_Bx + app*cpp.f_Bx +
               apm*cpm.f_Bx + amp*cmp.f_Bx + amc*cmc.f_Bx + acm*ccm.f_Bx + amm*cmm.f_Bx;

            by =
               acc*ccc.f_By + apc*cpc.f_By + acp*ccp.f_By + app*cpp.f_By +
               apm*cpm.f_By + amp*cmp.f_By + amc*cmc.f_By + acm*ccm.f_By + amm*cmm.f_By;

            bz =
               acc*ccc.f_Bz + apc*cpc.f_Bz + acp*ccp.f_Bz + app*cpp.f_Bz +
               apm*cpm.f_Bz + amp*cmp.f_Bz + amc*cmc.f_Bz + acm*ccm.f_Bz + amm*cmm.f_Bz;

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

            ex *= q2m;
            ey *= q2m;
            ez *= q2m;

            px += ex/(1.-Vx);
            py += ey/(1.-Vx);
            pz += ez/(1.-Vx);

            double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!

            if (f_GammaMax < gamma)
               f_GammaMax = gamma;

            double gamma_r = 1./gamma;																	 //!!!!!!
            /*
            double bx = axc*ccc.f_Bx + axm*mcc.f_Bx;
            double by= ayc*ccc.f_By + aym*cmc.f_By;
            double bz= azc*ccc.f_Bz + azm*ccm.f_Bz;
            */
            //		  double Bext = 1e-2*(PI*domain()->GetTs());
            //					by += Bext;

            bx += bXext;
            by += bYext;
            bz += bZext;
            ///////////////////////////////////////////////////////////////////////
            /////////                SCATTERING               /////////////////////
            //////////////////////////////////////////////////////////////////////

            if (isort == 0 && ifscatter) //We scatter only electrons
            {
               double P = sqrt(px*px+py*py+pz*pz);
               double IonDensityInCell = 0.;
               for (int is=1; is<nsorts; is++) {
                  IonDensityInCell += ccc.f_DensArray[is];
               };

               double Probability = p->GetScatteringProbability(P, IonDensityInCell);

               if (Probability > 0.)
               {
                  double Nx = 2*(double(rand())/RAND_MAX-0.5);
                  double Ny = 2*(double(rand())/RAND_MAX-0.5);
                  double Nz = 2*(double(rand())/RAND_MAX-0.5);

                  double N = sqrt(Nx*Nx + Ny*Ny + Nz*Nz);

                  //arbitary unitvector f =|Probability|
                  double fx = (Probability*Nx)/N;
                  double fy = (Probability*Ny)/N;
                  double fz = (Probability*Nz)/N;

                  /*float f = sqrt(fx*fx + fy*fy + fz*fz); //TO TEST.

                  cout<<"collision::"<<"prob = "<<Probability<<" f = "<<f<<endl; //TO TEST
                  */
                  bx += fx/(Ts()*PI);
                  by += fy/(Ts()*PI);
                  bz += fz/(Ts()*PI);
               }
            }

            ////////////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////// 
            double bx1 = bx;
            double by1 = by;
            double bz1 = bz;

            bx = bx*gamma_r*q2m/(1.-Vx);
            by = by*gamma_r*q2m/(1.-Vx);
            bz = bz*gamma_r*q2m/(1.-Vx);

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
            double damping = 1e-3;

            px -= damping*ts*px*gamma;
            py -= damping*ts*py*gamma;
            pz -= damping*ts*pz*gamma;
            */

            /////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////// Synchrotron Damping ///////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////

            double p2 = px*px + py*py + pz*pz;
            Synchrotron *pSynchrotron = domain()->GetSynchrotron();
            double eSynMin = pSynchrotron->GetSynMin();
            if ( 1.+p2 > eSynMin*eSynMin ) {

               double dpx = px - pxn;
               double dpy = py - pyn;
               double dpz = pz - pzn;
               double pdp = px*dpx + py*dpy + pz*dpz;

               dpx -= px*(pdp/p2); // This is now the transverse momentum change
               dpy -= py*(pdp/p2);
               dpz -= pz*(pdp/p2);

               double p2absrev = 1.f/sqrt(p2);
               double nx = px*p2absrev;
               double ny = py*p2absrev;
               double nz = pz*p2absrev;

               double dp2 = dpx*dpx + dpy*dpy + dpz*dpz;

               double dpabs = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
               double Gamma2 = 1. + p2;
               double Gamma = sqrt(Gamma2);

               double omega_c_omega_laser = 3.* Gamma2 * dpabs / (4.*PI*ts);

               double Ephoton   = ElaserPhoton*omega_c_omega_laser;
               double alfa = 1./137.;
               double nph = 4./9. * alfa * dpabs;

               /////////////////////////////
               double pt2    = py*py + pz*pz;
               double pt2_p2 = sqrt(pt2)/sqrt(p2);
               double theta  = asin(pt2_p2);
               double phi  = atan2(pz, py);

               pSynchrotron->AddParticle(p, nph, theta, phi, Ephoton);

               px -= (nph * Ephoton * nx / 5.11e5f)/(1.-Vx);
               py -= (nph * Ephoton * ny / 5.11e5f)/(1.-Vx);
               pz -= (nph * Ephoton * nz / 5.11e5f)/(1.-Vx);
            };

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            gamma = 1./sqrt(1. + px*px + py*py + pz*pz);
            Vx = px*gamma;
            double Vy = py*gamma;
            double Vz = pz*gamma;


            djx = weight*djx0[isort]*Vx;
            djy = weight*djy0[isort]*Vy;
            djz = weight*djz0[isort]*Vz;
            double drho = weight*drho0[isort];

            double dy = py*gamma/(1.-Vx)*hx/hy;
            double dz = pz*gamma/(1.-Vx)*hx/hz;

//            dy = dz = 0.;

            double ytmp = y+dy;
            double ztmp = z+dz;

            double full = 1.;

            int itmp = i;
            int j_jump = ytmp;
            int k_jump = ztmp;
            if (ytmp < 0.) {
               j_jump--;
            };
            if (ztmp < 0.) {
               k_jump--;
            };
            int jtmp = j + j_jump;
            int ktmp = k + k_jump;
            if (jtmp < 0) jtmp = -1;
            if (jtmp > l_My-1) jtmp = l_My;
            if (ktmp < 0) ktmp = -1;
            if (ktmp > l_Mz-1) ktmp = l_Mz;
            ytmp -= j_jump;
            ztmp -= k_jump;
            if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) {
               double checkpoint21 = 0.;
            };


            double xtmp = 0.;
            if (iFullStep) {
               p->SetP(px,py,pz);
               p->SetX(xtmp,ytmp,ztmp);
            };

            int ntmp = GetN(itmp,jtmp,ktmp);

            djx *= 1./(1.-Vx);
            djy *= 1./(1.-Vx);
            djz *= 1./(1.-Vx);
            drho *= 1./(1.-Vx);

            MoveInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, ytmp, ztmp, djx, djy, djz, drho);
            if (iFullStep) {
               long nnew = GetN(i-1,jtmp,ktmp);
               Cell &cnew = p_CellArray[nnew];
               p->p_Next = cnew.p_Particles;
               cnew.p_Particles = p;
               p->l_Cell = nnew;
               ccc.p_Particles = p_next;
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

}

//---Mesh::MoveInCell ---------------------------------------------->
void Mesh::MoveInCell(
                      Particle *p, int isort, 
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double y, double z, 
                      double djx, double djy, double djz, double drho)
{

   Cell &ccc = GetCell(i,j,k);
   Cell &cpc = GetCell(i,j+1,k);
   Cell &ccp = GetCell(i,j,k+1);
   Cell &cpp = GetCell(i,j+1,k+1);
   Cell &cmc = GetCell(i,j-1,k);
   Cell &ccm = GetCell(i,j,k-1);
   Cell &cmm = GetCell(i,j-1,k-1);
   Cell &cmp = GetCell(i,j-1,k+1);
   Cell &cpm = GetCell(i,j+1,k-1);

   double ys = y - 0.5;
   double zs = z - 0.5;

   double ayc = 1.-ys*ys;
   double aym = 0.5*(-ys + ys*ys);
   double ayp = 0.5*( ys + ys*ys);
   double azc = 1.-zs*zs;
   double azm = 0.5*(-zs + zs*zs);
   double azp = 0.5*( zs + zs*zs);

   double acc = ayc*azc;
   double apc = ayp*azc;
   double acp = ayc*azp;
   double app = ayp*azp;
   double apm = ayp*azm;
   double amp = aym*azp;
   double amc = aym*azc;
   double acm = ayc*azm;
   double amm = aym*azm;
   double weight = p->GetWeight();

   if (drho > 0. || acc<0. || apc<0. || amc<0. || acm<0. || apm<0. || amm<0. || app<0. || amp<0. || acp<0.) {
      double check = 0;
   }

   ccc.f_Jx += djx*acc;
   ccc.f_Jy += djy*acc;
   ccc.f_Jz += djz*acc;
   ccc.f_Dens += drho*acc;
   ccc.f_DensArray[isort] += weight*acc;

   cmc.f_Jx += djx*amc;
   cmc.f_Jy += djy*amc;
   cmc.f_Jz += djz*amc;
   cmc.f_Dens += drho*amc;
   cmc.f_DensArray[isort] += weight*amc;

   cpc.f_Jx += djx*apc;
   cpc.f_Jy += djy*apc;
   cpc.f_Jz += djz*apc;
   cpc.f_Dens += drho*apc;
   cpc.f_DensArray[isort] += weight*apc;

   ccm.f_Jx += djx*acm;
   ccm.f_Jy += djy*acm;
   ccm.f_Jz += djz*acm;
   ccm.f_Dens += drho*acm;
   ccm.f_DensArray[isort] += weight*acm;

   ccp.f_Jx += djx*acp;
   ccp.f_Jy += djy*acp;
   ccp.f_Jz += djz*acp;
   ccp.f_Dens += drho*acp;
   ccp.f_DensArray[isort] += weight*acp;

   cmm.f_Jx += djx*amm;
   cmm.f_Jy += djy*amm;
   cmm.f_Jz += djz*amm;
   cmm.f_Dens += drho*amm;
   cmm.f_DensArray[isort] += weight*amm;

   cpp.f_Jx += djx*app;
   cpp.f_Jy += djy*app;
   cpp.f_Jz += djz*app;
   cpp.f_Dens += drho*app;
   cpp.f_DensArray[isort] += weight*app;

   cpm.f_Jx += djx*apm;
   cpm.f_Jy += djy*apm;
   cpm.f_Jz += djz*apm;
   cpm.f_Dens += drho*apm;
   cpm.f_DensArray[isort] += weight*apm;

   cmp.f_Jx += djx*amp;
   cmp.f_Jy += djy*amp;
   cmp.f_Jz += djz*amp;
   cmp.f_Dens += drho*amp;
   cmp.f_DensArray[isort] += weight*amp;
}
