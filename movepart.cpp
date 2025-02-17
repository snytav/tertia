#include <stdlib.h>

#include <math.h>

#include "vlpl3d.h"

//---Mesh:: ---------------------------------------------->
void Mesh::MoveParticles() 
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
   double hx = Hx();
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


   for (int iLayer=l_Mx-1; iLayer>-1; iLayer--)
   {
      ClearCurrents(iLayer-1);
      for (k=0; k<l_Mz; k++)
      {
         for (j=0; j<l_My; j++)
         {
            i=iLayer;
            long nccc = GetN(i,  j,  k);
            long ncpc = GetN(i,  j+1,k);
            long nccp = GetN(i,  j,  k+1);
            long ncpp = GetN(i,  j+1,k+1);

            Particle *p = NULL;
            Cell &ccc = p_CellArray[nccc];
            Cell &cpc = p_CellArray[ncpc];
            Cell &ccp = p_CellArray[nccp];
            Cell &cpp = p_CellArray[ncpp];

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

               double ayc = 1.-y;
               double ayp = y;
               double azc = 1.-z;
               double azp = z;

               double acc = ayc*azc;
               double apc = ayp*azc;
               double acp = ayc*azp;
               double app = ayp*azp;
               double ex, ey, ez;

               double bx=0.;
               double by=0.;
               double bz=0.;

               ex =
                  acc*ccc.f_Ex + apc*cpc.f_Ex +
                  acp*ccp.f_Ex + app*cpp.f_Ex;

               ey =
                  acc*ccc.f_Ey + apc*cpc.f_Ey +
                  acp*ccp.f_Ey + app*cpp.f_Ey;

               ez =
                  acc*ccc.f_Ez + apc*cpc.f_Ez +
                  acp*ccp.f_Ez + app*cpp.f_Ez;

               bx =
                  acc*ccc.f_Bx + apc*cpc.f_Bx +
                  acp*ccp.f_Bx + app*cpp.f_Bx;

               by =
                  acc*ccc.f_By + apc*cpc.f_By +
                  acp*ccp.f_By + app*cpp.f_By;

               bz =
                  acc*ccc.f_Bz + apc*cpc.f_Bz +
                  acp*ccp.f_Bz + app*cpp.f_Bz;

               double ex1 = ex;
               double ey1 = ey;
               double ez1 = ez;


               if(isort > 0 && iAtomTypeArray[isort] > 0) {//if and only if ionizable ions
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

               px += (ex + px_new)/(1.-Vx);
               py += (ey + py_new)/(1.-Vx);
               pz += (ez + pz_new)/(1.-Vx);
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

               p->SetP(px,py,pz);

               djx = weight*djx0[isort]*Vx;
               djy = weight*djy0[isort]*Vy;
               djz = weight*djz0[isort]*Vz;
               double drho = weight*drho0[isort];

               double dy = py*gamma/(1.-Vx)*hx/hy;
               double dz = pz*gamma/(1.-Vx)*hx/hz;
               double ytmp = y+dy;
               double ztmp = z+dz;

               double full = 1.;

               int itmp = i-1;
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
               p->SetX(xtmp,ytmp,ztmp);

               int ntmp = GetN(itmp,jtmp,ktmp);
               Cell &ctmp = p_CellArray[ntmp];
               p->l_Cell = ntmp;

               DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, djx, djy, djz, drho);
               DepositRhoInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, djx, djy, djz, drho);
               ccc.p_Particles = p = p_next;
               if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
                  double check1=0;
               };
            }
         }
      }
   }
   long totalNe = domain()->GetSpecie(0)->GetNp();
   //   cout << "We have " << totalNe << " electrons \n";

}
