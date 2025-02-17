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

   double part = 1.;
   int iFullStep = 0;

   GuessFieldsHydroLinLayer(iLayer);
   ExchangeFields(iLayer);
//   return;

   iFullStep = 0;
   MoveParticlesLayer(iLayer, iFullStep, part); 
   ExchangeCurrents(iLayer);

   IterateFieldsHydroLinLayer(iLayer);
   ExchangeFields(iLayer);
   ClearCurrents( iLayer);
   part = 1.;
   iFullStep = 1;
   MoveParticlesLayer(iLayer, iFullStep, part); 
   ExchangeCurrents(iLayer);
   double err = IterateFieldsHydroLinLayer(iLayer);
   ExchangeFields(iLayer);

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
   for (int isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += spec->Density(xco,yco,zco)*spec->GetQ2M();
   };

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long ncc = GetN(iLayer,  j,  k);
         Cell &ccc = p_CellArray[ncc];
         ccc.f_Dens = -dens;
      }
   }

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
         long ncc = GetN(ip,  j,  k);
         long npc = GetN(ip,  j+1,k);
         long ncp = GetN(ip,  j,  k+1);
         long npp = GetN(ip,  j+1,k+1);
         long nmc = GetN(ip,  j-1,k);
         long ncm = GetN(ip,  j,  k-1);
         long nmm = GetN(ip,  j-1,k-1);
         long nmp = GetN(ip,  j-1,k+1);
         long npm = GetN(ip,  j+1,k-1);

         Particle *p = NULL;
         Cell &ccc = p_CellArray[ncc];
         Cell &cpc = p_CellArray[npc];
         Cell &ccp = p_CellArray[ncp];
         Cell &cpp = p_CellArray[npp];
         Cell &cmc = p_CellArray[nmc];
         Cell &ccm = p_CellArray[ncm];
         Cell &cmm = p_CellArray[nmm];
         Cell &cmp = p_CellArray[nmp];
         Cell &cpm = p_CellArray[npm];
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
            double Vx = px / gammap;
            double q2m = p->f_Q2m*hx/2.;

            double Vxp = Vx;
            double Vyp = py/gammap;
            double Vzp = pz/gammap;

            double y_est = j + yp + Vyp/(1.-Vxp)*hx/hy;
            double z_est = k + zp + Vzp/(1.-Vxp)*hx/hz;

            if (y_est > l_My) y_est = l_My;
            if (y_est < 0)    y_est = 0;
            if (z_est > l_Mz) z_est = l_Mz;
            if (z_est < 0)    z_est = 0;

            int j_est = y_est;
            int k_est = z_est;

            long icc = ncc - 1;
            long ipc = npc - 1;
            long icp = ncp - 1;
            long ipp = npp - 1;
            long icm = ncm - 1;
            long imc = nmc - 1;
            long imp = nmp - 1;
            long ipm = npm - 1;
            long imm = nmm - 1;

            if (j_est != j || k_est !=k) {
               icc = GetN(i,  j,  k);
               ipc = GetN(i,  j+1,k);
               icp = GetN(i,  j,  k+1);
               ipp = GetN(i,  j+1,k+1);
               imc = GetN(i,  j-1,k);
               icm = GetN(i,  j,  k-1);
               imm = GetN(i,  j-1,k-1);
               imp = GetN(i,  j-1,k+1);
               ipm = GetN(i,  j+1,k-1);
            }

            Cell &mcc = p_CellArray[icc];
            Cell &mpc = p_CellArray[ipc];
            Cell &mcp = p_CellArray[icp];
            Cell &mpp = p_CellArray[ipp];
            Cell &mmc = p_CellArray[imc];
            Cell &mcm = p_CellArray[icm];
            Cell &mmm = p_CellArray[imm];
            Cell &mmp = p_CellArray[imp];
            Cell &mpm = p_CellArray[ipm];

            double ym = y_est - j_est;
            double zm = z_est - k_est;

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

            double acc = ayc*azc;
            double apc = ayp*azc;
            double acp = ayc*azp;
            double app = ayp*azp;
            double apm = ayp*azm;
            double amp = aym*azp;
            double amc = aym*azc;
            double acm = ayc*azm;
            double amm = aym*azm;

            double amcc = myc*mzc;
            double ampc = myp*mzc;
            double amcp = myc*mzp;
            double ampp = myp*mzp;
            double ampm = myp*mzm;
            double ammp = mym*mzp;
            double ammc = mym*mzc;
            double amcm = myc*mzm;
            double ammm = mym*mzm;

            double ex, ey, ez;
            double exp, eyp, ezp;
            double bxp, byp, bzp;
            double exm, eym, ezm;
            double bxm, bym, bzm;

            double bx=0.;
            double by=0.;
            double bz=0.;

            exp =
               acc*ccc.f_Ex + apc*cpc.f_Ex + acp*ccp.f_Ex + app*cpp.f_Ex +
               apm*cpm.f_Ex + amp*cmp.f_Ex + amc*cmc.f_Ex + acm*ccm.f_Ex + amm*cmm.f_Ex;

            eyp =
               acc*ccc.f_Ey + apc*cpc.f_Ey + acp*ccp.f_Ey + app*cpp.f_Ey +
               apm*cpm.f_Ey + amp*cmp.f_Ey + amc*cmc.f_Ey + acm*ccm.f_Ey + amm*cmm.f_Ey;

            ezp =
               acc*ccc.f_Ez + apc*cpc.f_Ez + acp*ccp.f_Ez + app*cpp.f_Ez +
               apm*cpm.f_Ez + amp*cmp.f_Ez + amc*cmc.f_Ez + acm*ccm.f_Ez + amm*cmm.f_Ez;

            bxp =
               acc*ccc.f_Bx + apc*cpc.f_Bx + acp*ccp.f_Bx + app*cpp.f_Bx +
               apm*cpm.f_Bx + amp*cmp.f_Bx + amc*cmc.f_Bx + acm*ccm.f_Bx + amm*cmm.f_Bx;

            byp =
               acc*ccc.f_By + apc*cpc.f_By + acp*ccp.f_By + app*cpp.f_By +
               apm*cpm.f_By + amp*cmp.f_By + amc*cmc.f_By + acm*ccm.f_By + amm*cmm.f_By;

            bzp =
               acc*ccc.f_Bz + apc*cpc.f_Bz + acp*ccp.f_Bz + app*cpp.f_Bz +
               apm*cpm.f_Bz + amp*cmp.f_Bz + amc*cmc.f_Bz + acm*ccm.f_Bz + amm*cmm.f_Bz;

            exm =
               amcc*mcc.f_Ex + ampc*mpc.f_Ex + amcp*mcp.f_Ex + ampp*mpp.f_Ex +
               ampm*mpm.f_Ex + ammp*mmp.f_Ex + ammc*mmc.f_Ex + amcm*mcm.f_Ex + ammm*mmm.f_Ex;

            eym =
               amcc*mcc.f_Ey + ampc*mpc.f_Ey + amcp*mcp.f_Ey + ampp*mpp.f_Ey +
               ampm*mpm.f_Ey + ammp*mmp.f_Ey + ammc*mmc.f_Ey + amcm*mcm.f_Ey + ammm*mmm.f_Ey;

            ezm =
               amcc*mcc.f_Ez + ampc*mpc.f_Ez + amcp*mcp.f_Ez + ampp*mpp.f_Ez +
               ampm*mpm.f_Ez + ammp*mmp.f_Ez + ammc*mmc.f_Ez + amcm*mcm.f_Ez + ammm*mmm.f_Ez;

            bxm =
               amcc*mcc.f_Bx + ampc*mpc.f_Bx + amcp*mcp.f_Bx + ampp*mpp.f_Bx +
               ampm*mpm.f_Bx + ammp*mmp.f_Bx + ammc*mmc.f_Bx + amcm*mcm.f_Bx + ammm*mmm.f_Bx;

            bym =
               amcc*mcc.f_By + ampc*mpc.f_By + amcp*mcp.f_By + ampp*mpp.f_By +
               ampm*mpm.f_By + ammp*mmp.f_By + ammc*mmc.f_By + amcm*mcm.f_By + ammm*mmm.f_By;

            bzm =
               amcc*mcc.f_Bz + ampc*mpc.f_Bz + amcp*mcp.f_Bz + ampp*mpp.f_Bz +
               ampm*mpm.f_Bz + ammp*mmp.f_Bz + ammc*mmc.f_Bz + amcm*mcm.f_Bz + ammm*mmm.f_Bz;

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

               double dpx = px - pxp;
               double dpy = py - pyp;
               double dpz = pz - pzp;
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

            gamma = sqrt(1. + px*px + py*py + pz*pz);
            double Vxm = px/gamma;
            double Vym = py/gamma;
            double Vzm = pz/gamma;

            Vx = 0.5*(Vxm+Vxp);
            double Vy = 0.5*(Vym+Vyp);
            double Vz = 0.5*(Vzm+Vzp);

            djx = weight*djx0[isort]*Vx;
            djy = weight*djy0[isort]*Vy;
            djz = weight*djz0[isort]*Vz;
            double drho = weight*drho0[isort];

            double xtmp = 1.;
            double ytmp = y;
            double ztmp = z;

            double full = 1.;
            double part_step = 1.;

            int itmp = i;
            int jtmp = j;
            int ktmp = k;

            djx *= 1./(1.-Vx);
            djy *= 1./(1.-Vx);
            djz *= 1./(1.-Vx);
            drho *= 1./(1.-Vx);

            double dy = Vy/(1.-Vx)*hx/hy;
            double dz = Vz/(1.-Vx)*hx/hz;

            double partdx = 0.;
            double step = 1.;
            // --- first half-step

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

               ytmp += dy*part_step;
               ztmp += dz*part_step;
               int j_jump = ytmp;
               int k_jump = ztmp;
               if (ytmp < 0.) {
                  j_jump--;
               };
               if (ztmp < 0.) {
                  k_jump--;
               };
               jtmp = j + j_jump;
               ktmp = k + k_jump;
               if (jtmp < 0) jtmp = -1;
               if (jtmp > l_My-1) jtmp = l_My;
               if (ktmp < 0) ktmp = -1;
               if (ktmp > l_Mz-1) ktmp = l_Mz;
               ytmp -= j_jump;
               ztmp -= k_jump;
               if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) {
                  double checkpoint21 = 0.;
               };

               int ntmp = GetN(itmp,jtmp,ktmp);

               if (fabs(djy) > 0.) {
                  int check = 0;
               };
               double part_djx  = djx*part_step;
               double part_djy  = djy*part_step;
               double part_djz  = djz*part_step;
               double part_drho = drho*part_step;
               DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
                  part_djx, part_djy, part_djz, part_drho);
               /*
               DepositRhoInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
               part_djx, part_djy, part_djz, part_drho);
               */
            }

            if (iFullStep) {
               xtmp = 0.;
               p->SetP(px,py,pz);
               p->SetX(xtmp,ytmp,ztmp);
               long nnew = GetN(iLayer,jtmp,ktmp);
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

//---Mesh::DepositCurrentsInCell ---------------------------------------------->
void Mesh::DepositCurrentsInCell(
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
   Cell &cmc = GetCell(i,j-1,k);
   Cell &ccm = GetCell(i,j,k-1);
   Cell &cmm = GetCell(i,j-1,k-1);
   Cell &cmp = GetCell(i,j-1,k+1);
   Cell &cpm = GetCell(i,j+1,k-1);

   x = 0.;

   double xs = x - 0.5;
   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.;
   double axp = 0.;
   double ayc = 1.-ys*ys;
   double aym = 0.5*(-ys + ys*ys);
   double ayp = 0.5*( ys + ys*ys);
   double azc = 1.-zs*zs;
   double azm = 0.5*(-zs + zs*zs);
   double azp = 0.5*( zs + zs*zs);

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;
   double acpm = axc*ayp*azm;
   double acmp = axc*aym*azp;
   double acmc = axc*aym*azc;
   double accm = axc*ayc*azm;
   double acmm = axc*aym*azm;
   /*
   double apcc = axp*ayc*azc;
   double appc = axp*ayp*azc;
   double apcp = axp*ayc*azp;
   double appp = axp*ayp*azp;
   double appm = axp*ayp*azm;
   double apmp = axp*aym*azp;
   double apmc = axp*aym*azc;
   double apcm = axp*ayc*azm;
   double apmm = axp*aym*azm;
   */
   double weight = fabs(drho);

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

   ccc.f_Dens += drho*ayc*azc;
   ccc.f_DensArray[isort] += weight*accc;

   cmc.f_Dens += drho*aym*azc;
   cmc.f_DensArray[isort] += weight*acmc;

   cpc.f_Dens += drho*ayp*azc;
   cpc.f_DensArray[isort] += weight*acpc;

   ccm.f_Dens += drho*ayc*azm;
   ccm.f_DensArray[isort] += weight*accm;

   ccp.f_Dens += drho*ayc*azp;
   ccp.f_DensArray[isort] += weight*accp;

   cmm.f_Dens += drho*aym*azm;
   cmm.f_DensArray[isort] += weight*acmm;

   cpp.f_Dens += drho*ayp*azp;
   cpp.f_DensArray[isort] += weight*acpp;

   cpm.f_Dens += drho*ayp*azm;
   cpm.f_DensArray[isort] += weight*acpm;

   cmp.f_Dens += drho*aym*azp;
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
   Cell &cmc = GetCell(i,j-1,k);
   Cell &ccm = GetCell(i,j,k-1);
   Cell &cmm = GetCell(i,j-1,k-1);
   Cell &cmp = GetCell(i,j-1,k+1);
   Cell &cpm = GetCell(i,j+1,k-1);

   x = 0.;

   double xs = x - 0.5;
   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.-x;
   double axp = x;
   double ayc = 1.-ys*ys;
   double aym = 0.5*(-ys + ys*ys);
   double ayp = 0.5*( ys + ys*ys);
   double azc = 1.-zs*zs;
   double azm = 0.5*(-zs + zs*zs);
   double azp = 0.5*( zs + zs*zs);

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;
   double acpm = axc*ayp*azm;
   double acmp = axc*aym*azp;
   double acmc = axc*aym*azc;
   double accm = axc*ayc*azm;
   double acmm = axc*aym*azm;

   double apcc = axp*ayc*azc;
   double appc = axp*ayp*azc;
   double apcp = axp*ayc*azp;
   double appp = axp*ayp*azp;
   double appm = axp*ayp*azm;
   double apmp = axp*aym*azp;
   double apmc = axp*aym*azc;
   double apcm = axp*ayc*azm;
   double apmm = axp*aym*azm;

   double weight = fabs(drho);

   ccc.f_Dens += drho*ayc*azc;
   ccc.f_DensArray[isort] += weight*accc;

   cmc.f_Dens += drho*aym*azc;
   cmc.f_DensArray[isort] += weight*acmc;

   cpc.f_Dens += drho*ayp*azc;
   cpc.f_DensArray[isort] += weight*acpc;

   ccm.f_Dens += drho*ayc*azm;
   ccm.f_DensArray[isort] += weight*accm;

   ccp.f_Dens += drho*ayc*azp;
   ccp.f_DensArray[isort] += weight*accp;

   cmm.f_Dens += drho*aym*azm;
   cmm.f_DensArray[isort] += weight*acmm;

   cpp.f_Dens += drho*ayp*azp;
   cpp.f_DensArray[isort] += weight*acpp;

   cpm.f_Dens += drho*ayp*azm;
   cpm.f_DensArray[isort] += weight*acpm;

   cmp.f_Dens += drho*aym*azp;
   cmp.f_DensArray[isort] += weight*acmp;  

}

//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeFields(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      for (int idim=0; idim<FLD_DIM; idim++) {
         GetCell(iLayer,-1,k).f_Fields[idim] = GetCell(iLayer,l_My-1,k).f_Fields[idim];
         GetCell(iLayer,l_My,k).f_Fields[idim] = GetCell(iLayer,0,k).f_Fields[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      for (int idim=0; idim<FLD_DIM; idim++) {
         GetCell(iLayer,j,-1).f_Fields[idim] = GetCell(iLayer,j,l_Mz-1).f_Fields[idim];
         GetCell(iLayer,j,l_Mz).f_Fields[idim] = GetCell(iLayer,j,0).f_Fields[idim];
      }
   }
};

//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeRho(int iLayer) 
{
   int nsorts = domain()->GetNsorts();
   for (int k=-1; k<l_Mz+1; k++) {
      GetCell(iLayer,0,k).f_Dens += GetCell(iLayer,l_My,k).f_Dens;
      GetCell(iLayer,l_My-1,k).f_Dens += GetCell(iLayer,-1,k).f_Dens;
      for (int isort=0; isort<nsorts; isort++) {
         GetCell(iLayer,0,k).f_DensArray[isort] += GetCell(iLayer,l_My,k).f_DensArray[isort];
         GetCell(iLayer,l_My-1,k).f_DensArray[isort] += GetCell(iLayer,-1,k).f_DensArray[isort];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      GetCell(iLayer,j,0).f_Dens += GetCell(iLayer,j,l_Mz).f_Dens;
      GetCell(iLayer,j,l_Mz-1).f_Dens += GetCell(iLayer,j,-1).f_Dens;
      for (int isort=0; isort<nsorts; isort++) {
         GetCell(iLayer,j,0).f_DensArray[isort] += GetCell(iLayer,j,l_Mz).f_DensArray[isort];
         GetCell(iLayer,j,l_Mz-1).f_DensArray[isort] += GetCell(iLayer,j,-1).f_DensArray[isort];
      }
   }
}
//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeCurrents(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      for (int idim=0; idim<3; idim++) {
         GetCell(iLayer,0,k).f_Currents[idim] += GetCell(iLayer,l_My,k).f_Currents[idim];
         GetCell(iLayer,l_My-1,k).f_Currents[idim] += GetCell(iLayer,-1,k).f_Currents[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      for (int idim=0; idim<3; idim++) {
         GetCell(iLayer,j,0).f_Currents[idim] += GetCell(iLayer,j,l_Mz).f_Currents[idim];
         GetCell(iLayer,j,l_Mz-1).f_Currents[idim] += GetCell(iLayer,j,-1).f_Currents[idim];
      }
   }
};