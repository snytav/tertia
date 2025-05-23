#include <math.h>

#include <stdlib.h>
#include "vlpl3d.h"
#include "cell3d.h"
#include "controls.h"
#include "para.h"

double rnd_gaussian (const double sigma);

//---------------------------- Mesh::Mesh --------------------
Mesh::Mesh(long mx, long ofX, long my, long ofY, long mz, long ofZ)
{
   SetSizes(mx,my,mz);
   l_offsetX = ofX; 
   l_offsetY = ofY; 
   l_offsetZ = ofZ; 
   l_MovieStarted = 0;
   l_Processed = 0;
   i_OptimalWakeRecorded = 0;
   f_WakeZeroPosition = f_RecentWakeZeroPosition = 0.;
   f_WakeCorrection = 1.;

   p_CellArray = new Cell[l_sizeXYZ];
   SetCellNumbers();
   p_CellLayerP = new Cell[l_sizeYZ];
   p_CellLayerC = new Cell[l_sizeYZ];
   p_CellLayerM = new Cell[l_sizeYZ];

   f_aJx = new double[l_Mx+l_My+l_Mz+2*l_dMx];
//   pf_FileWakeControl = fopen("WakeControl.data","wt");
}

//---------------------------- Mesh::makeit --------------------
void Mesh::MakeIt(void)
{
   long i = 4;
   long j = l_My/2;
   long k = l_Mz/2;
   long n=0;
   long xUpperParallelLimit;

   if(GetRank() < GetSize()-1)
   {
      xUpperParallelLimit = (GetRank() + 1)*(l_Mx-1) + 1;
   }
   else
   {
      xUpperParallelLimit = (GetRank() + 1)*(l_Mx-1);
   }
   printf("xUpperParallelLimit %d rank %d \n",xUpperParallelLimit,GetRank());
  
   srand(domain()->GetmyPE());
   //  SeedParticles(i,j,k);
   //  return;
   for (k=0; k<l_Mz; k++) {
      for (j=0; j<l_My; j++) { 
         for (i = GetRank()*(l_Mx-1); i <= xUpperParallelLimit; i++) {
	//  for (i = GetRank()*(l_Mx-1); i < (GetRank() + 1)*(l_Mx-1) + 1; i++) {
#ifdef CUDA_WRAP_PARALLEL_DEBUG	   
	    printf("seedbeam call rank %5d %5d %5d %5d  \n",GetRank(),i,j,k);
#endif	    
            SeedBeamParticles(i,j,k);
         }
      }
   }
   for (n=0; n<l_sizeXYZ; n++) {
      for (k=0; k<3; k++) {
         p_CellArray[n].f_Currents[k]=0.;
         if (p_CellArray[n].f_DensArray) {
            delete[] p_CellArray[n].f_DensArray;
            p_CellArray[n].f_DensArray = NULL;
         }
         int nsorts = domain()->GetNsorts();
         if (nsorts>0) {
            p_CellArray[n].f_DensArray = new double[nsorts];
         }
      }
   }
   return;
   // Debugging 
   for (k=0; k<l_Mz; k++) 
      for (j=0; j<l_My; j++) 
         for (i=0; i<l_Mx; i++) {
            n = GetN(i,j,k);
            Particle *p = p_CellArray[n].p_Particles;
            while (p) {
               double x = X(i);
               p->f_Px = .1;
               p->f_Py = 0;
               p->f_Pz = 0;
               p = p->p_Next;
            }
         }
}

//---------------------------- Mesh::SeedFrontParticles --------------------
void Mesh::SeedFrontParticles()
{
   int qq = 0;

   for (int k=0; k< l_Mz; k++) {
      for (int j=0; j<l_My; j++) {
         int i = l_Mx;
         SeedParticles(i,j,k);
      }
   }
}
//---------------------------- Mesh::SeedParticles --------------------
void Mesh::SeedBeamParticles(long i, long j, long k)
{
  double xtmp=0., ytmp=0., ztmp=0.;
  long nseed=0, iseed=0;
  double xco = X(i)+domain()->p_Cntrl->GetShift()*Hx();
  double yco = Y(j) - domain()->GetYlength()/2.;;
  double zco = Z(k) - domain()->GetZlength()/2.;;
  double dens = -1.;
  Particle *p=NULL;
  double ts2hx = domain()->GetTs()/(domain()->GetHx());
  double ts2hy = domain()->GetTs()/(domain()->GetHy());
  double ts2hz = domain()->GetTs()/(domain()->GetHz());

  long n = GetN(i-GetRank()*(l_Mx-1),j,k);
#ifdef CUDA_WRAP_PARALLEL_DEBUG  
  printf("xco %e i %d local i %d rank %d \n",xco,i,i-GetRank()*(l_Mx-1),GetRank());
#endif  
  if (n<0 || n> l_sizeXYZ-1) {
    cout << "We have problems with n="<<n<<
      " i=" << i <<" j=" << j <<" k=" << k << endl;
    exit(-15);
  }
  Cell &c = p_CellArray[n];

  int isort = domain()->GetNsorts();
  while (isort--) {
    Specie *spec = domain()->GetSpecie(isort);
    if (!spec->IsBeam()) continue;
    dens = spec->Density(xco,yco,zco);

    if (dens > 0.) {
      nseed = iseed = spec->GetPperCell();
      if (nseed > 0) {
	double fiside=pow(double(nseed-1),1./3.)+1.;
	long iside = (long)(fiside);
	long iside2 = iside*iside;
	while(iseed--) {
	  if (nseed == 1) {
	    xtmp = 0.5;
	    ytmp = ztmp = 0.5;
	  } else if (nseed == 2 || nseed == 3) {
	    xtmp = (iseed+1.)/(nseed+1.);
	    ytmp = ztmp = 0.5;
	  } else if (nseed == 4) {
	    xtmp = 0.5;
	    iside2 = 2;
	    ytmp = 1./3.*(1.+iseed%iside2);
	    ztmp = 1./3.*(1.+iseed/iside2);
	  } else { 
	    long iz = iseed/iside2;
	    long iy = (iseed-iside2*iz)/iside;
	    long ix = iseed-iside*(iy+iside*iz);
	    xtmp = 1./iside*(ix+.5);
	    ytmp = 1./iside*(iy+.5);
	    ztmp = 1./iside*(iz+.5);
	  }

	  double weight = spec->GetWeight()*dens;
	  double q2m = spec->GetQ2M();
	  double px0 = spec->GetPx();
	  double py0 = spec->GetPy();
	  double pz0 = spec->GetPz();

	  double spreadX = spec->GetPspreadX();
	  double spreadY = spec->GetPspreadY();
	  double spreadZ = spec->GetPspreadZ();

	  double random;
	  random = rnd_gaussian (spreadX);
	  px0 += random;

     if (spec->GetPhaseSpaceFillFlag()) {
        int yDir, zDir;
        if (nseed == 9) {
           if (iseed) {
              double phi = iseed*2.*PI/8.;
               py0 += spreadY*cos(phi);
               pz0 += spreadZ*sin(phi);
           }
        } else if (nseed == 19) {
           if (iseed >0 && iseed <= 6) {
              double phi = iseed*2.*PI/6.;
               py0 += 0.5*spreadY*cos(phi);
               pz0 += 0.5*spreadZ*sin(phi);
           } else if (iseed >6 && iseed < 19) {
              double phi = (iseed-6.5)*2.*PI/12.;
               py0 += 1.1*spreadY*cos(phi);
               pz0 += 1.1*spreadZ*sin(phi);
           }
        } else {
           py0 += spreadY*(yco + ytmp*Hy())/spec->f_RadiusY;
           pz0 += spreadZ*(zco + ztmp*Hz())/spec->f_RadiusZ;
        };
        ytmp = ztmp = 0.5;
        xtmp = 0.9999;
     } else {
        random = rnd_gaussian (spreadY);
        py0 += random;
        random = rnd_gaussian (spreadZ);
        pz0 += random;
     };


	  int state = 0;
	  int type = spec->GetType();
	  double totalJ = 0.;
	  double denstmp = dens;

	  switch (type) {
	  case 3: // dielectric
	    p_CellArray[n].f_Epsilon += dens/nseed;
	    break;
	  case 2: // hybrid
	    totalJ = 0.;
	    for (int istmp=0; istmp<domain()->GetNsorts(); istmp++) {
	      Specie *sptmp = domain()->GetSpecie(istmp);
	      double dns = sptmp->Density(xco,yco,zco);
         if (domain()->GetCntrl()->GetWakeControlFlag()) {
            dens *= f_WakeCorrection;
         };
	      if (sptmp->GetType() < 2) {
		totalJ -= dns*sptmp->GetPolarity();
	      }
	    }
	    if (totalJ !=0.) {
	      //px0 += 0.125*totalJ/dens;
	    };
/*
       denstmp = dens + domain()->GetSpecie(1)->Density(xco,yco,zco);
	    px0 = .125*domain()->GetSpecie(1)->Density(xco,yco,zco)/dens;
       */
	    p_CellArray[n].f_DensH =  p_CellArray[n].f_DeltaDensH = dens/nseed;
	    //p_CellArray[n].f_DensH += totalJ/nseed;
	    p_CellArray[n].f_PxH = px0;
	    p_CellArray[n].f_PyH = py0;
	    p_CellArray[n].f_PzH = pz0;

	    break;
	  case 1:
#ifdef CUDA_WRAP_PARALLEL_DEBUG	    
	    printf("mesh::seedBeam Ion xyz %e %e %e rank %d\n",xtmp,ytmp,ztmp,GetRank());
#endif	    
	    state=spec->GetState0();
	    //	    printf("Mesh::SeedParticles %d\n", isort);
	    //printf("mesh::seedBeam IOn xyz %e %e %e \n",xtmp,ytmp,ztmp);
	    p = new Ion(p_CellArray+n, isort, 0, weight, q2m,
			xtmp, ytmp, ztmp, px0, py0, pz0 );
	    while (state>0) {
	      state--;
	      p->RemoveElectron(p_CellArray+n);
	    };
	    break;
	  case 0:
	  default:
#ifdef CUDA_WRAP_PARALLEL_DEBUG	    
	    printf("mesh::seedBeam Ele xyz %e %e %e rank %d \n",xtmp,ytmp,ztmp,GetRank());
#endif	    
	    p = new Electron(p_CellArray+n, weight, 
			     xtmp, ytmp, ztmp, px0, py0, pz0);
	    p->GetX(xtmp,ytmp,ztmp);
	    p->GetP(px0,py0,pz0);
	    break;
	  }
	}
      }
    } 
  }
  return;
}
/*
//---------------------------- Mesh::SeedBeamParticles --------------------
void Mesh::SeedBeamParticles(long i, long j, long k)
{
   double xtmp=0., ytmp=0., ztmp=0.;
   long nseed=0, iseed=0;
   double xco = X(i)+domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;;
   double zco = Z(k) - domain()->GetZlength()/2.;;
   double dens = -1.;
   Particle *p=NULL;
   double ts2hx = domain()->GetTs()/(domain()->GetHx());
   double ts2hy = domain()->GetTs()/(domain()->GetHy());
   double ts2hz = domain()->GetTs()/(domain()->GetHz());

   long n = GetN(i,j,k);
   if (n<0 || n> l_sizeXYZ-1) {
      cout << "We have problems with n="<<n<<
         " i=" << i <<" j=" << j <<" k=" << k << endl;
      exit(-15);
   }
   Cell &c = p_CellArray[n];

   int isort = domain()->GetNsorts();
   while (isort--) {
      Specie *spec = domain()->GetSpecie(isort);
      if (!spec->IsBeam()) continue;
      dens = spec->Density(xco,yco,zco);
      double djx0, djy0, djz0, drho0;
      spec->GetdJ( djx0, djy0, djz0, drho0 );

      if (dens > 0.) {
         nseed = iseed = spec->GetPperCell();
         if (nseed > 0) {
            double fiside=pow(double(nseed-1),1./3.)+1.;
            long iside = (long)(fiside);
            long iside2 = iside*iside;
            while(iseed--) {
               if (nseed == 1) {
                  xtmp = 0.5;
                  ytmp = ztmp = 0.5;
               } else if (nseed == 2 || nseed == 3) {
                  xtmp = (iseed+1.)/(nseed+1.);
                  ytmp = ztmp = 0.5;
               } else if (nseed == 4) {
                  xtmp = 0.5;
                  iside2 = 2;
                  ytmp = 1./3.*(1.+iseed%iside2);
                  ztmp = 1./3.*(1.+iseed/iside2);
               } else { 
                  long iz = iseed/iside2;
                  long iy = (iseed-iside2*iz)/iside;
                  long ix = iseed-iside*(iy+iside*iz);
                  xtmp = 1./iside*(ix+.5);
                  ytmp = 1./iside*(iy+.5);
                  ztmp = 1./iside*(iz+.5);
               }

               double weight = spec->GetWeight()*dens;
               double q2m = spec->GetQ2M();
               double px0 = spec->GetPx();
               double py0 = spec->GetPy();
               double pz0 = spec->GetPz();

               double spreadX = spec->GetPspreadX();
               double spreadY = spec->GetPspreadY();
               double spreadZ = spec->GetPspreadZ();

               double random;
               random = rnd_gaussian (spreadX);
               px0 += random;

               random = rnd_gaussian (spreadY);
               py0 += random;

               random = rnd_gaussian (spreadZ);
               pz0 += random;

               int state = 0;
               int type = spec->GetType();
               double totalJ = 0.;
               double denstmp = dens;

               switch (type) {
     case 3: // dielectric
        p_CellArray[n].f_Epsilon += dens/nseed;
        break;
     case 2: // hybrid
        totalJ = 0.;
        for (int istmp=0; istmp<domain()->GetNsorts(); istmp++) {
           Specie *sptmp = domain()->GetSpecie(istmp);
           double dns = sptmp->Density(xco,yco,zco);
           if (domain()->GetCntrl()->GetWakeControlFlag()) {
              dens *= f_WakeCorrection;
           };
           if (sptmp->GetType() < 2) {
              totalJ -= dns*sptmp->GetPolarity();
           }
        }
        if (totalJ !=0.) {
           px0 += 0.125*totalJ/dens;
        };
        p_CellArray[n].f_DensH = dens/nseed;
        p_CellArray[n].f_DensH += totalJ/nseed;
        p_CellArray[n].f_PxH = px0;
        p_CellArray[n].f_PyH = py0;
        p_CellArray[n].f_PzH = pz0;

        break;
     case 1:
        state=spec->GetState0();
        //	    printf("Mesh::SeedParticles %d\n", isort);
        p = new Ion(p_CellArray+n, isort, 0, weight, q2m,
           xtmp, ytmp, ztmp, px0, py0, pz0 );
        while (state>0) {
           state--;
           p->RemoveElectron(p_CellArray+n);
        };
        break;
     case 0:
     default:
        p = new Electron(p_CellArray+n, weight, 
           xtmp, ytmp, ztmp, px0, py0, pz0);
        p->GetX(xtmp,ytmp,ztmp);
        p->GetP(px0,py0,pz0);
        break;
               }
            }
         }
      } 
   }
   return;
}
*/

//---------------------------- Mesh::SeedParticles --------------------
void Mesh::SeedParticles(long i, long j, long k)
{
   double xtmp=0., ytmp=0., ztmp=0.;
   long nseed=0, iseed=0;
   double xco = X(i)+domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;;
   double zco = Z(k) - domain()->GetZlength()/2.;;
   double dens = -1.;
   Particle *p=NULL;
   double ts2hx = domain()->GetTs()/(domain()->GetHx());
   double ts2hy = domain()->GetTs()/(domain()->GetHy());
   double ts2hz = domain()->GetTs()/(domain()->GetHz());

   long n = GetN(i,j,k);
   if (n<0 || n> l_sizeXYZ-1) {
      cout << "We have problems with n="<<n<<
         " i=" << i <<" j=" << j <<" k=" << k << endl;
      exit(-15);
   }
   Cell &c = p_CellArray[n];

   int isort = domain()->GetNsorts();
   while (isort--) {
      Specie *spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens = spec->Density(xco,yco,zco);
      double djx0, djy0, djz0, drho0;
      spec->GetdJ( djx0, djy0, djz0, drho0 );

      if (dens > 0.) {
         nseed = iseed = spec->GetPperCell();
         if (nseed > 0) {
            double fiside=pow(double(nseed-1),1./2.)+1.;
            long iside = (long)(fiside);
            long iside2 = iside*iside;
            while(iseed--) {
               if (nseed == 1) {
                  xtmp = 0;
                  ytmp = ztmp = 0.5;
               } else if (nseed == 2 || nseed == 3) {
                  xtmp = (iseed+1.)/(nseed+1.);
                  ytmp = ztmp = 0.5;
               } else if (nseed == 4) {
                  xtmp = 0;
                  iside2 = 2;
                  ytmp = 1./3.*(1.+iseed%iside2);
                  ztmp = 1./3.*(1.+iseed/iside2);
               } else { 
                  long iz = iseed/iside;
                  long iy = iseed-iside*iz;
                  xtmp = 0.;
                  ytmp = 1./iside*(iy+.5);
                  ztmp = 1./iside*(iz+.5);
               }

               double weight = spec->GetWeight()*dens;
               double q2m = spec->GetQ2M();
               double px0 = spec->GetPx();
               double py0 = spec->GetPy();
               double pz0 = spec->GetPz();

               double spreadX = spec->GetPspreadX();
               double spreadY = spec->GetPspreadY();
               double spreadZ = spec->GetPspreadZ();

               double random;
               random = rnd_gaussian (spreadX);
               px0 += random;

               random = rnd_gaussian (spreadY);
               py0 += random;

               random = rnd_gaussian (spreadZ);
               pz0 += random;

               int state = 0;
               int type = spec->GetType();
               double totalJ = 0.;
               double denstmp = dens;

               switch (type) {
     case 3: // dielectric
        p_CellArray[n].f_Epsilon += dens/nseed;
        break;
     case 2: // hybrid
        totalJ = 0.;
        for (int istmp=0; istmp<domain()->GetNsorts(); istmp++) {
           Specie *sptmp = domain()->GetSpecie(istmp);
           double dns = sptmp->Density(xco,yco,zco);
           if (domain()->GetCntrl()->GetWakeControlFlag()) {
              dens *= f_WakeCorrection;
           };
           if (sptmp->GetType() < 2) {
              totalJ -= dns*sptmp->GetPolarity();
           }
        }
        if (totalJ !=0.) {
           px0 += 0.125*totalJ/dens;
        };
        /*
        denstmp = dens + domain()->GetSpecie(1)->Density(xco,yco,zco);
        px0 = .125*domain()->GetSpecie(1)->Density(xco,yco,zco)/dens;
        */
        p_CellArray[n].f_DensH = dens/nseed;
        p_CellArray[n].f_DensH += totalJ/nseed;
        p_CellArray[n].f_PxH = px0;
        p_CellArray[n].f_PyH = py0;
        p_CellArray[n].f_PzH = pz0;

        break;
     case 1:
        state=spec->GetState0();
        //	    printf("Mesh::SeedParticles %d\n", isort);
        p = new Ion(p_CellArray+n, isort, 0, weight, q2m,
           xtmp, ytmp, ztmp, px0, py0, pz0 );
        while (state>0) {
           state--;
           p->RemoveElectron(p_CellArray+n);
        };
        break;
     case 0:
     default:
        q2m = domain()->GetSpecie(0)->GetQ2M();
        p = new Electron(p_CellArray+n, weight, 
           xtmp, ytmp, ztmp, px0, py0, pz0, q2m, 0);
        p->GetX(xtmp,ytmp,ztmp);
        p->GetP(px0,py0,pz0);
        break;
               }
            }
         }
      } 
   }
   return;
}

//---------------------------------------------------------------------------------
double rnd_gaussian (const double sigma)
{
   double x, y, r2;

   do
   {
      /* choose x,y in uniform square (-1,-1) to (+1,+1) */

      x = (2.*rand()-RAND_MAX)/RAND_MAX;
      y = (2.*rand()-RAND_MAX)/RAND_MAX;

      /* see if it is in the unit circle */
      r2 = x * x + y * y;
   }
   while (r2 > 1.0 || r2 == 0);

   /* Box-Muller transform */
   return sigma * y * sqrt (-2.0 * log (r2) / r2);
}

//---------------------------- Mesh::Shift --------------------
void Mesh::Shift(void)
{
   long i, j, k;
   double f[FLD_DIM+CURR_DIM];
   for (i=0; i<FLD_DIM+CURR_DIM; i++) f[i]=0.;

   int what = SPACK_F;
   Send(domain()->GetBndXm(), what);
   Receive(domain()->GetBndXp(),what);

   for (k=-l_dMz; k<l_Mz+l_dMz; k++) 
      for (j=-l_dMy; j<l_My+l_dMy; j++) {
         i = -1;
         long n = GetN(i,j,k);
         long nm = n;
         for (i=0; i<l_Mx+1; i++) {
            nm = n;
            n++;
            p_CellArray[nm] = p_CellArray[n];
         }
      }

      what = SPACK_P+SPACK_F;
      Send(domain()->GetBndXm(), what);
      Receive(domain()->GetBndXp(),what);

      if (domain()->XpEdge()) {
         for (k=0; k<l_Mz; k++) 
            for (j=0; j<l_My; j++) {
               i=l_Mx-1;
               SeedParticles(i,j,k);
            }
      }
      if (domain()->nPE() == 0) {
         //		      cout << " mesh has shifted! \n ";
      }
}
//---------------------------- Mesh::EM_EnergyLayer --------------------
double Mesh::EM_EnergyLayer(int iLayer) 
{
   double energy = 0.;
   double hx = Hx();
   double hy = Hy();
   double hz = Hz();
   double vol = hx*hy*hz;
   for (long k=-l_dMz; k<l_Mz+l_dMz; k++) { 
      for (long j=-l_dMy; j<l_My+l_dMy; j++) { 
         long i = iLayer;
         Cell &c = GetCell(i,j,k);
         energy += vol*c.GetIntensityG();
      }
   }
   return energy;
};


//---------------------------- Mesh::ClearFields --------------------
void Mesh::ClearFields(void) {
   for (long n=0; n<l_sizeXYZ; n++) {
      for (int i=0; i<FLD_DIM; i++) {
         p_CellArray[n].f_Fields[i] = 0.;
      }
   }
}

//---------------------------- Mesh::ClearDensity --------------------
void Mesh::ClearDensity(void) {
   return;
   for (long n=0; n<l_sizeXYZ; n++) {
      p_CellArray[n].f_Dens=0.;
      for (int i=0; i<domain()->GetNsorts(); i++) {
         p_CellArray[n].f_DensArray[i] = 0.;
      }
   }
}

//---------------------------- Mesh::ClearBeamCurrents --------------------
void Mesh::ClearBeamCurrents() {
   for (long k=-l_dMz; k<l_Mz+l_dMz; k++) { 
      for (long j=-l_dMy; j<l_My+l_dMy; j++) { 
         for (long i=-l_dMx; i<l_Mx+l_dMx; i++) { 
            Cell &c = GetCell(i,j,k);
            for (long n=0; n<CURR_DIM; n++) {
               c.f_JBeam[n]=0.;
            }
            for (int i=0; i<domain()->GetNsorts(); i++) {
               c.f_DensArray[i] = 0.;
            }
         }
      }
   }
}

//---------------------------- Mesh::ClearCurrents --------------------
void Mesh::ClearCurrents(int layer) {
   for (long k=-l_dMz; k<l_Mz+l_dMz; k++) { 
      for (long j=-l_dMy; j<l_My+l_dMy; j++) { 
         Cell &c = GetCell(layer,j,k);
         for (long n=0; n<CURR_DIM; n++) {
            c.f_Currents[n]=0.;
         }
      }
   }
}


//---------------------------- Mesh::ClearCurrents --------------------
void Mesh::ClearRho(int layer) {
   for (long k=-l_dMz; k<l_Mz+l_dMz; k++) { 
      for (long j=-l_dMy; j<l_My+l_dMy; j++) { 
         Cell &c = GetCell(layer,j,k);
         c.f_Dens=0.;
         for (int i=0; i<domain()->GetNsorts(); i++) {
            c.f_DensArray[i] = 0.;
         }
      }
   }
}


//---------------------------- Mesh::ClearCurrents --------------------
void Mesh::ClearCurrentsSplit() {
   for (long n=0; n<l_sizeYZ; n++) { 
      Cell &c = p_CellLayerC[n];
      for (long n=0; n<CURR_DIM; n++) {
         c.f_Currents[n]=0.;
      }
   }
}


//---------------------------- Mesh::ClearCurrents --------------------
void Mesh::ClearRhoSplit() {
   for (long n=0; n<l_sizeYZ; n++) { 
      Cell &c = p_CellLayerC[n];
      c.f_Dens=0.;
   }
}


//---------------------------- Mesh::Density --------------------
void Mesh::Density(int isort)
{
   ClearDensity();
   for (long k=0; k<l_Mz; k++) 
      for (long j=0; j<l_My; j++)
      {
         long i = 0;
         Cell3Dm c(this,i,j,k);
         for (i=0; i < l_Mx; i++)
         {
            Particle *p = c.XYZ->p_Particles;
            while (p)
            {
               if (p->GetSort() == 0) {
                  long nionized = p->GetNionized();
               }
               if (isort == ALLSORTS) {
                  c.AddDensity(p, p->GetSpecie()->GetPolarity());
               } else if (isort == p->GetSort()) {
                  c.AddDensity(p, 1.);
               }
               p = p->Next();
            }
            p = c.XYZ->p_BeamParticles;
            while (p)
            {
               if (p->GetSort() == 0) {
                  long nionized = p->GetNionized();
               }
               if (isort == ALLSORTS) {
                  c.AddDensity(p, p->GetSpecie()->GetPolarity());
               } else if (isort == p->GetSort()) {
                  c.AddDensity(p, 1.);
               }
               p = p->Next();
            }
            c.Next();
         }
      }
      domain()->Exchange(SPACK_J);
}

//---------------------------- Mesh::TotalCurrents --------------------
void Mesh::TotalCurrents()
{
   int ic=0;
   long i, j, k;
   for (ic=0; ic<CURR_DIM; ic++) f_Currents[ic]=0.;

   for (k=0; k<l_Mz; k++) 
      for (j=0; j<l_My; j++) {
         for (i=0; i < l_Mx; i++) {
            Cell &c = GetCell(i,j,k);
            for (ic=0; ic<CURR_DIM; ic++) f_Currents[ic]+=c.f_Currents[ic];
            //	if (c.f_Jx != 0.) domain()->out_Flog << i << j << k << 
            //		    " Jz=" << c.f_Jx << " total=" << f_Jx << "\n";
         }
      }
      i = l_Mx/2+1;
      j = l_My/2;
      for (k=-l_dMz; k < l_Mz+l_dMz; k++) {
         Cell &c = GetCell(i,j,k);
         f_aJx[k+l_dMz] = c.f_Jz;
      }
}

//---------------------------- int Mesh::SetCellNumbers; --------------------
void Mesh::SetCellNumbers() {
   long n=0;
   for (n=0; n<l_sizeXYZ; n++) {
      p_CellArray[n].l_N = n;
      Particle *p = p_CellArray[n].p_Particles;
      while (p) {
         p->l_Cell = n;
         p = p->p_Next;
      }
      p = p_CellArray[n].p_BeamParticles;
      while (p) {
         if (n == 74435) {
            double dummy = 1.;
         }
         p->l_Cell = n;
         p = p->p_Next;
      }
   }
}


//---------------------------- int Mesh::GetI_from_CellNumber(long n); --------------------
int Mesh::GetI_from_CellNumber(long n) {
   while (n >= l_sizeX) {
      n -= l_sizeX;
   }
   n -= l_dMx;
   return n;
}

//---------------------------- int Mesh::GetJ_from_CellNumber(long n); --------------------
int Mesh::GetJ_from_CellNumber(long n) {
   n /= l_sizeX;
   while (n >= l_sizeY) {
      n -= l_sizeY;
   }
   n -= l_dMy;
   return n;
}

//---------------------------- int Mesh::GetK_from_CellNumber(long n); --------------------
int Mesh::GetK_from_CellNumber(long n) {
   n /= l_sizeXY;
   n -= l_dMz;
   return n;
}

//---------------------------- Mesh::SaveCadrMovie --------------------
void Mesh::SaveCadrMovie(FILE* fout)
{
   if(!MovieWriteEnabled())
      return;

   long istat=-1;
   GETTYPE getf;
   long i_Hfig = domain()->i_NMovieFrames;
   long i_Vfig = domain()->i_NMovieFrames;
   long i_Nfig = i_Vfig + i_Hfig;
   long ifig = 0;

   double hxw = domain()->GetHx();
   double hyw = domain()->GetHy();
   double hzw = domain()->GetHz();

   if (l_MovieStarted==0)
   {
      l_MovieStarted++;
      fwrite(&l_Mx,sizeof(long),1,fout);
      fwrite(&l_My,sizeof(long),1,fout);
      fwrite(&l_Mz,sizeof(long),1,fout);

      fwrite(&i_Vfig,sizeof(long),1,fout);
      fwrite(&i_Hfig,sizeof(long),1,fout);
      fwrite(&i_Nfig,sizeof(long),1,fout);
   }

   fwrite(&istat,sizeof(long),1,fout);

   double phase=domain()->GetPhase();
   fwrite(&phase,sizeof(double),1,fout);
   double shift = domain()->p_Cntrl->GetShift()*hxw;
   double ShiftPeriod = 0;
   long ShiftPad=0;
   long ShiftN=domain()->p_Cntrl->GetShift();
   fwrite(&shift,sizeof(double),1,fout);
   fwrite(&ShiftPeriod,sizeof(double),1,fout);
   fwrite(&ShiftPad,sizeof(long),1,fout);
   fwrite(&ShiftN,sizeof(long),1,fout);

   for (ifig=0; ifig<i_Hfig; ifig++)
   {
      getf = domain()->p_MovieFrame->Gets[ifig];
      fwrite(&hxw,sizeof(double),1,fout);
      fwrite(&hyw,sizeof(double),1,fout);
      fwrite(&l_Mx,sizeof(long),1,fout);
      fwrite(&l_My,sizeof(long),1,fout);
      long k = l_Mz/2;

      for (long j=0; j<l_My; j++)
         for (long i=0; i<l_Mx; i++)
         {
            Cell &c = GetCell(i,j,k);
            double dum = (c.*getf)();
            fwrite(&dum,sizeof(double),1,fout);
         }
   }

   for (ifig=0; ifig<i_Vfig; ifig++)
   {
      getf = domain()->p_MovieFrame->Gets[ifig];
      fwrite(&hxw,sizeof(double),1,fout);
      fwrite(&hzw,sizeof(double),1,fout);
      fwrite(&l_Mx,sizeof(long),1,fout);
      fwrite(&l_Mz,sizeof(long),1,fout);
      long j = l_My/2;
      for (long k=0; k<l_Mz; k++)
         for (long i=0; i<l_Mx; i++)
         {
            Cell &c = GetCell(i,j,k);
            double dum = (c.*getf)();
            fwrite(&dum,sizeof(double),1,fout);
         }
   }
}

//---------------------------- Mesh::MovieWriteEnabled --------------------
bool Mesh::MovieWriteEnabled(void)
{

   // Check for Postprocessor movie file
   // if it is Postprocessor movie file is not allowed
   //
   if (domain()->p_Cntrl->PostProcessing())
   {
      return false;
   };

   //  int Xpart = domain()->p_MPP->GetXpart();
   int Ypart = domain()->p_MPP->GetYpart();
   int Zpart = domain()->p_MPP->GetZpart();

   //  int iPE = domain()->p_MPP->GetiPE();
   int jPE = domain()->p_MPP->GetjPE();
   int kPE = domain()->p_MPP->GetkPE();

   int odd_k = Zpart % 2;
   int odd_j = Ypart % 2;
   //  double my_i = fmod(Xpart, 2.0);

   int k_p = 1;
   int j_p = 1;
   //  int i_p = 1;

   //long k = 0;

   if(!odd_k)
   {
      // even number
      k_p = (Zpart / 2) - 1;
      //     k     = l_Mz-1;
   }
   else
   {
      k_p = (Zpart - 1)/2;
      //k     = l_Mz/2;
   };

   if(!odd_j)
   {
      // even number
      j_p = (Ypart / 2) - 1;
      //j = l_My-1;
   }
   else

   {
      j_p = (Ypart - 1)/2;
      //j = l_My/2;
   };

   if(kPE != k_p && jPE != j_p)
      return false;

   return true;
}

//---------------------------- Mesh::ExchangeEndian --------------------
template <class C> inline void Mesh::ExchangeEndian(C& value) const
{
   char *d = reinterpret_cast<char*> (&value);
   char *dout = new char[sizeof(value)];
   for(size_t i = 0; i < sizeof(value); i++)
      dout[i] = d[ sizeof(value) - 1 - i ];
   memcpy(&value, dout, sizeof(value));
   delete[] dout;
}
//---------------------------- Mesh::SaveCadrMovie2 --------------------
int Mesh::SaveCadrMovie2(FILE* fout)
{
   int isort = ALLSORTS;
//   Density(isort);  //seting up density for //Cell::GetDens() for partcles of sort ALLSORTS
   long istat=-1;
   GETTYPE getf;
   long i_Hfig = domain()->i_NMovieFrames;
   long i_Vfig = domain()->i_NMovieFrames;
   long i_Nfig = i_Vfig + i_Hfig;
   long ifig = 0;

   float hxw = domain()->GetHx();
   float hyw = domain()->GetHy();
   float hzw = domain()->GetHz();

   //  int Xpart = domain()->p_MPP->GetXpart();
   int Ypart = domain()->p_MPP->GetYpart();
   int Zpart = domain()->p_MPP->GetZpart();

   //  int iPE = domain()->p_MPP->GetiPE();
   int jPE = domain()->p_MPP->GetjPE();
   int kPE = domain()->p_MPP->GetkPE();

   int odd_k = Zpart % 2;
   int odd_j = Ypart % 2;

   int k_p = 1;
   int j_p = 1;
   //  int i_p = 1;

   long k = 0;

   if(!odd_k)
   {
      // even number
      k_p = (Zpart / 2) - 1;
      k     = l_Mz-1;
   }
   else
   {
      k_p = (Zpart - 1)/2;
      k     = l_Mz/2;
   }

   long j = 0;

   if(!odd_j)
   {
      // even number
      j_p = (Ypart / 2) - 1;
      j = l_My-1;
   }
   else

   {
      j_p = (Ypart - 1)/2;
      j = l_My/2;
   }

   if(kPE != k_p && jPE != j_p)
      return 3;

   /*
   if(iPE != i_p)
   break;
   */

   if (l_MovieStarted==0) // Attention p_Cntrl->Reload() will be clean in next upper step, see int Domain::Check(void) 
   {
      l_MovieStarted++;

#ifdef MOVIE_CONVERT
      ExchangeEndian(l_Mx);
      ExchangeEndian(l_My);
      ExchangeEndian(l_Mz);

      ExchangeEndian(i_Vfig);
      ExchangeEndian(i_Hfig);
      ExchangeEndian(i_Nfig);
#endif      

      fwrite(&l_Mx,sizeof(long),1,fout);
      fwrite(&l_My,sizeof(long),1,fout);
      fwrite(&l_Mz,sizeof(long),1,fout);

      fwrite(&i_Vfig,sizeof(long),1,fout);
      fwrite(&i_Hfig,sizeof(long),1,fout);
      fwrite(&i_Nfig,sizeof(long),1,fout);

#ifdef MOVIE_CONVERT
      ExchangeEndian(l_Mx);
      ExchangeEndian(l_My);
      ExchangeEndian(l_Mz);

      ExchangeEndian(i_Vfig);
      ExchangeEndian(i_Hfig);
      ExchangeEndian(i_Nfig);
#endif            
   }

#ifdef MOVIE_CONVERT
   ExchangeEndian(istat);
#endif 

   fwrite(&istat,sizeof(long),1,fout);

   float phase=domain()->GetPhase();

#ifdef MOVIE_CONVERT
   ExchangeEndian(phase);
#endif   

   fwrite(&phase,sizeof(float),1,fout);
   float shift = domain()->p_Cntrl->GetShift()*hxw;
   float ShiftPeriod = domain()->p_Cntrl->GetShiftPeriod();
   long ShiftPad = domain()->p_Cntrl->GetShiftPad();
   long ShiftN=domain()->p_Cntrl->GetShift();

#ifdef MOVIE_CONVERT
   ExchangeEndian(shift);
   ExchangeEndian(ShiftPeriod);
   ExchangeEndian(ShiftPad);
   ExchangeEndian(ShiftN);
#endif    

   fwrite(&shift,sizeof(float),1,fout);
   fwrite(&ShiftPeriod,sizeof(float),1,fout);
   fwrite(&ShiftPad,sizeof(long),1,fout);
   fwrite(&ShiftN,sizeof(long),1,fout);

   if(kPE == k_p)
      for (ifig=0; ifig<i_Hfig; ifig++)
      {
         getf = domain()->p_MovieFrame->Gets[ifig];

#ifdef MOVIE_CONVERT
         ExchangeEndian(hxw);
         ExchangeEndian(hyw);
         ExchangeEndian(l_Mx);
         ExchangeEndian(l_My);
#endif

         fwrite(&hxw,sizeof(float),1,fout);
         fwrite(&hyw,sizeof(float),1,fout);
         fwrite(&l_Mx,sizeof(long),1,fout);
         fwrite(&l_My,sizeof(long),1,fout);

#ifdef MOVIE_CONVERT
         ExchangeEndian(hxw);
         ExchangeEndian(hyw);
         ExchangeEndian(l_Mx);
         ExchangeEndian(l_My);
#endif

         for (long j=0; j<l_My; j++)
            for (long i=0; i<l_Mx; i++)
            {
               Cell &c = GetCell(i,j,k);
               float dum = (c.*getf)();
#ifdef MOVIE_CONVERT
               ExchangeEndian(dum);
#endif
               fwrite(&dum,sizeof(float),1,fout);
            }
      }

      if(jPE == j_p)
      {
         for (ifig=0; ifig<i_Vfig; ifig++)
         {
            getf = domain()->p_MovieFrame->Gets[ifig];

#ifdef MOVIE_CONVERT
            ExchangeEndian(hxw);
            ExchangeEndian(hzw);
            ExchangeEndian(l_Mx);
            ExchangeEndian(l_Mz);
#endif            

            fwrite(&hxw,sizeof(float),1,fout);
            fwrite(&hzw,sizeof(float),1,fout);
            fwrite(&l_Mx,sizeof(long),1,fout);
            fwrite(&l_Mz,sizeof(long),1,fout);

#ifdef MOVIE_CONVERT
            ExchangeEndian(hxw);
            ExchangeEndian(hzw);
            ExchangeEndian(l_Mx);
            ExchangeEndian(l_Mz);
#endif 

            for (long k=0; k<l_Mz; k++)
               for (long i=0; i<l_Mx; i++)
               {
                  Cell &c = GetCell(i,j,k);
                  float dum = (c.*getf)();
#ifdef MOVIE_CONVERT
                  ExchangeEndian(dum);
#endif                 
                  fwrite(&dum,sizeof(float),1,fout);
               }
         }
      }
      return 1;
}

