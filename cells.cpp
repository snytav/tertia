
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vlpl3d.h"

//---------------------- Class Cell --------------------------------
double Cell::sf_DimFields = double(1.);
double Cell::sf_DimCurr = double(1.);
double Cell::sf_DimDens = double(1.);
int Cell::si_Sort = 0;

Cell::Cell()
{
   f_Ex = f_Ey = f_Ez = f_Bx = f_By = f_Bz = f_Jx = f_Jy = f_Jz = f_Dens = NIL;
   f_JxBeam = f_JyBeam = f_JzBeam = f_RhoBeam = NIL;
   p_Particles = NULL; p_BeamParticles = NULL; p_Hook = NULL; p_BeamHook = NULL;
   f_DensArray = NULL;
   f_PxH = f_PyH = f_PzH = f_Dens = f_DensH = f_DeltaDensH = NIL;
   f_Epsilon = 1.;
};


void Cell::operator=(Cell &ctmp)
{
   int i=0;
   for (i=0; i<FLD_DIM; i++) {
      f_Fields[i] = ctmp.f_Fields[i];
   }
   for (i=0; i<CURR_DIM; i++) {
      f_Currents[i] = ctmp.f_Currents[i];
   }
   f_Epsilon = ctmp.f_Epsilon;
   f_DensH = ctmp.f_DensH;
   f_DeltaDensH = ctmp.f_DeltaDensH;
   f_PxH = ctmp.f_PxH;
   f_PyH = ctmp.f_PyH;
   f_PzH = ctmp.f_PzH;
   p_Particles = ctmp.p_Particles;
   ctmp.p_Particles = NULL;
   Particle *p = p_Particles;
   while (p) {
      p->l_Cell = l_N;
      p = p->p_Next;
   }
}

//---Cell::StepXi--------------------------------------------->
void Cell::RelayParticles()
{
   Cell* nextCell = this + 1;
   Cell &ctmp = *nextCell;

   p_Particles = ctmp.p_Particles;
   ctmp.p_Particles = NULL;
   Particle *p = p_Particles;
   while (p) {
      p->l_Cell = l_N;
      p = p->p_Next;
   }
};

//---Cell::GetDens--------------------------------------------->
double Cell::GetDens(int isort)
{
   if (isort>=0 && isort < domain()->GetNsorts()) {
      return f_DensArray[isort];
   }
   return f_Dens;
};
//---Cell::GetDensG--------------------------------------------->
double Cell::GetDensG(void)
{
   return f_Dens;
}

//---Cell::AddParticle--------------------------------------------->
void Cell::AddParticle(Particle* p) {
   p->l_Cell = l_N;
   int isort = p->GetSort();
   Specie* spec = p->GetSpecie();
   if (p->IsBeam()) {
      p->p_Next = p_BeamParticles;
      p_BeamParticles=p;
   } else {
      p->p_Next = p_Particles;
      p_Particles=p;
   }
};

//---Cell::AddBeamParticle--------------------------------------------->
void Cell::AddBeamParticle(Particle* p) {
   AddParticle(p);
};

//---Cell::RemoveParticle--------------------------------------------->
Particle* Cell::RemoveParticle(Particle* p, Particle* prev) {
   if (prev) prev->p_Next=p->p_Next;
   else  {
      if (p->IsBeam()) {
         p_BeamParticles = p->p_Next;
      } else {
         p_Particles = p->p_Next;
      }
   }
   prev = p->p_Next;

   // ParticleSergK
   //  p->p_Next = NULL;
   // ParticleSergK
   return prev;  // pointer to the next particle is returned (not previous!!)
};

//---Cell::RemoveBeamParticle--------------------------------------------->
Particle* Cell::RemoveBeamParticle(Particle* p, Particle* prev) {
   return RemoveParticle(p, prev);
};

//---Cell::KillParticle--------------------------------------------->
Particle* Cell::KillParticle(Particle* p, Particle* prev) {
   prev = RemoveParticle(p, prev);
   delete p;
   return prev;  // pointer to the next particle is returned (not previous!!)
};

//---Cell::KillBeamParticle--------------------------------------------->
Particle* Cell::KillBeamParticle(Particle* p, Particle* prev) {
   prev = RemoveBeamParticle(p, prev);
   delete p;
   return prev;  // pointer to the next particle is returned (not previous!!)
};

//---Cell::PCount--------------------------------------------->
long Cell::PCount(int sort)
{
   int tmp = 0;
   Particle *ptmp = p_Particles;
   while(ptmp) {
      if (sort==ALLSORTS || ptmp->GetSort()==sort) tmp++;
      ptmp=ptmp->p_Next;
   };
   return tmp;
}

//---Cell::--------------------------------------------->
void Cell::PackF(CBuffer *j, int cond, int what)
{
   double nil = 0.;
   int i = 0;
   int isort = 0;
   if (what & SPACK_J) {
      //  cout << "Packing j \n";
      if (cond)  {
         for (i=0; i<CURR_DIM; i++) *j << nil;
         for (isort=0; isort < domain()->GetNsorts(); isort++) {
            *j << nil;
         }
      } else {
         for (i=0; i<CURR_DIM; i++) {
            *j << f_Currents[i];
            f_Currents[i]=0.;
         };
         for (isort=0; isort < domain()->GetNsorts(); isort++) {
            *j << f_DensArray[isort];
         }
      }
   };
   if (what & SPACK_JB) {
      //  cout << "Packing j \n";
      if (cond)  {
         for (i=0; i<CURR_DIM; i++) *j << nil;
      } else {
         for (i=0; i<CURR_DIM; i++) {
            *j << f_JBeam[i];
            f_JBeam[i]=0.;
         };
      }
   };
   if (what & SPACK_E) {
      if (cond)  {
         for (i=0; i<E_DIM; i++) *j << nil;
         for (i=0; i<4; i++) *j << nil;
         //       for (i=0; i<CURR_DIM; i++) *j << nil;
         //       for (int i=0; i<H_DIM; i++) *j << nil;
      } else { 
         for (i=0; i<E_DIM; i++) *j << f_Fields[i];
//         *j << vc_A;
//         *j << vc_A1;
         //       for (i=0; i<CURR_DIM; i++) *j << f_Currents[i];
         //       for (int i=0; i<H_DIM; i++) *j << f_hData[i];
      }
   }
   if (what & SPACK_B) {
      if (cond)  for (i=E_DIM; i<FLD_DIM; i++) *j << nil;
      else for (i=E_DIM; i<FLD_DIM; i++) *j << f_Fields[i];
   }
   if (what & SPACK_H) {
      if (cond)  {
         for (i=0; i<H_DIM; i++) *j << nil;
         *j << nil << nil << nil;
      }
      else {
         for (i=0; i<H_DIM; i++) *j << f_hData[i];
         *j << f_Jx << f_Jy << f_Jz;
      }
   }
}

//---Cell::--------------------------------------------->
void Cell::UnPackF(CBuffer *j, int cond, int what)
{
   //  return;
   double cur=0.;
   if (what & SPACK_J) {
      //  cout << "UnPacking j "<<what<<\n";
      for (int i=0; i<CURR_DIM; i++) {*j >> cur; f_Currents[i]+=cur;}
      for (int isort=0; isort < domain()->GetNsorts(); isort++) {
         *j >> cur; f_DensArray[isort] += cur;
      };
   };
   if (what & SPACK_JB) {
      //  cout << "UnPacking j "<<what<<\n";
      for (int i=0; i<CURR_DIM; i++) {*j >> cur; f_JBeam[i]+=cur;}
   };
   if (what & SPACK_E) {
      for (int i=0; i<E_DIM; i++) *j >> f_Fields[i];
//      *j >> vc_A;
//      *j >> vc_A1;
      //     for (int i=0; i<CURR_DIM; i++) *j >> f_Currents[i];
      //     for (int i=0; i<H_DIM; i++) *j >> f_hData[i];
   }
   if (what & SPACK_B) {
      for (int i=E_DIM; i<FLD_DIM; i++) *j >> f_Fields[i];
   }
   if (what & SPACK_H) {
      for (int i=0; i<H_DIM; i++) *j >>  f_hData[i];
      *j >> f_Jx >> f_Jy >> f_Jz;
   }
}

//---Cell::--------------------------------------------->
long Cell::PackP(Boundary *bnd, CBuffer *j, 
                 double x, double y, double z, int cond, int sort)
{
   double hx = domain()->GetHx();
   double hy = domain()->GetHy();
   double hz = domain()->GetHz();
   // x,y,z are the absolute cell center positions
   //  return;
   int fin;
   Particle *p, *ptmp;
   long pcnt = 0;

   switch (cond) {
  case NIL:
     ptmp = p_Particles;
     while (p=ptmp) {
        p->l_Cell = l_N;
        int psort = p->GetSort();
        if (psort != 0) {
           //	printf("A particle of psort %d is found \n", psort);
        }
        *j << psort;
        //		  p->AddX(x,y,z);
        p->Pack(j);
        ptmp = p->p_Next;
        KillParticle(p, NULL);
        pcnt++;
     }
     return pcnt;
  case PKILL:
     ptmp = p_Particles;
     while (p=ptmp) {
        ptmp = p->p_Next;
        p->f_X *= hx;
        p->f_X += x + domain()->GetCntrl()->GetShift()*hx;
        p->f_Y *= hy;
        p->f_Y += y;
        p->f_Z *= hz;
        p->f_Z += z;
        bnd->AddZombie(p,this);
        pcnt++;
     }
     return pcnt;
  case FINISH:
     fin = FINISH;
     *j << fin;
     return pcnt;
  default:
     return pcnt;
   }
   return pcnt;
}
//---Cell::--------------------------------------------->
long Cell::PackPB(Boundary *bnd, CBuffer *j, 
                  double x, double y, double z, int cond, int sort)
{
   double hx = domain()->GetHx();
   double hy = domain()->GetHy();
   double hz = domain()->GetHz();
   // x,y,z are the absolute cell center positions
   //  return;
   int fin;
   Particle *p, *ptmp;
   long pcnt = 0;

   switch (cond) {
  case NIL:
     ptmp = p_BeamParticles;
     while (p=ptmp) {
        p->l_Cell = l_N;
        int psort = p->GetSort();
        if (psort != 0) {
           //	printf("A particle of psort %d is found \n", psort);
        }
        *j << psort;
        //		  p->AddX(x,y,z);
        p->Pack(j);
        ptmp = p->p_Next;
        KillBeamParticle(p, NULL);
        pcnt++;
     }
     return pcnt;
  case PKILL:
     ptmp = p_BeamParticles;
     while (p=ptmp) {
        ptmp = p->p_Next;
        p->f_X *= hx;
        p->f_X += x + domain()->GetCntrl()->GetShift()*hx;
        p->f_Y *= hy;
        p->f_Y += y;
        p->f_Z *= hz;
        p->f_Z += z;
        bnd->AddZombie(p,this);
        pcnt++;
     }
     return pcnt;
  case FINISH:
     fin = FINISH;
     *j << fin;
     return pcnt;
  default:
     return pcnt;
   }
   return pcnt;
}

//---Cell::--------------------------------------------->
long Cell::SaveP(FILE* pFile, double x, double y, double z, int sort)
{
   // x,y,z are the absolute cell center positions
   //  return;
   Particle *p, *ptmp;
   long pcnt = 0;
   long ldump = 0;

   ptmp = p_Particles;
   while (p=ptmp) {
      int psort = p->GetSort();
      if (psort==sort || psort==ALLSORTS) {
         //      ldump += fwrite(&psort,1,sizeof(int),pFile);
         p->AddX(x,y,z);
         ldump += p->Save(pFile);
         p->ExtractX(x,y,z);
      }
      ptmp = p->p_Next;
      pcnt++;
   }
   return ldump;
}


//---Cell::--------------------------------------------->
void Cell::Load(CBuffer *j)
{
   return;
}

//---Cell::--------------------------------------------->
long Cell::PKill(int sort)
{
   Particle *p = p_Particles;
   Particle *prev = NULL;
   int tmp = 0;
   while(p) {
      if (sort==ALLSORTS || p->GetSort()==sort) {
         p = KillParticle(p, prev);
         tmp++;
      }
      else {
         prev = p;
         p=p->p_Next;
      }
   }
   return tmp;
}

//---Cell::--------------------------------------------->
//double Cell::x(void) {return GetMesh()->X(i_myI);}
//double Cell::y(void) {return GetMesh()->Y(i_myJ);}
//double Cell::z(void) {return GetMesh()->Z(i_myK);}


//---Cell::--------------------------------------------->



//---Cell::--------------------------------------------->
/*
void Cell::GetDistribution(int sort)
{
Particle *ptmp = p_Particles;
double tmp = 0.;
double one = double(1.);
int j = 0;
for(double i = 0; i < 0.0003; i += 0.000003)
{
ptmp = p_Particles;
while(ptmp)
{
if (sort==-1 || ptmp->GetSort()==sort)
{
double gamma = ptmp->FindP2();
double energy = (gamma)/(one + sqrt(one + gamma));
energy = energy/ptmp->GetQ2m();
double MeV = energy / double(2.);
if(MeV >= i && MeV < i + 0.000003)
Distribution[(int)fmod(i,0.000003)]++;
}
ptmp = ptmp->p_Next;
}
}
}
*/

//---Cell::--------------------------------------------->
double Cell::GetTemperature(int sort)
{
   Particle *ptmp = p_Particles;
   double tmp = 0.;
   double one = double(1.);
   while(ptmp) {
      if (sort==-1 || ptmp->GetSort()==sort) {
         double gamma = ptmp->FindP2();
         double energy = (gamma)/(one+sqrt(one+gamma));
         tmp+=fabs(ptmp->f_Weight*energy/ptmp->GetQ2m());
      }
      ptmp = ptmp->p_Next;
   }
   return tmp;
}

//---Cell::~Cell--------------------------------------------->
Cell::~Cell()
{
   if (f_DensArray) {
      delete[] f_DensArray;
   }
}
