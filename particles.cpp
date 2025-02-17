
#include <stdlib.h>
#include "vlpl3d.h"

static long nIonized = 0;

//---------------------- Class Particle --------------------------------
Particle::Particle(Cell *c, double weights, 
		   double xs, double ys, double zs, 
		   double pxs, double pys, double pzs, double q2ms, int isort)
{
   i_Sort = isort;
	l_Cell = 0; 
	if (c) c->AddParticle(this); 
  else p_Next = 0;

  SetX(xs,ys,zs);
  SetP(pxs,pys,pzs);
  f_Weight = weights;
  f_Q2m = q2ms;
}
void Particle::operator=(SavedParticle &p)
  {
	  l_Cell = p.l_Cell;
	  f_X = p.f_X;
	  f_Y = p.f_Y;
	  f_Z = p.f_Z;
	  f_Px = p.f_Px;
	  f_Py = p.f_Py;
	  f_Pz = p.f_Pz;
	  f_Weight = p.f_Weight;
	  f_Q2m = p.f_Q2m;
  }
//---Particle::----------------------------------->
long Particle::GetNionized() {
   return nIonized;
};


//---Particle::----------------------------------->
void Particle::Pack(CBuffer *j)
{
  *j << f_X << f_Y << f_Z << f_Px << f_Py << f_Pz << f_Weight << f_Q2m << l_Cell;
}

//---Particle::----------------------------------->
void Particle::UnPack(CBuffer *j)
{
  *j >> f_X >> f_Y >> f_Z >> f_Px >> f_Py >> f_Pz >> f_Weight >> f_Q2m >> l_Cell;
}

//---Particle::----------------------------------->
long Particle::Save(FILE* pFile)
{
  long ldump = 0;
  ldump += fwrite(f_Momenta,1,sizeof(double)*PART_DIM,pFile);
  return ldump;
}

//---Particle::----------------------------------->
long Particle::Load(FILE* pFile)
{
  long ldump = 0;
  ldump += fread(f_Momenta,1,sizeof(double)*PART_DIM,pFile);
  return ldump;
}
//-----Particle::ScatteringProbability---------------------------------------->
double Particle::GetScatteringProbability(double P, double IonDensityInCell)
{
   double Probability;
   P = abs(P);

   if(P)
   {
      Specie* pSpecie = GetSpecie();

     //scattering probability
     double LaserWaveLengthCGS = domain()->GetCGS()->GetWavelength();
     double TsNum = domain()->GetTs();//numerical timestep in domain

     double XLength = TsNum*LaserWaveLengthCGS*(P/sqrt(1. + P*P));//path traversed in 1 timestep
     double IonDensityInCellCGS = IonDensityInCell*domain()->GetCGS()->GetCritDensity();

     double FreepathLength = 1./(IonDensityInCellCGS*pSpecie->GetScatteringCrossSection(P));//

     Probability = XLength/FreepathLength;
   
     if (Probability >1.) Probability = 1.;
   }
   else {
      Probability = 1.;
   };
    
   if (Probability > 0.)
      return  Probability;
   else return 0.;
}                                                              
 
//---------------------- Class Electron --------------------------------
Electron::Electron(Cell *c,  double weights, 
		   double xs, double ys, double zs, 
		   double pxs, double pys, double pzs,
		   double q2ms, int isort)
  : Particle(c, weights, xs, ys, zs, pxs, pys, pzs, q2ms, isort)
{
  domain()->Add2Specie(isort);
}

//---Electron::----------------------------------->
Electron::~Electron(){int sort=0; domain()->RemoveFromSpecie(sort);};

//---Electron::----------------------------------->
void Electron::Collide(Cell &c) { };

//---------------------- Class Ion --------------------------------
Ion::Ion(Cell *c, int sorts, int state, double weights, double q2ms,
	 double xs, double ys, double zs,  
	 double pxs, double pys, double pzs)
  : Particle(c, weights, xs, ys, zs, pxs, pys, pzs, q2ms, sorts)
{
  i_Z = state;
  i_Sort = sorts;
//	
  #ifdef _DEBUG
	printf("Ion::Ion sorts = %d\n", sorts);
	#endif
//
	
  domain()->Add2Specie(sorts);
}

//---Ion::----------------------------------->
Ion::~Ion(){domain()->RemoveFromSpecie(i_Sort);};

//---Ion::----------------------------------->
double Ion::GetQ2m(void) 
{ return f_Q2m;}

//---Ion::----------------------------------->
void Ion::Pack(CBuffer *j)
{
  Particle::Pack(j);
  *j << i_Z;
}

//---Particle::----------------------------------->
long Ion::Save(FILE* pFile)
{
  long ldump = 0;
  ldump += Particle::Save(pFile);
  ldump += fwrite(&i_Z,1,sizeof(int),pFile);
  return ldump;
}

//---Particle::----------------------------------->
long Ion::Load(FILE* pFile)
{
  long ldump = 0;
  ldump += Particle::Load(pFile);
  ldump += fread(&i_Z,1,sizeof(int),pFile);
  return ldump;
}

//---Ion::----------------------------------->
void Ion::UnPack(CBuffer *j)
{
  Particle::UnPack(j);
  *j >> i_Z;
}
void Ion::operator=(SavedIon &p)
  {
	  l_Cell = p.l_Cell;
	  f_X = p.f_X;
	  f_Y = p.f_Y;
	  f_Z = p.f_Z;
	  f_Px = p.f_Px;
	  f_Py = p.f_Py;
	  f_Pz = p.f_Pz;
	  f_Weight = p.f_Weight;
	  f_Q2m = p.f_Q2m;
	  i_Z = p.i_Z;
	  i_Sort = p.i_Sort;
  }

//---Ion::----------------------------------->
void Ion::Collide(Cell &c) { };

//---Ion::RemoveElectron----------------------------------->
void Ion::RemoveElectron(Cell *c)
{
   IonSpecie* pIonSpecie = (IonSpecie*)GetSpecie();
   if (i_Z >= pIonSpecie->GetZmax()) {
      return;
   }
    // increase ions charge
   // create new electron
   if (pIonSpecie->GetAtomType() > 0) {
      Electron* pNewElectron = new Electron(c, f_Weight, f_X, f_Y, f_Z, f_Px, f_Py, f_Pz);
      nIonized++;
   };
   i_Z++;
}
 
//--------------Ion::Ionize--------------------------------------------
long Ion::Ionize(Cell* c, double E)
{
   if (E!=0.)// check the field
   {
      IonSpecie* pIonSpecie = (IonSpecie*)GetSpecie();
      assert(pIonSpecie);

      double probability = pIonSpecie->GetIonizationProbability(E, i_Z);
      if (probability <= 0.) return nIonized;

      double rnd = double(rand())/RAND_MAX;

      if (rnd < probability){
         RemoveElectron(c);
      }
#ifdef _SAVE_ION_CHARGE
      domain()->LogIonization<<i_Z<<" ";
#endif//_SAVE_ION_CHARGE
   }
   return nIonized;
} 

void Ion::Recombine(Cell &c) { };
