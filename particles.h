#ifndef H_PARTICLES
#define H_PARTICLES

#include "vlpl3dclass.h"
#include "vlpl3d.h"

#define PART_DIM 8
class SavedParticle;
class SavedIon;

//---------------------------- Particle class -----------------------
class Particle {
public:
	int l_Cell;
  union{ double f_Momenta[1];
    double f_X;};
  double f_Y, f_Z, f_Px, f_Py, f_Pz, f_Weight, f_Q2m;
  int i_Sort;

  Particle *p_Next;
  Domain *domain() {return Domain::p_D;}
  void SetX(double x, double y, double z) {
    f_X = x; f_Y = y; f_Z = z;
  }
  void AddX(double x, double y, double z) {
    f_X += x; f_Y += y; f_Z += z;
  }
  void ExtractX(double x, double y, double z) {
    f_X -= x; f_Y -= y; f_Z -= z;
  }
  void SetP(double px, double py, double pz)
    {f_Px=px; f_Py=py; f_Pz=pz;};
  void GetX(double &x, double &y, double &z) {
    x = f_X; y = f_Y; z = f_Z;
  }
  void GetP(double &px, double &py, double &pz)
    {px=f_Px; py=f_Py; pz=f_Pz;};
  double GetWeight() {return f_Weight;};
  long GetCellNumber() {return l_Cell;};

  Particle *Next() {return p_Next;};
  virtual double GetQ2m(void) {return f_Q2m;};
  virtual int GetSort(void) {return i_Sort;};
  virtual Specie* GetSpecie(void) {return domain()->GetSpecie(i_Sort);};
  virtual int GoingZombie(void) {return this->GetSpecie()->GoingZombie();};
  virtual int IsBeam(void) {return this->GetSpecie()->IsBeam();};
  virtual double FindP2(void) {return f_Px*f_Px+f_Py*f_Py+f_Pz*f_Pz ;};
  virtual void Pack(CBuffer *j);
  virtual void UnPack(CBuffer *j);
  virtual long Save(FILE* pFile);
  virtual long Load(FILE* pFile);
  virtual void Collide(Cell &c){};
  virtual long Ionize(Cell *c=NULL, double Efield=0.){ return 0;}; 
  virtual void RemoveElectron(Cell *c=NULL){ return;}; 
  virtual void Recombine(Cell &c){};
   double GetScatteringProbability(double P, double IonDensityInCell);
   virtual int GetZ() {return 1;};
  void operator=(SavedParticle &p);
  long GetNionized();

  Particle(Cell *c=NULL, double weights=0.,
	   double xs=0., double ys=0., double zs=0.,
	   double pxs=0., double pys=0., double pzs=0.,
	   double q2ms=1., int isort=0);
  virtual ~Particle() {;};
};

//---------------------------- Electron class -----------------------

class Electron : public Particle {
 public:

  virtual long Ionize(Cell *c=NULL, double Efield=0.){ return 0;}; 
  void Recombine(Cell &c) { return;};
  virtual void RemoveElectron(Cell *c=NULL){ return;}; 
  void Collide(Cell &c);

  Electron(Cell *c=NULL, double weights=0,
	   double xs=0., double ys=0., double zs=0.,
	   double pxs=0., double pys=0., double pzs=0., double q2ms=1., int isort=0);
  ~Electron();
};

//---------------------------- Ion class -----------------------

class Ion : public Particle {
 public:
  int i_Z;
  int GetSort(void){return i_Sort;};
  Specie *p_Specie;
  virtual Specie *GetSpecie(void) {return domain()->GetSpecie(i_Sort);};

  double GetQ2m(void);
  int GetZ() {return i_Z;};
  virtual long Ionize(Cell *c=NULL, double Efield=0.);
  virtual void RemoveElectron(Cell *c=NULL); 
  virtual void Recombine(Cell &c);
  virtual void Collide(Cell &c);
  virtual void Pack(CBuffer *j);
  virtual void UnPack(CBuffer *j);
  virtual long Save(FILE* pFile);
  virtual long Load(FILE* pFile);
  void operator=(SavedIon &p);

  Ion(Cell *c=NULL, int sorts=1, int state=0, double weights=0., double q2ms=1.,
      double xs=0.,  double ys=0.,  double zs=0.,
      double pxs=0., double pys=0., double pzs=0. );
  ~Ion();
};

class SavedParticle {
	public:
	int l_Cell;
  union{ double f_Momenta[1];
    double f_X;};
  double f_Y, f_Z, f_Px, f_Py, f_Pz, f_Weight, f_Q2m;
  void operator=(Particle &p)
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
};

class SavedIon {
	public:
	int l_Cell;
  union{ double f_Momenta[1];
    double f_X;};
  double f_Y, f_Z, f_Px, f_Py, f_Pz, f_Weight, f_Q2m;
  int i_Z;
  int i_Sort;
  void operator=(Particle &p)
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
	  i_Z = p.GetZ();
	  i_Sort = p.GetSort();
  }
};

#endif
