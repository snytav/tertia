#ifndef H_BOUNDARY
#define H_BOUNDARY

#include <stdio.h>



#include "vlpl3dclass.h"

#define XDIR 4
#define YDIR 8
#define ZDIR 16

#define MDIR 1
#define PDIR 2

#define TOXM XDIR+MDIR
#define TOXP XDIR+PDIR
#define TOYM YDIR+MDIR
#define TOYP YDIR+PDIR
#define TOZM ZDIR+MDIR
#define TOZP ZDIR+PDIR

//---------------------------- Boundary class -----------------------

class Boundary : public NList {
private:
  Domain *domain() {return Domain::p_D;};

  char c_Begin[3]; // begin of variables // 3 or 4 ???
  char c_Where;
  int i_FieldCnd;
  int i_ParticleCnd;
  int i_RefreshN;
  int i_StopN;
  char c_End[4]; // end of variables

  Cell *p_RefreshCell;
  Cell *p_StopCell;
  Processor *p_MatePE;
  Particle *p_Zombies;

public:
  int GetFcnd(){return i_FieldCnd;};
  int GetPcnd(){return i_ParticleCnd;};
  char Where(){return c_Where;};
  Processor *MatePE() {return p_MatePE;};
  //  virtual void Send(int what);
  //  virtual void Receive(int what);
  long Save(FILE* pFile);
  long Load(FILE* pFile);
  Particle* AddZombie(Particle* p, Cell* c);
  long SaveZombies(FILE* pFile);
  
  Boundary(char *nm, FILE *f, char where, Processor* pe);
};
#endif
