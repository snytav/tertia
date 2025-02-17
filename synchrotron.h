#ifndef H_SYNCHROTRON
#define H_SYNCHROTRON

#include <stdio.h>



#include "vlpl3dclass.h"

//---------------------------- Class UnitsCGS -----------------------
class Synchrotron: public NList {
private:
  Domain *domain(void) {return Domain::p_D;};
  char c_Begin[4]; // begin of variables

  double f_Emin;
  double f_Emax;
  double f_LogEmin;
  double f_LogEmax;
  double f_SynMin; // minimum electron g-factor to consider radiation
  int i_nEbins;
  int i_nThetabins;
  int i_nPhibins;
  double *p_PhotonsArray3D;
  double f_EStepFactor;
  double f_PhiStep;
  double f_ThetaStep;
  int i_Save;

  char c_End[4]; // end of variables
public:
  void AddParticle(Particle* p, double nph, double theta, double phi, double Ephoton);
  int StoreDistributionHDF5();
  double GetRadiatedEnergy_eV();
  double GetRadiatedEnergy_J();
  double GetSynMin() {return f_SynMin;};
  long Save(FILE* pFile);
  long Load(FILE* pFile);

  Synchrotron(char *nm, FILE *f);
  ~Synchrotron(){ delete[] p_PhotonsArray3D;};
};
#endif
