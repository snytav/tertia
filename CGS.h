#ifndef H_CGS
#define H_CGS

#include <stdio.h>



#include "vlpl3dclass.h"

//---------------------------- Class UnitsCGS -----------------------
class UnitsCGS
{
private:
  Domain *domain(void) {return Domain::p_D;};
  char c_Begin[4]; // begin of variables
  double f_Qe;
  double f_Me;
  double f_Mp;

  double f_OmegaLaser, f_Ncrit;
  double f_Wavelength;
//  double f_Ts;
  double f_Hx;
  double f_Hy;
  double f_Hz;
  double f_Clight;
  double f_r_e; // Electron radius.
  double f_r_B; // Bohr radius.
  double f_OmegaAtomic; // Atomic Unit. of freq.   char c_End[4]; // end of variables
public:
  double GetQe() {return f_Qe;};
  double GetMe() {return f_Me;};
  double GetMp() {return f_Mp;};
  double GetOmegaLaser() {return f_OmegaLaser;};
  double GetCritDensity() {return f_Ncrit;};
  double GetWavelength() {return f_Wavelength;};
  double GetTs() {return domain()->GetTs()*f_Wavelength/f_Clight;};
  double GetHx() {return f_Hx;};
  double GetHy() {return f_Hy;};
  double GetHz() {return f_Hz;};
  double GetClight() {return f_Clight;};
  double Getre()   {return f_r_e;};
  double GetOmegaAtomic() { return f_OmegaAtomic; }
  double GetrB() { return f_r_B; }  
  long Save(FILE* pFile);
  long Load(FILE* pFile);

  UnitsCGS(void);
};
#endif
