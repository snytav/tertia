#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include "vlpl3d.h"

//---------------------------- UnitsCGS::UnitsCGS -----------------------
UnitsCGS::UnitsCGS(void)
{
  f_Qe = -4.8e-10;
  f_Me = 0.911e-27;
  f_Mp = 1836.*f_Me;

  f_Wavelength = domain()->GetWavelength();
  f_Hx = domain()->GetHx()*f_Wavelength;
  f_Hy = domain()->GetHy()*f_Wavelength;
  f_Hz = domain()->GetHz()*f_Wavelength;

  f_Clight = 3e10;
//	f_Ts = domain()->GetTs()*f_Wavelength/f_Clight;

  f_OmegaLaser = f_Clight*2.*PI/f_Wavelength;
  f_Ncrit = f_Me*f_OmegaLaser*f_OmegaLaser/(4.*PI*f_Qe*f_Qe);
  //For ADK Probability
  f_OmegaAtomic = 4.134E16; // Atomic Unit. of freq.
  f_r_B = 5.2918E-9; // Bohr radius.
  f_r_e = 2.8179E-13; // Electron radius. 
}
