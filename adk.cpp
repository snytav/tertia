/***************************************************************************
                          adk.cpp  -  description
                             -------------------
    begin                : Mon July 2 2005
    copyright            : (C) 2005 by Anupam Karmakar
    email                : pukhov@tp1.uni-duesseldorf.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *  This function calculates the ADK probability for Ionization of atoms.  *
 *  Completed in July 2005.(Anupam Karmakar)                                  *
 *                                                                         *
 *                                                                         *
 ***************************************************************************/

#include "vlpl3d.h"
#include <stdlib.h>

#define E_NATLOG 2.718281828

//-----------------IonSpecie::GetADKprob----------------------------->
double IonSpecie::GetADKprob(double E, int charge)
{

   //------------CONSTANTS---------------------------------------//

   double wAtomic = p_CGS->GetOmegaAtomic();//4.134E16; // Atomic Unit. of freq.
   double rBohr = p_CGS->GetrB();//5.2918E-9; // Bohr radius.
   double rElec = p_CGS->Getre();//2.8179E-13; // Electron radius.

   //---------simulation parameters-----------------------------//
   double ts_Num = domain()->GetTs();// Code Numerical Time-steps
//   double Ts = p_CGS->GetHx(); //This is CGS timestep
   double Wavelength = p_CGS->GetWavelength(); // laser wavelength
   double OmegaBasic = p_CGS->GetOmegaLaser();//Laser freq.

   //-----------Ion parameters----------------------------------//

   double IPotAE = (1/27.212)*GetIonizationLevel(charge);// Ionization Potential. in A.U.
   int zAtom = charge + 1.; // charge of ion created after ionization; should be +1 after(?)

   double nEff = zAtom/sqrt(2.*IPotAE);// effective principle q no.
   double alpha = 1/137.;
   double laser_photon_energy = 1.2398e-4/Wavelength;
   double E_atomic_rel = alpha*alpha*alpha*(5.12e5/laser_photon_energy);
   double Eeff = E/E_atomic_rel; //Elec. field Normalized.


   //-------ADK Ionization formula in At. Units---------------//

   double C_nl = pow(((2.*E_NATLOG)/nEff),nEff)/sqrt(2*PI*nEff); //ADK Costant C_n*l
   int F_lm = 1.; //ADK constant f(l,m)
   //-------------------------------------------------------//
   double frac = pow((2*IPotAE),1.5)/Eeff;
   double term1 = sqrt(0.9549/frac);
   double term2 = pow((2.*frac),(2*nEff-1));
   double term3 = exp(-2./3.*frac);
   double Wadk = C_nl*C_nl*F_lm*IPotAE*term1*term2*term3;//(ADK prob. in Atomic Unit)

   //--------Porbability in Code parameters -----------------//
   double NumProb = (wAtomic/OmegaBasic)*2*PI*ts_Num*Wadk;

//   cout << "wAtomic="<<wAtomic<< " OmegaBasic="<<OmegaBasic<<" ts_Num="<<ts_Num<<" Wadk="<<Wadk<< endl;

   if (NumProb > 1.) return 1.;
   if (NumProb < 0.) {
      return 0.;
   }
   
   return NumProb;
}

