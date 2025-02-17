#ifndef H_ELEMENTS
#define H_ELEMENTS

#include <stdio.h>
#include "vlpl3dclass.h"

//---------------------------- Specie class (electrons) ---------------------
class Specie : public NList {
	friend class Domain;
	//protected:
public:
	Domain *domain() {return Domain::p_D;};

	char c_Begin[4]; // begin of variables
	int i_Sort;
	int i_Type;
   int i_ScatterFlag;
   int i_BeamFlag;
   double d_Sigma;
	int i_Distribution;

	double f_qCGS, f_mCGS, f_q2mCGS;  // charge, mass, q/m ratio in cgs
	double f_q, f_m, f_q2m;    // charge, mass, q/m ratio relative to electron
	double f_dJx, f_dJy, f_dJz, f_dRho;              // dimensionless current deposits

	int l_perCell;           // How many particles per cell
	double f_Narb;          // density in N_crit
	double f_Narb1;          // density1 in N_crit
	double f_Scale;          // scale
	double f_AngleY;          
	double f_AngleZ;          
	double f_OmegaParb;           // Plasma frequency in w_crit
	double f_Ncgs;
	double f_OmegaPcgs;
	double f_Weight;           // What part of Narb in a cell it gives
	double f_WeightCGS;        // How many real particles substitutes f_Weight=1
	long l_Np;                // How many numerical particles in PE

	double f_Begin;
	double f_PlateauBegin;
	double f_PlateauEnd;
	double f_End;
	double f_Px0, f_Py0, f_Pz0;
	double f_x0, f_y0, f_z0;
	double f_PspreadX, f_PspreadY, f_PspreadZ;
	double f_Radius, f_RadiusX, f_RadiusY, f_RadiusZ;
	double f_Delta;
	double f_Xperiod, f_Yperiod, f_Zperiod;
	double f_Xcurvature, f_Ycurvature, f_Zcurvature;
	double f_CurvatureBegin;
	double f_CurvatureEnd;

   int i_PhaseSpaceFillFlag; 
	int i_Zombie;

	char c_End[4]; // end of variables

	UnitsCGS *p_CGS;
public:
	double GetPspreadX() {return f_PspreadX;};
	double GetPspreadY() {return f_PspreadY;};
	double GetPspreadZ() {return f_PspreadZ;};
	virtual double Density(double xco, double yco, double zco);
	int GetSort() {return i_Sort;};
	int GetType() {return i_Type;}; // 0 - e, 1 - ion, 2 - hybrid
	int GetScatterFlag() {return i_ScatterFlag;}; // 0 - do not scatter
   double GetScatteringCrossSection(double P); // P is the electron momentum
	void GetdJ(double &jx, double &jy, double &jz, double &drho)
	{jx=f_dJx;    jy=f_dJy;    jz=f_dJz;    drho=f_dRho;};
	long Add(void) {
      return ++l_Np;
   };
	long Remove(void) {return --l_Np;};
	double GetWeight() {return f_Weight;};
	double GetWeightCGS() {return f_WeightCGS;};
	virtual double GetPolarity() {return 1.;};
	virtual int GetState0() {return 1;};
	long  GetNp() {
      return l_Np;
   };
	double GetPx() {return f_Px0;};
	double GetPy() {return f_Py0;};
	double GetPz() {return f_Pz0;};
	double GetQ2M() {return f_q2m;};
	int GetPperCell() {return l_perCell;};
	int GetPhaseSpaceFillFlag() {return i_PhaseSpaceFillFlag;}; 
   int GoingZombie() {return i_Zombie;};
	int IsBeam() {return i_BeamFlag;};
	virtual int GetAtomType() {return -1;}; 
	virtual int GetZmax() {return 1;}; 	
	virtual int GetZ() {return 1;}; 	
   virtual long Save(FILE* pFile);
	virtual long Load(FILE* pFile);
	virtual void ResetNp() {l_Np = 0;};
	Specie(char *nm, FILE *f);
};

//---------------------------- IonSpecie class -----------------------
class IonSpecie : public Specie {
private:
	int i_Zmax;
	int i_State0;
	double *p_fPion;  // Table of ionization potentials, size i_Zmax;
	double *p_fIion;  // Table of CGS intensities for Barrier-Suppr.-Ioniz.;
	double *p_fEion;  // Table of numerical fields for Barrier-Suppr.-Ioniz.;
	double f_NuclMass;          // Nuclear mass in a.e.;
	char *c_name;
	double f_Polarity;
public:
	double GetPolarity() {return f_Polarity;};
	virtual int GetState0() {return i_State0;};
	virtual int GetZmax() {return i_Zmax;}; 	
   virtual long Save(FILE* pFile);
	virtual long Load(FILE* pFile);
	IonSpecie(int isort, char *nm, FILE *file);

   //------ionization parameters-----/
   virtual double GetIonizationLevel(int charge) ;
   virtual double GetIonizationProbability(double E, int charge);
   virtual double GetADKprob(double E, int charge);  //Numerical ADK Probability
	virtual int GetAtomType() {return i_AtomType;};

   int i_AtomType;//what type of atom?
   vector <double> Ionization_Levels; //Table of Ionization potential in eV      
private:
   enum AtomType{
      Hydrogen = 1,
      Helium = 2,
      Carbon = 6,
      Aluminium = 13,
      Silicon = 14,
      Copper = 29,
      Xenon = 54,
      }; 
};

#endif
