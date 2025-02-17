#ifndef H_PULSE
#define H_PULSE

#include <stdio.h>



#include "vlpl3dclass.h"

#define EX 1
#define EY 2
#define EZ 3
#define BX 4
#define BY 5
#define BZ 6

//---------------------------- Pulse class -----------------------

class Pulse : public NList {
  friend class Domain;
private:
  Domain *domain() {return Domain::p_D;};

  char c_Begin[4]; // begin of variables
  double f_OmegaCGS, f_NcCGS;
  double f_Omega, f_Nc;

  double f_A, f_Anorm;
  double f_Xpol, f_Ypol, f_Zpol;
  double f_Length;
  double f_Ywidth;
  double f_Zwidth;
  double f_Xcenter;
  double f_Ycenter;
  double f_Zcenter;
  double f_YcenterOscillationAmplitude, f_ZcenterOscillationAmplitude;
  double f_YcenterOscillationPeriod, f_ZcenterOscillationPeriod;
  double f_YcenterOscillationPhase, f_ZcenterOscillationPhase;
  double f_Rise;
  double f_Drop;
  double f_Yphase;
  double f_Zphase;
  int i_Tprofile;
  int i_Lprofile;

  union
  {
    double f_Fields[1];
    double f_Ex;
  };

  double f_Ey, f_Ez, f_Bx, f_By, f_Bz;

  double f_Kx, f_Ky, f_Kz; // k-vector of the pulse
  double f_Kxy, f_Kxyz; // absvalues of the k-vector
  int i_From;

  int i_Pulse;
  char c_End[4]; // end of variables

  Boundary *p_B;
  char str_B[4];

public:
  Boundary* GetBnd() {return p_B;};
  int GetFrom() {return i_From;};

  void GetFields(double &ex, double &ey, double &ez, double &bx, double &by, double &bz)
  {
    ex = f_Ex; ey = f_Ey; ez = f_Ez; bx = f_Bx; by = f_By; bz = f_Bz;
  }

  double* GetFields()
  {
    return f_Fields;
  };

  double ProfileL(double xco, double yco, double zco);
  double ProfileT(double xco, double yco, double zco);
  double Form(double xco, double yco, double zco, int pol);
  double* Form(double xco, double yco, double zco);
  long Save(FILE* pFile);
  long Load(FILE* pFile);

  Pulse(char *nm, FILE *f, int ipulse=0);
};
#endif
