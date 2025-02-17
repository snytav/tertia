#ifndef H_DOMAIN
#define H_DOMAIN

#include <stdio.h>



#include "vlpl3dclass.h"

#define PI 3.1415927

//---------------------------- Domain class -----------------------
class Domain : public NList{
  friend class Mesh;

 public:
  static Domain *p_D;
  char *p_BufferMPI;
  int l_BufferMPIsize;
  ofstream out_Flog;
 private:

  // begin variables
  char c_Begin[4];
  double f_Ts;
  double f_TsIni;
  double f_TimeIni;
  double f_Hx;
  double f_HxSplit;
  int    i_NxSplit;
  double f_Hy;
  double f_Hz;
  double f_Xlength;
  double f_Ylength;
  double f_Zlength;
  int    i_nIter;

  long l_Xsize, l_Ysize, l_Zsize;
  long l_dX, l_dY, l_dZ;

  double f_Ts2Hx;
  double f_Ts2Hy;
  double f_Ts2Hz;
  double f_BTs2Hx;
  double f_BTs2Hy;
  double f_BTs2Hz;

  double f_Wavelength;

  int i_Nsorts;
  int i_Npulses;
  int i_NMovieFrames;
  int i_NMovieFramesH5;

  double f_BxExternal;
  double f_ByExternal;
  double f_BzExternal;

  int i_MaxwellSolver;
  int i_Hybrid;

  char c_End[4];
  // end variables

  FILE *p_File;
  FILE *p_SaveFile;
  FILE *p_MovieFile;
  FILE *p_PoyntingFile;
  ofstream out_Fig8;

  char *str_File;
  char str_MovieFile[256];
  char str_FileName[256];
  char *str_SName;
  char *str_DName;
  char *str_LogName;
  char str_FNumber[6];
  char str_PeNumber[6];

  char *str_DataDirectory;
  char *str_LogDirectory;
  char *str_MovieDirectory;

  FieldStencil m_Stencil;

  Mesh *p_M;
  Partition *p_MPP;
  Boundary *p_BndXm;
  Boundary *p_BndXp;
  Boundary *p_BndYm;
  Boundary *p_BndYp;
  Boundary *p_BndZm;
  Boundary *p_BndZp;

  UnitsCGS *p_CGS;
  Controls *p_Cntrl;
  Plot *p_Plot;
  Pulse **pa_Pulses;
  Specie **pa_Species;

  MovieFrame* p_MovieFrame;
  MovieFrame* p_MovieFrameH5;

  Synchrotron *p_Synchrotron;

public:                      // functions

   FILE* GetIniFile() {return p_File;}
   ofstream& Getout_Flog() {return out_Flog;}

  double GetHx() {return f_Hx;};
  double GetTs();
  double GetHxSplit() {return f_HxSplit;};
  int    GetNxSplit() {return i_NxSplit;};
  int    nIter() {return i_nIter;};
  double GetHy() {return f_Hy;};
  double GetHz() {return f_Hz;};
  int GetMaxwellSolver() {return i_MaxwellSolver;};
  int HybridIncluded() {return i_Hybrid;};
  double GetXlength() {return f_Xlength;};
  double GetYlength() {return f_Ylength;};
  double GetZlength() {return f_Zlength;};
  double GetWavelength() {return f_Wavelength;};

  int        GetNsorts(void) {return i_Nsorts;};
  Specie*    GetSpecie(int sort);
  long       Add2Specie(int sort);
  long       RemoveFromSpecie(int sort);
  Mesh*      GetMesh(void)   {return p_M;};
  Controls*  GetCntrl(void)  {return p_Cntrl;};
  Partition* GetMPP(void)    {return p_MPP;};
  UnitsCGS*  GetCGS(void)    {return p_CGS;};
  int        GetmyPE(void);
  Pulse*     GetPulse(int ipulse);
  Synchrotron* GetSynchrotron() {return p_Synchrotron;};
  double      GetPhase();

  double      GetBxExternal() {return f_BxExternal;};
  double      GetByExternal() {return f_ByExternal;};
  double      GetBzExternal() {return f_BzExternal;};

//<SergK>
  CBuffer*	 GetBufMPP();
	void		 ResetBufMPP();
//</SergK>

  void BroadCast(CBuffer *b);
  FILE* GetPoyntingFile(void) {return p_PoyntingFile;};

  void ArrangeThings(void);
  long l_NProcessed;
  int Check(void);
  int Diagnose(void);
  int Step(void);
  int GroupSteps(void);
  int Run(void);
  void Exchange(int what);

  Boundary* GetBndXm(){return p_BndXm;};
  Boundary* GetBndXp(){return p_BndXp;};

  Boundary* GetBndYm(){return p_BndYm;};
  Boundary* GetBndYp(){return p_BndYp;};

  Boundary* GetBndZm(){return p_BndZm;};
  Boundary* GetBndZp(){return p_BndZp;};

  int XmEdge(void);
  int XpEdge(void);
  int YmEdge(void);
  int YpEdge(void);
  int ZmEdge(void);
  int ZpEdge(void);
  long Save();
  long Load(int rank);
  int   nPEs(void);
  int   nPE(void);
  int   iPE(void);
  int   jPE(void);
  int   kPE(void);
  int Xpartition(void);
  int Ypartition(void);
  int Zpartition(void);
  int MyTID(void);
  Processor *GetMyPE(void);
  void BCast(CBuffer*);

  Domain(char *infile, int rank);
  ~Domain();

};

#endif
