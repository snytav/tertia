#ifndef H_PARTITION
#define H_PARTITION

#include "vlpl3d.h"

//---------------------------- Processor class -----------------------

class Processor {
  friend class Partition;
 private:
  Domain* p_D;

  char c_Begin[4]; // begin of variables
  int i_N;
  int i_Tid;
  int i_iPE, i_jPE, i_kPE, i_nPE;
  char c_End[4];  // end of variables

  Processor *p_Xm, *p_Xp;
  Processor *p_Ym, *p_Yp;
  Processor *p_Zm, *p_Zp;
  Partition *p_MPP;

  void Set(Partition *prt, int i, int j, int k);
 public:
  int GetiPE() { return i_iPE;};
  int GetjPE() { return i_jPE;};
  int GetkPE() { return i_kPE;};
  int GetnPE() { return i_nPE;};
  int XmEdge() { return i_iPE==0;};
  int YmEdge() { return i_jPE==0;};
  int ZmEdge() { return i_kPE==0;};
  int XpEdge();
  int YpEdge();
  int ZpEdge();
  long Save(FILE* pFile);
  long Load(FILE* pFile);
  Mesh* GetMesh();
};

//---------------------------- Partition class -----------------------

class Partition : public NList {
  friend class Domain;
private:
  char c_Begin[4]; // begin of variables
  int i_Xpart, i_Ypart, i_Zpart;
  int i_nPEs;
  int i_myPE;

  double f_mksElectron;
  double f_mksIon;
  double f_mksCell;
  char c_End[4]; // end of variables

  Processor *pa_PEs;
  Processor *p_myPE;
  Processor *p_XmPE;
  Processor *p_XpPE;
  Processor *p_YmPE;
  Processor *p_YpPE;
  Processor *p_ZmPE;
  Processor *p_ZpPE;

  Domain *p_D;

  int *ia_X, *ia_Y, *ia_Z;
  double *fa_Loading;

 public:
  CBuffer *p_Buf;
  CBuffer *p_Buftmp;
  void Init(void);
  void Xloading(void);
  void Yloading(int i);
  void Zloading(int i, int j);
  double GetCellLoading(int i, int j, int k);
  void Balance(void);
  int GetiPE(void) {return p_myPE->GetiPE();};
  int GetjPE(void) {return p_myPE->GetjPE();};
  int GetkPE(void) {return p_myPE->GetkPE();};
  int GetnPE(void) {return p_myPE->GetnPE();};
  int GetnPEs(void) {return i_nPEs;};
  int GetnPE(int i, int j, int k) { return i+i_Xpart*(j+i_Ypart*k);}
  CBuffer* GetBuf() { return p_Buf;};
  int GetXpart() {return i_Xpart;};
  int GetYpart() {return i_Ypart;};
  int GetZpart() {return i_Zpart;};
  Processor *GetPE(int i, int j, int k) { return pa_PEs+GetnPE(i,j,k);}
  Processor *GetPE(int n) { return pa_PEs+n;}
  int XmEdge(void) {return p_myPE->XmEdge();};
  int XpEdge(void) {return p_myPE->XpEdge();};
  int YmEdge(void) {return p_myPE->YmEdge();};
  int YpEdge(void) {return p_myPE->YpEdge();};
  int ZmEdge(void) {return p_myPE->ZmEdge();};
  int ZpEdge(void) {return p_myPE->ZpEdge();};

  void Send(Processor *ToPE, int what_a);
  void Receive(Processor *FromPE, int what_a);
  int Cycle(int i, int imax)
  {
		    if (i<0)
		    	i=Cycle(i+imax,imax);
		    else
		    if (i>=imax)
		    	i=Cycle(i-imax,imax);
		    return i;
  };
  int XCycle(int i) {return Cycle(i,i_Xpart);};
  int YCycle(int i) {return Cycle(i,i_Ypart);};
  int ZCycle(int i) {return Cycle(i,i_Zpart);};

  long Save(FILE* pFile);
  long Load(FILE* pFile);

  Partition(char *nm, FILE *f);
  ~Partition() { delete[] pa_PEs;
  delete[] ia_X; delete[] ia_Y; delete[] ia_Z;
  delete[] fa_Loading; };
};

#endif
