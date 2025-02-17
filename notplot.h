#ifndef H_NOTPLOT
#define H_NOTPLOT
#define SSIZE 128

#include "vlpl3d.h"

#define NumberColors 256

typedef double (Cell::*GETTYPE)(void);

//---------------------------- Plot class -----------------------

class Plot : public NList{
public:
  char* c_Name;
  FILE *p_File;

  Domain *domain() {return Domain::p_D;};
  Viewport *p_View;

  int Run(void) {return domain()->Run();};
  int i_Nview, i_Nx, i_Ny;
  int i_Width, i_Height;
  Plot(FILE *f) : NList(NULL) {return;};
};

//---------------------------- Viewport class -----------------------

class Viewport : public NList{
private:
  int i_View;
  double f_Xmin, f_Xmax;
  double f_Ymin, f_Ymax;
  double f_Zmin, f_Zmax;

  int i_Width, i_Height;
  int i_gWidth, i_gHeight;
  int i_Xoff, i_Yoff;
  double *fa_Xgrid, *fa_Ygrid, *fa_Function;
  int i_Nx, i_Ny;
  char str_Name[SSIZE], str_NameX[SSIZE], str_nameY[SSIZE];
  char str_What[SSIZE], str_WhatX[SSIZE], str_WhatY[SSIZE];

  Plot *p_Plot;

public:
  Domain *domain() {return Domain::p_D;};
  Viewport *p_NextView;
  Viewport(void) : NList(NULL) {return;};
};
#endif
