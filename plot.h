#ifdef X_ACCESS
#ifndef H_PLOT
#define H_PLOT
//#define X_ACCESS
#define SSIZE 128
#include <Xm/MainW.h>
#include <Xm/DrawingA.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/List.h>
#include <Xm/LabelG.h>
#include <Xm/Label.h>
#include <Xm/RowColumn.h>
#include <Xm/ScrolledW.h>
#include <Xm/Form.h>
#include <Xm/PanedW.h>
#include <Xm/TextF.h>
#include <Xm/Frame.h>
#define NumberColors 256
typedef double (Cell::*GETTYPE)(void);
typedef Cell*  (Cell::*NEXT)(void);
//---------------------------- Class Pallette  -----------------------
class PlotPallette {public:
  Plot* m_plot;
  Widget m_w;
  XStandardColormap m_Cmap;
  Colormap m_cmpID;
  Colormap m_scmpID;
  XColor m_colors[NumberColors];
  void SetRGB(void);
  void SetGRAY(void);
  unsigned long GetRGBPixel(double r, double g, double b);
  unsigned long GetGrayPixel(double val);
};

//---------------------------- Plot class -----------------------

class Plot : public NList{
private:
public:
  Widget toplevel, main_w;
  char* name;
  FILE *file;

  Domain *domain() {return Domain::domain;};
  Viewport *v;
  XtWorkProcId  work_id;

  int nview, nX, nY;
  int width, height;
  Pixmap pixmap;
  GC gc;
  XGCValues gcv;
  XtAppContext app;
  PlotPallette m_pal;

public:
  Boolean runX(XtPointer client_data);
  void show(void);
  void run(void) {XtAppMainLoop (app);};
  Plot(FILE *f);
};

//---------------------------- Viewport class -----------------------

class Viewport : public NList{
private:
  int iv;
  int dim, dir, at1, at2;
  Widget frame_sw, sw, d_a;
  double xmin, xmax;
  double ymin, ymax;
  double fmin, fmax;
  int width, height;
  int gwidth, gheight;
  int x_off, y_off;
  int m_fontWidth,  m_fontHeight;
  double *xgrid, *ygrid, *f;
  int nx, ny;
  char name[SSIZE], nameX[SSIZE], nameY[SSIZE];
  char what[SSIZE], whatX[SSIZE], whatY[SSIZE];
  Plot *p;

  GC gc;
  XWindowAttributes WAd_a;
  Pixmap pixmap;
  XtAppContext app;

public:
  Domain *domain() {return Domain::domain;};
  Viewport *next;
  GETTYPE getx;
  GETTYPE gety;
  GETTYPE getz;
  GETTYPE getf;
  NEXT NextCell;
  NEXT NextXCell;
  NEXT NextYCell;
  void clear(int flag);
  void clear(Pixmap pm_a);
  void redraw(void);
  void make_my_pallette(void);
  void set_color_fgr(void);
  void update(void);
  void sect1d(void);
  void sect2d(void);
  void plot1d(void);
  void plot2d(void);
  void axes(void);
  void show(void);
  void SetSCmap(void);
  void set_limits(double xmin_a, double xmax_a, double ymin_a, double ymax_a);
  void set_Flimits(double fmin_a, double fmax_a);
  int world2winX(double xco_a);
  int world2winY(double yco_a);
  double world2winF(double fco_a);
  void world2win(double xco_a, double yco_a, int &i_a, int &j_a);
  void world2win(double xco_a, double yco_a, short &i_a, short &j_a);
  void make_my_pallette( XColor* carray_a, int ncolors_a, int nfig_a );
  Viewport(int iview, Plot *plot_a, char *name_a, Viewport *next_a);
};
#endif
#endif
