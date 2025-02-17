#define X_ACCESS

#include "vlpl3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <X11/Xlib.h>

#ifdef X_ACCESS 

Boolean runX(XtPointer client_data);

//---------------------------- Plot::Plot -----------------------
Plot::Plot(FILE *fl) : NList("Plot")
{
  char ntmp[10];

  file = fl;
  name = "vlpl1d";
  v = NULL;

  add_entry("name", name);
  add_entry("NWindows", &nview);
  add_entry("nX", &nX, 1);
  add_entry("width",&width,800);
  add_entry("height",&height,800);

//<SergK> ??????
  if(file)
  {
	rewind(file);
   read(file);
  };
#ifdef V_MPI
//<SergK> ?????????
  domain()->GetBufMPP()->reset();
//  domain()->GetBufMPP()->Init();
  pack_nls(domain()->GetBufMPP());
  domain()->BroadCast(domain()->GetBufMPP());
  if (f==NULL) unpack_nls(domain()->GetBufMPP());
#endif
//</SergK>

  if (nview<=0 || nX<=0) return;
  nY = nview/nX;

  XtSetLanguageProc (NULL, NULL, NULL);
  int iview = 1;
  char *argv[2];
  argv[0] = "Vlpl";
  
  //  toplevel = XtVaCreateWidget("TOP",xmFormWidgetClass,NULL);
  toplevel = XtVaAppInitialize (&app, "Vlpl1d", NULL, 0, 
  				&iview, argv, NULL,
  				NULL);
  main_w = XtVaCreateManagedWidget ("Vlpl1d-MOVIE",
				    xmFormWidgetClass, toplevel,
				    XmNspacing, 5,
				    NULL);
  gcv.foreground = WhitePixelOfScreen (XtScreen (main_w));

  iview = nview;
  XtManageChild (main_w);
  while (iview--) {
    sprintf(ntmp,"Window%d",iview);
    v = new Viewport(iview, this, ntmp, v);
  }
  XtRealizeWidget (toplevel);
  XtWorkProc procX;
  procX = ::runX;
  work_id = XtAppAddWorkProc (app, procX, XtPointer(this));
  
}

//--- Plot:: -----------------------
void Plot::show(void)
{
  Viewport *vtmp = v;
  if (v) { 
    m_pal.m_plot=this; 
    if (m_pal.m_w != main_w) { 
      m_pal.m_w=main_w; m_pal.SetGRAY(); 
    }
  } 
  while (vtmp) {
    vtmp->show();
    vtmp = vtmp->next;
  }
  XmUpdateDisplay(toplevel);
}

//--- ::  -----------------------
Boolean runX(XtPointer client_data)
{
  Boolean finished = False;
  Plot *plt = (Plot*)client_data;

  Viewport *vtmp = plt->v;
  while (vtmp) {
    vtmp->clear(0);
    vtmp = vtmp->next;
  }
  finished = plt->domain()->group_steps();
  if (finished) exit(0);
  return finished;
}

//---------------------------- Viewport::Viewport -----------------------
Viewport::Viewport(int iview, Plot *plot, char *nm, Viewport *nxt) 
  : NList(nm) 
{
  iv = iview;
  p = plot;
  next = nxt;
  xgrid = ygrid = f = NULL;
  nx = ny = 0;

  width = p->width/p->nX;
  gwidth = width-50;
  if (gwidth<40) gwidth=40;
  x_off = 40;
  height = p->height/p->nY;
  gheight = height-50;
  if (gheight<40) gheight=40;
  y_off = 20;

  sprintf(what,"dens");
  sprintf(whatX,"x");
  sprintf(whatY,"nothing");

  add_entry("what", what);
  add_entry("whatX", whatX);
  add_entry("whatY", whatY);
  add_entry("name", name);
  add_entry("nameX", nameX);
  add_entry("nameY", nameY);
  add_entry("xmin", &xmin, -1.);
  add_entry("xmax", &xmax, 1.);
  add_entry("ymin", &ymin, -1.);
  add_entry("ymax", &ymax, 1.);
  add_entry("fmin", &fmin, -1.);
  add_entry("fmax", &fmax, 1.);
  add_entry("dimension", &dim, 1);
  add_entry("direction", &dir, 1);
  add_entry("at1", &at1, -1);
  add_entry("at2", &at2, -1);

  rewind(plot->file);
  read(plot->file);

  getf = &Cell::get_densG;
  if (strcmp(what,"ex") == 0) getf = &Cell::get_exG;
  if (strcmp(what,"ey") == 0) getf = &Cell::get_eyG;
  if (strcmp(what,"ez") == 0) getf = &Cell::get_ezG;
  if (strcmp(what,"bx") == 0) getf = &Cell::get_bxG;
  if (strcmp(what,"by") == 0) getf = &Cell::get_byG;
  if (strcmp(what,"bz") == 0) getf = &Cell::get_bzG;
  if (strcmp(what,"I") == 0) getf = &Cell::get_intensityG;
  if (strcmp(what,"dens") == 0) getf = &Cell::get_densG;

  gety = &Cell::y;
  getx = &Cell::x;
  if (strcmp(whatX,"x") == 0) { 
    getx=&Cell::x; dir=1;
    if (strcmp(whatY,"y") == 0) { gety = &Cell::y; dir=1; dim=2;}
    if (strcmp(whatY,"z") == 0) { gety = &Cell::z; dir=2; dim=2;}
  }
  if (strcmp(whatX,"y") == 0) { 
    getx = &Cell::y; dir = 2;
    if (strcmp(whatY,"x") == 0) { gety = &Cell::x; dir=4; dim=2;}
    if (strcmp(whatY,"z") == 0) { gety = &Cell::z; dir=3; dim=2;}
  }
  if (strcmp(whatX,"z") == 0) { 
    getx = &Cell::z; dir = 3;
    if (strcmp(whatY,"x") == 0) { gety = &Cell::x; dir=5; dim=2;}
    if (strcmp(whatY,"y") == 0) { gety = &Cell::y; dir=6; dim=2;}
  }

  if (next) {
    frame_sw=XtVaCreateManagedWidget("frame4",
				     xmFrameWidgetClass, p->main_w,
				     XmNshadowType,      XmSHADOW_ETCHED_IN,
				     XmNtopAttachment,   XmATTACH_WIDGET,
				     XmNtopWidget,       next->frame_sw,
				     XmNleftAttachment,  XmATTACH_FORM,
				     XmNrightAttachment, XmATTACH_FORM,
				     NULL);
  }
  else {
    frame_sw=XtVaCreateManagedWidget("frame4",
				     xmFrameWidgetClass, p->main_w,
				     XmNshadowType,      XmSHADOW_ETCHED_IN,
				     XmNtopAttachment,   XmATTACH_FORM,
				     XmNtopWidget,       XmATTACH_FORM,
				     XmNleftAttachment,  XmATTACH_FORM,
				     XmNrightAttachment, XmATTACH_FORM,
				     NULL);
  }
  sw=XtVaCreateManagedWidget("scrolled_win",
				xmScrolledWindowWidgetClass, frame_sw,
				XmNwidth,                  width,
				XmNheight,                  height,
				XmNscrollingPolicy,        XmAUTOMATIC,
				XmNscrollBarDisplayPolicy, XmAS_NEEDED,
				XmNtopAttachment,          XmATTACH_FORM,
				XmNbottomAttachment,       XmATTACH_FORM,
				XmNleftAttachment,         XmATTACH_FORM,
				XmNrightAttachment,        XmATTACH_FORM,
				NULL);
  width = width - 10;
  height = height - 10;

  d_a = XtVaCreateWidget ("drawing_area",
				xmDrawingAreaWidgetClass, sw,
				XmNunitType, XmPIXELS,
				XmNwidth, width,
				XmNheight, height,
				XmNresizePolicy, XmNONE,  
				/* remain this a fixed size */
				NULL);
  XtVaSetValues (d_a, XmNunitType, XmPIXELS, NULL);
  gc = XCreateGC (XtDisplay (d_a),
		  RootWindowOfScreen (XtScreen (d_a)), 
		  GCForeground, &(p->gcv));
  XtVaSetValues (d_a, XmNuserData, gc, NULL);

  XtManageChild (frame_sw);
  XtManageChild (sw);
  XtManageChild (d_a);
  pixmap = XCreatePixmap (XtDisplay (d_a),
			  RootWindowOfScreen (XtScreen (d_a)), 
			  width, height,
			  DefaultDepthOfScreen (XtScreen (d_a)));
  /* Change the pallette */
  //  XGetWindowAttributes(XtDisplay(d_a), 
  //		       XtWindow(d_a),
  //		       &WAd_a);
}

//--- Viewport:: -----------------------
void Viewport::set_limits(float xmin_a, float xmax_a, 
			  float ymin_a, float ymax_a)
{
  xmin = xmin_a; xmax = xmax_a; ymin = ymin_a; ymax = ymax_a;
}

//--- Viewport:: -----------------------
int Viewport::world2winX(float xco_a)
{
  int tmp = gwidth*((xco_a-xmin)/(xmax-xmin));
  if (tmp<0) tmp = 0;
  if (tmp>gwidth-1) tmp = gwidth-1;
  return tmp+x_off;
}

//--- Viewport:: -----------------------
int Viewport::world2winY(float yco_a)
{
  int tmp = gheight*((ymax-yco_a)/(ymax-ymin));
  if (tmp<0) tmp = 0;
  if (tmp>gheight-1) tmp = gheight-1;
  return tmp+y_off;
}

//--- Viewport:: -----------------------
float Viewport::world2winF(float fco_a)
{
  float tmp;
  if (fmax>fmin) {
    tmp = ((-fmin+fco_a)/(fmax-fmin));
    if (tmp<0.) tmp = 0.;
    if (tmp>.99999) tmp = .99999;
  } else tmp = 0.;
  return tmp;
}

//--- Viewport:: -----------------------
void Viewport::world2win(float xco_a, float yco_a, int &i_a, int &j_a)
{
  i_a = world2winX(xco_a);  j_a = world2winY(yco_a);
}

//--- Viewport:: -----------------------
void Viewport::world2win(float xco_a, float yco_a, short &i_a, short &j_a)
{
  i_a = world2winX(xco_a);  j_a = world2winY(yco_a);
}

//--- Viewport:: -----------------------
void Viewport::sect1d(void)
{
  if (xgrid) delete[] xgrid; xgrid = NULL;
  if (ygrid) delete[] ygrid; ygrid = NULL;
  if (f) delete[] f; f = NULL;

  Domain *d = Domain::domain;
  Mesh *m = d->mesh;
  int i, j, k;
  switch (dir) {
  case 1:
    i = -m->gX;    j = at1;     k = at2;
    if (j<0 || j>m->sizeY) j = m->sizeY/2;
    if (k<0 || k>m->sizeZ) k = m->sizeZ/2;
    nx = ny = m->fsizeX-1;
    NextCell = &Cell::nextX;
    break;
  case 2:
    i = at1;    j = -m->gY;     k = at2;
    if (i<0 || i>m->sizeX) i = m->sizeX/2;
    if (k<0 || k>m->sizeZ) k = m->sizeZ/2;
    nx = ny = m->fsizeY-1;
    NextCell = &Cell::nextY;
    break;
  case 3:
    i = at1;    j = at2;     k = -m->gZ;
    if (i<0 || i>m->sizeX) i = m->sizeX/2;
    if (j<0 || j>m->sizeY) j = m->sizeY/2;
    nx = ny = m->fsizeZ-1;
    NextCell = &Cell::nextZ;
    break;
  default:
    return;
  }

  Cell *c = m->cell(i,j,k);
  if (c==NULL) return;
  xgrid = new float[nx];
  ygrid = new float[nx];
  f = new float[nx];

  while (c=(c->*NextCell)()) {
    if (++i>=nx) break;
    xgrid[i] = (c->*getx)();
    ygrid[i] = f[i] = (c->*getf)();
  }
  sprintf(whatY,what);
  set_limits(xmin,xmax,fmin,fmax);
}

//--- Viewport:: -----------------------
void Viewport::plot1d(void)
{
  Display* display;
  display = XtDisplay(d_a);

  clear(0);
  /* drawing is now drawn into with "black"; change the gc */
  XSetForeground (display, gc,
		  BlackPixelOfScreen (XtScreen (d_a)));

  int i = nx-1;
  XPoint* points = new XPoint[nx];
  world2win(xgrid[i], f[i], points[i].x, points[i].y);
  while(i--) {
    world2win(xgrid[i], f[i], points[i].x, points[i].y);
  };
  XDrawLines(display, pixmap, gc, points, nx-1, CoordModeOrigin);
  XDrawRectangle(display, pixmap, gc, x_off, y_off, gwidth, gheight);
  axes();
  XCopyArea(XtDisplay (d_a), pixmap, XtWindow(d_a), gc, 
	    0, 0, width, height, 0, 0);
  delete[] points;
}

//--- Viewport:: -----------------------
void Viewport::axes(void)
{
  int x, y;
  float value;
  char straxis[SSIZE];

  Display* display = XtDisplay(d_a);
  /* drawing is now drawn into with "black"; change the gc */
  XSetForeground (display, gc,
		  BlackPixelOfScreen (XtScreen (d_a)));

  sprintf (straxis,"%4g",ymax);    // YMAX label
  int Length = strlen(straxis);
  x = x_off - Length*m_fontWidth;
  y = y_off + m_fontHeight;
  XDrawString(display, pixmap, gc, x, y, straxis, Length);

  sprintf (straxis,"%4g",ymin);    // YMIN label
  Length = strlen(straxis);
  x = x_off - Length*m_fontWidth;
  y = y_off + gheight;
  XDrawString(display, pixmap, gc, x, y, straxis, Length);

  Length = strlen(whatY);    // WHATY label
  x = 1;
  y = y_off + gheight/2;
  XDrawString(display, pixmap, gc, x, y, whatY, Length);

  sprintf (straxis,"%4g",xmin);    // XMIN label
  Length = strlen(straxis);
  x = x_off - (Length-1)*m_fontWidth;
  y = y_off + gheight + m_fontHeight;
  XDrawString(display, pixmap, gc, x, y, straxis, Length);

  sprintf (straxis,"%4g",xmax);    // XMAX label
  Length = strlen(straxis);
  x = gwidth + x_off - m_fontWidth*Length;
  y = y_off + gheight + m_fontHeight;
  XDrawString(display, pixmap, gc, x, y, straxis, Length);

  Length = strlen(whatX);    // WHATY label
  x = x_off+gwidth/2;
  y = y_off + gheight + m_fontHeight;
  XDrawString(display, pixmap, gc, x, y, whatX, Length);

  Length = strlen(what);    // WHAT label
  x = x_off+gwidth/2;
  y = y_off - m_fontHeight*0.33;
  XDrawString(display, pixmap, gc, x, y, what, Length);

  sprintf (straxis,"Time=%5g",domain()->cntrl->phase);    // PHASE LABEL
  Length = 10;
  x = x_off + gwidth - (Length)*m_fontWidth;
  y = y_off - m_fontHeight*0.33;
  XDrawString(display, pixmap, gc, x, y, straxis, Length);

  value = 0.;
  world2win(value, value, x, y);
  XDrawLine(XtDisplay (d_a), pixmap, gc, x_off, y, x_off+gwidth, y);
  XDrawLine(XtDisplay (d_a), pixmap, gc, x, y_off, x, y_off+gheight);
}

//--- Viewport:: -----------------------
void Viewport::clear(int flag)
{
  XSetForeground (XtDisplay (d_a), gc,
		  WhitePixelOfScreen (XtScreen (d_a)));
  XFillRectangle (XtDisplay (d_a), pixmap, gc, 
		  0, 0, width, height);
  /* drawing is now drawn into with "black"; change the gc */
  XSetForeground (XtDisplay (d_a), gc,
		  BlackPixelOfScreen (XtScreen (d_a)));
  XDrawRectangle(XtDisplay (d_a), pixmap, gc, x_off, y_off, gwidth, gheight);
  if (flag) XCopyArea(XtDisplay (d_a), pixmap, XtWindow(d_a), gc, 
	    0, 0, width, height, 0, 0);
} 

//--- Viewport::2D -----------------------
void Viewport::sect2d(void)
{
  if (xgrid) delete[] xgrid; xgrid = NULL;
  if (ygrid) delete[] ygrid; ygrid = NULL;
  if (f) delete[] f; f = NULL;

  Domain *d = Domain::domain;
  Mesh *m = d->mesh;
  int i, j, k;
  switch (dir) {
  case 1:
    i = -m->gX;    j = -m->gY;     k = at1;
    if (k<0 || k>m->sizeZ) k = m->sizeZ/2;
    nx = m->fsizeX-1;
    ny = m->fsizeY-1;
    NextXCell = &Cell::nextX;
    NextYCell = &Cell::nextY;
    break;
  case 2:
    i = -m->gX;    j = at1;     k = -m->gZ;
    if (j<0 || j>m->sizeY) j = m->sizeY/2;
    nx = m->fsizeX-1;
    ny = m->fsizeZ-1;
    NextXCell = &Cell::nextX;
    NextYCell = &Cell::nextZ;
    break;
  case 3:
    i = at1;    j = -m->gY;     k = -m->gZ;
    if (i<0 || i>m->sizeX) i = m->sizeX/2;
    nx = m->fsizeY-1;
    ny = m->fsizeZ-1;
    NextXCell = &Cell::nextY;
    NextYCell = &Cell::nextZ;
    break;
  case 4:
    i = -m->gX;    j = -m->gY;     k = at1;
    if (k<0 || k>m->sizeZ) k = m->sizeZ/2;
    ny = m->fsizeX-1;
    nx = m->fsizeY-1;
    NextXCell = &Cell::nextY;
    NextYCell = &Cell::nextX;
    break;
  case 5:
    i = -m->gX;    j = at1;     k = -m->gZ;
    if (j<0 || j>m->sizeY) j = m->sizeY/2;
    ny = m->fsizeX-1;
    nx = m->fsizeZ-1;
    NextXCell = &Cell::nextZ;
    NextYCell = &Cell::nextX;
    break;
  case 6:
    i = at1;    j = -m->gY;     k = -m->gZ;
    if (i<0 || i>m->sizeX) i = m->sizeX/2;
    ny = m->fsizeY-1;
    nx = m->fsizeZ-1;
    NextXCell = &Cell::nextZ;
    NextYCell = &Cell::nextY;
    break;
  default:
    return;
  }

  Cell *c = m->cell(i,j,k);
  Cell *c0 = c;
  if (c==NULL) return;
  xgrid = new float[nx];
  ygrid = new float[ny];
  f = new float[nx*ny];

  i = -1;
  while (c=(c->*NextXCell)()) {
    if (++i>=nx) break;
    xgrid[i] = (c->*getx)();
  }
  c = c0;
  i = -1;
  while (c=(c->*NextYCell)()) {
    if (++i>=ny) break;
    ygrid[i] = (c->*gety)();
  }

  j = -1;
  while (c0=(c0->*NextYCell)()) {
    if (++j>=ny) break;
    c = c0;
    i = -1;
    while (c=(c->*NextXCell)()) {
      if (++i>=nx) break;
      f[i+nx*j] = (c->*getf)();
    }
  }
}

//--- Viewport:: -----------------------
void Viewport::plot2d(void)
{
  int x1=0, y1=0;
  int x2=0, y2=0;
  char cname[SSIZE];
  float value;
  unsigned long fgr;
  Display* display = XtDisplay(d_a);
  clear(0);
  int i=0, j=0;

  fgr = 0;
  for (j=0; j<ny-1; j++) {
    for (i=0; i<nx-1; i++) {
      world2win(xgrid[i], ygrid[j+1], x1, y1);
      world2win(xgrid[i+1], ygrid[j], x2, y2);
      value = world2winF(f[i+nx*j]);
      //      value = float(i*j)/float(nx-1.)/float(ny-1.);
      float r=value, g=value, b=value;
      fgr = p->m_pal.GetGrayPixel(r);
      XSetForeground(display, gc, fgr);
/*      if (j==my/2) printf("e=%g, color=%d, i=%d\n",fl,fgr,i);*/
      XDrawRectangle(display, pixmap, gc, 
		     x1, y1, x2-x1+1, y2-y1+1);
      XFillRectangle(display, pixmap, gc, 
		     x1, y1, x2-x1+1, y2-y1+1);
    };
  };

  XSetForeground (display, gc,
		  BlackPixelOfScreen (XtScreen (d_a)));
  XDrawRectangle(display, pixmap, gc, x_off, y_off, gwidth, gheight);
  axes();
  XCopyArea(XtDisplay (d_a), pixmap, XtWindow(d_a), gc, 
	    0, 0, width, height, 0, 0);
}

//--- PlotPallette:: -----------------------
void PlotPallette::SetRGB(void)
{
  if (m_plot==NULL) return;
  Display* display = XtDisplay(m_w);
  int i, pixel=0;
  m_scmpID = XDefaultColormap(display, XDefaultScreen(display));
  XWindowAttributes WAmw;
  XGetWindowAttributes(display, XtWindow(m_w), &WAmw);
  m_cmpID = XCreateColormap(display, XtWindow(m_w),
			  WAmw.visual, AllocNone);

  m_Cmap.base_pixel=16;
  m_Cmap.red_mult=36;
  m_Cmap.green_mult=6;
  m_Cmap.blue_mult=1;
  m_Cmap.red_max=6;
  m_Cmap.green_max=6;
  m_Cmap.blue_max=6;
  
  for (i=0; i< m_Cmap.base_pixel; i++) {
    m_colors[pixel].pixel = pixel;
    pixel++;
    XQueryColor(XtDisplay(m_w), m_scmpID, &m_colors[i]);
  }
  int r, g, b;
  i = m_Cmap.base_pixel;
  for (r=0; r<m_Cmap.red_max; r++) {
    int red = float(r)/(m_Cmap.red_max-1.)*65535.;
    for (g=0; g<m_Cmap.green_max; g++) {
      int green = float(g)/(m_Cmap.green_max-1.)*65535.;
      for (b=0; b<m_Cmap.blue_max; b++) {
	int blue = float(b)/(m_Cmap.blue_max-1.)*65535.;
	if (pixel>=NumberColors) break;
	m_colors[pixel].pixel = pixel;
	m_colors[pixel].flags = DoRed | DoGreen | DoBlue;
	m_colors[pixel].red = red;
	m_colors[pixel].green = green;
	m_colors[pixel].blue = blue;
	pixel++;
      }
    }
  }

  for (i=0; i<NumberColors; i++) {
    if (!XAllocColor(XtDisplay(m_w), m_cmpID, &m_colors[i]))
      printf("Can't allocate color %d \n",i);
  }
  XInstallColormap(XtDisplay(m_w), m_cmpID);
  XSetWindowColormap(XtDisplay(m_w), XtWindow(m_w), m_cmpID);
}

//--- PlotPallette:: -----------------------
unsigned long PlotPallette::GetRGBPixel(float r, float g, float b)
{
  int i = m_Cmap.base_pixel + 
    int (0.5 + r*m_Cmap.red_max)*m_Cmap.red_mult + 
    int (0.5 + g*m_Cmap.green_max)*m_Cmap.green_mult + 
    int (0.5 + b*m_Cmap.blue_max)*m_Cmap.blue_mult;
  if (i<0) i = 0;
  if (i>=NumberColors) i = NumberColors-1;
  return m_colors[i].pixel;
}

//--- PlotPallette:: -----------------------
unsigned long PlotPallette::GetGrayPixel(float r)
{
  int i = m_Cmap.base_pixel + r*(NumberColors- m_Cmap.base_pixel);
  if (i<0) i = 0;
  if (i>=NumberColors) i = NumberColors-1;
  return m_colors[i].pixel;
}
 
//--- PlotPallette:: -----------------------
void PlotPallette::SetGRAY(void)
{
  if (m_plot==NULL) return;
  Display* display = XtDisplay(m_w);
  int i, pixel=0;
  m_scmpID = XDefaultColormap(display, XDefaultScreen(display));
  XWindowAttributes WAmw;
  XGetWindowAttributes(display, XtWindow(m_w), &WAmw);
  m_cmpID = XCreateColormap(display, XtWindow(m_w),
			  WAmw.visual, AllocNone);

  m_Cmap.base_pixel=32;
  m_Cmap.red_mult=0;
  m_Cmap.green_mult=0;
  m_Cmap.blue_mult=1;
  m_Cmap.red_max=0;
  m_Cmap.green_max=0;
  m_Cmap.blue_max=NumberColors-m_Cmap.base_pixel;
  
  for (i=0; i< m_Cmap.base_pixel; i++) {
    m_colors[pixel].pixel = pixel;
    pixel++;
    XQueryColor(XtDisplay(m_w), m_scmpID, &m_colors[i]);
  }
  int b;
  i = m_Cmap.base_pixel;
  for (b=0; b<m_Cmap.blue_max; b++) {
    int blue = float(b)/(m_Cmap.blue_max-1.)*65535.;
    if (pixel>=NumberColors) break;
    m_colors[pixel].pixel = pixel;
    m_colors[pixel].flags = DoRed | DoGreen | DoBlue;
    m_colors[pixel].red = m_colors[pixel].green = 
      m_colors[pixel].blue = blue;
    pixel++;
  }

  for (i=0; i<NumberColors; i++) {
    if (!XAllocColor(XtDisplay(m_w), m_cmpID, &m_colors[i]))
      printf("Can't allocate color %d \n",i);
  }
  XInstallColormap(XtDisplay(m_w), m_cmpID);
  XSetWindowColormap(XtDisplay(m_w), XtWindow(m_w), m_cmpID);
}

//--- Viewport:: -----------------------
void Viewport::show(void)
{
  Display* display = XtDisplay(d_a);
  int DirectionReturn, FontAscentReturn, FontDescentReturn;
  XCharStruct OverallReturn;

  XQueryTextExtents(display, XGContextFromGC(gc), "W", 1, 
		    &DirectionReturn, &FontAscentReturn,
		    &FontDescentReturn, &OverallReturn);
  m_fontWidth = OverallReturn.width;
  m_fontHeight = FontAscentReturn+FontDescentReturn;
  x_off = 8*m_fontWidth;
  y_off = 1.5*m_fontHeight;
  gwidth = width - x_off - 15;
  gheight = height - 3* m_fontHeight;
  switch (dim) {
  case 1:
    sect1d();
    plot1d();
    break;
  case 2:
    sect2d();
    plot2d();
    break;
  }
}
#endif
