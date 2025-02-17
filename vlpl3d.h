#ifndef H_VLPL3D

#define H_VLPL3D

//#include <use_ansi.h>

#include <vector>
#include <cmath>
#include <assert.h>
 
#define NIL 0

#undef _DEBUG

//#undef V_MPI

#define NO_X_ACCESS
#undef X_ACCESS

static double EdjX, EdjY, EdjZ;

#ifdef V_MPI
#include <mpi.h>
#endif

#include "buffers.h"
#include "namelist.h"

#define S_BAVERAGE
#include "vlpl3dclass.h"

#include "stencil.h"
#include "domain.h"
#include "cells.h"
#include "mesh.h"
#include "elements.h"
#include "partition.h"
#include "particles.h"
#include "boundary.h"
#include "CGS.h"
#include "controls.h"
#include "pulse.h"
#include "cell3d.h"
#include "synchrotron.h"
#include "vcomplex.h"

//#include "DistributionObject.h"
//#include "postprocessor.h"
//#include "ParticlesSpectrum.h"


#include "movieframe.h"

/*
#ifdef NO_X_ACCESS
#include "notplot.h"
#endif

#ifdef X_ACCESS
#include "plot.h"
#endif
*/

#endif
//#define DEB //

