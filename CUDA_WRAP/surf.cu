#include"surf.h"


__device__ void surf2Dread
(double *x_re,
                           double *in_surfaceT,
                           int nx,int ny,
                           int NY)
{
	double t;

	t = in_surfaceT[nx*NY+ ny];
	*x_re = t;	
}

