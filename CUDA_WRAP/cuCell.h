#ifndef CUDA_WRAP_CU_CELL_H
#define CUDA_WRAP_CU_CELL_H

#include "../particles.h"


typedef struct {
        double f_Ex,f_Ey,f_Ez,f_Bx,f_By,f_Bz;
} cudaCell;

typedef struct {
        double f_X,f_Y,f_Z,f_Px,f_Py,f_Pz,f_Weight,f_Q2m;
	int i_X,i_Y,i_Z;
	int isort;
} beamParticle;

typedef struct {
  double *rho,*jx,*jy,*jz;
} beamCurrents;

typedef struct {
        double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho,*RhoBeam,*JxBeam,*fftRhoBeamHydro,*fftJxBeamHydro;
	beamParticle *particles;
	int Np,Ny,Nz;
} cudaLayer;

#endif
