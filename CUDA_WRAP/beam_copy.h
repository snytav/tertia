#ifndef CUDA_WRAP_BEAM_COPY
#define CUDA_WRAP_BEAM_COPY

#include "../particles.h"
#include "../cells.h"
#include "../mesh.h"


extern double *d_RhoBeam3D,*d_JxBeam3D,*d_Rho3D,*d_Jx3D,*d_Jy3D,*d_Jz3D,*d_JyBeam3D,*d_JzBeam3D;
extern double *d_Ex3D,*d_Ey3D,*d_Ez3D,*d_Bx3D,*d_By3D,*d_Bz3D;

int CUDA_WRAP_copyBeamToArray(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray,double **d_rho_beam,double **d_jx_beam);

int CUDA_WRAP_3Dto2D(int iLayer,int Ny,int Nz,double *d_3d,double *d_2d);

int CUDA_WRAP_2Dto3D(int iLayer,int Nx,int Ny,int Nz,double *d_3d,double *d_2d);

int CUDA_WRAP_alloc3Dcurrents(int Nx,int Ny,int Nz);

int CUDA_WRAP_copyLayerCurrents(int iLayer,int Nx,int Ny,int Nz,double *rho,double *jx,double *jy,double *jz);

int CUDA_WRAP_restoreLayerCurrents(int iLayer,int Nx,int Ny,int Nz,double *rho,double *jx,double *jy,double *jz);

int CUDA_WRAP_setCurrentsToZero(int Ny,int Nz,double *jx,double *jy,double *jz);

int CUDA_WRAP_alloc3Dfields(int Nx,int Ny,int Nz);

int CUDA_WRAP_alloc3DArray(int Nx,int Ny,int Nz,double **d_x);

int CUDA_WRAP_copyLayerFields(int iLayer,int Nx,int Ny,int Nz,double *ex,double *ey,double *ez,double *bx,double *by,double *bz);

int CUDA_WRAP_compare3DFields(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray);

#endif