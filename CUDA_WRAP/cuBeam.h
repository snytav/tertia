#ifndef CU_BEAM_H
#define CU_BEAM_H
#include "cuCell.h"


int CUDA_MALLOC_TEST(char *where);

int CUDA_WRAP_beam_prepare(int Nx,int Ny,int Nz,Mesh *mesh,Cell *p_CellArray);

int CUDA_WRAP_beam_move(int Np,int Nx,int Ny,int Nz,double hx,double hy,double hz,double ts,int nstep);

int CUDA_WRAP_compareBeamCurrents(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray);

int CUDA_WRAP_copy3Dfields(Mesh *mesh,Cell *p_CellArray,int Nx,int Ny,int Nz);

int CUDA_WRAP_diagnose(int l_Xsize,int l_Ysize,int l_Zsize,double hx, double hy,int step,Mesh *mesh,Cell *p_CellArray);

int CUDA_WRAP_copyLayerToDevice(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer **dl);

double cuda_atomicAdd(double *address, double val);

int CUDA_WRAP_printParticleListFromHost(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,char *where,int nstep);

int CUDA_WRAP_writeMatrixFromDevice(int Ny,int Nz,double hy,double hz,double *d_m,int iLayer,char *name);

int getBeamNp();
int setBeamNp(int n);


#endif

