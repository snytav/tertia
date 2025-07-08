#ifndef CUDA_WRAP_CU_LAYERS_H 
#define CUDA_WRAP_CU_LAYERS_H

#include "cuCell.h"
#include "../particles.h"
#include "../cells.h"
#include "../mesh.h"

int copyLayerStructureFromHostToDevice(cudaLayer **dl_dp,cudaLayer *hl_dp);

int copyLayerFromHostToDevice(cudaLayer **hl_dp,cudaLayer *hl_hp);

void setLayersPC(cudaLayer *c,cudaLayer*p);

void getLayersPC(cudaLayer **c,cudaLayer **p);

int CUDA_WRAP_copyLayerFrom3D(int iLayer,int Ny,int Nz,int Np,cudaLayer **h_cl);

int CUDA_WRAP_copyLayerDeviceToDevice(int Ny,int Nz,int Np,cudaLayer *h_dl,cudaLayer *h_sl);

int CUDA_WRAP_interpolateLayers(int iSplit,int NxSplit,int Ny,int Nz,cudaLayer *h_cl,cudaLayer *h_pl,cudaLayer *h_left,cudaLayer *h_right);

int CUDA_WRAP_allocLayerOnHost(cudaLayer **h_l,int Ny,int Nz,int Np);

int CUDA_WRAP_copyArraysDevice(
int a_size,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9
);

int CUDA_WRAP_copyArraysDeviceIterate(int l_My,int l_Mz,double *d_rJx,double *d_rJy,double *d_rJz,double *d_rJxBeam,double *d_rRhoBeam,double *d_rRho); 


int CUDA_WRAP_storeArraysToDevice(
int Nx,int Ny,
double *d_Ex,
double *d_Ey,
double *d_Ez,
double *d_Bx,
double *d_By,
double *d_Bz,
double *d_Jx,
double *d_Jy,
double *d_Jz,
double *d_Rho
);

int CUDA_WRAP_storeArraysToDeviceC(
int Nx,int Ny,
double *d_Ex,
double *d_Ey,
double *d_Ez,
double *d_Bx,
double *d_By,
double *d_Bz,
double *d_rRho
);

int  get2D_index(cudaLayer *cl,int i,int j);

int CUDA_WRAP_printLayerParticles(cudaLayer *h_l,char *s);

int CUDA_WRAP_copyLayerParticles(cudaLayer *h_dst,cudaLayer *_src);

int CUDA_WRAP_clearCurrents(int Ny,int Nz,int rho_flag);

int CUDA_WRAP_printParticles(beamParticle *d_bp,char *s);

int cuLayerPrintCentre(cudaLayer *h_cl,int iLayer,Mesh *mesh,Cell *p_CellArray,char *where);

int CUDA_WRAP_printBeamDensity3D(Mesh *p_M,int step,char *where);

int CUDA_WRAP_printBeamParticles(Mesh *p_M,int step,char *where);

int CUDA_WRAP_printPlasmaParticles(Mesh *p_M,int step,char *where);

#endif
