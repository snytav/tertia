#ifndef CUDA_WRAP_PARA_CPU_LAYERS_H
#define CUDA_WRAP_PARA_CPU_LAYERS_H

int CUDA_WRAP_getLayerFromMesh(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer **host_layer);

int CUDA_WRAP_setLayerToMesh(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer *host_layer,int first_flag);

int CUDA_WRAP_getLayerParticlesNumber(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,char *where);

void CUDA_WRAP_setBeamFFT(double *jx,double *rho,int ncomplex);

#endif