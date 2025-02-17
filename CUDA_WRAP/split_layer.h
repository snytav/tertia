#ifdef CUDA_WRAP_SPLIT_LAYER_H
#define CUDA_WRAP_SPLIT_LAYER_H

#include "cuCell.h"




int CUDA_WRAP_allocLayer(struct cudaLayer **dl,int Ny,int Nz,int Np);

int CUDA_WRAP_copyLayerToDevice(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,struct cudaLayer **dl);

#endif