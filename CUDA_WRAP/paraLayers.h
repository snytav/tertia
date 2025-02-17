#ifndef CUDA_WRAP_PARA_LAYERS_H
#define CUDA_WRAP_PARA_LAYERS_H

#define LAYER_ATTRIBUTE_NUMBER 12
#define BEAM_PARTICLE_ATTRIBUTE_NUMBER 12

int CUDA_WRAP_fillLayer(cudaLayer *h_dst_l,int Ny,int Nz,int Np);

int CUDA_WRAP_copyLayerFromDevice(cudaLayer **h_l,cudaLayer *d_l);

int CUDA_WRAP_copyToNewLayerOnDevice(cudaLayer **d_l,cudaLayer *h_l);

int CUDA_WRAP_copyToLayerOnDevice(cudaLayer *d_l,cudaLayer *h_l);

int CUDA_WRAP_createNewLayer(cudaLayer **h_l,cudaLayer *h_d_l);


#endif
