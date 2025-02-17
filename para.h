#ifndef CUDA_WRAP_PARA_H
#define CUDA_WRAP_PARA_H

#define SEND_BUFSIZE_TAG    13132
#define SEND_BUFFER_TAG     13133
#define PARTICLE_TAG        13134
#define PARTICLE_NUMBER_TAG 13135


#include "CUDA_WRAP/cuCell.h"

int SendBeamParticles(int *Np);
int SendBeamParticlesUp(int *Np);
int SendBeamParticlesDown(int *Np);

int SendLayer(cudaLayer *d_l,int Ny,int Nz,int Np);

int ReceiveLayer(cudaLayer *h_l,int Ny,int Nz,int Np);

int ParallelInit(int argc,char *argv[]);

int SetXSize(long *l_Mx,double x_max);

int Set_l_Mx(int *l_Mx);

int ParallelFinalize();

int GetRank();

int GetSize();

double GetXmin();
double GetXmax();

int ParallelExit();

int getBeamNp();

int setBeamNp(int n);

#endif