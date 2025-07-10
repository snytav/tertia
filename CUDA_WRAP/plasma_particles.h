#ifndef CUDA_WRAP_PLASMA_PARTICLES_H
#define CUDA_WRAP_PLASMA_PARTICLES_H

#include "cuCell.h"

int create_h_plasma_particles(int Np);

int particlesPrepareAtLayer(Mesh *mesh,Cell *p_CellArrayP,Cell *p_CellArrayC,int iLayer,int Ny,int Nz,int Np);

void cuMoveSplitParticles(int iLayer,int Np,cudaLayer *d_cl,cudaLayer *d_pl,int Ny);

int CUDA_WRAP_write_plasma_value(int i,int num_attr,int n,double t);

//writing a value to the control array for a definite particle in a definite cell
int write_plasma_value(int i,int num_attr,int n,double *d_p,double t);

void cuMoveSplitParticles(int iLayer,int iSplit,int Np,
int Mx,int Ny,int Nz,double hx,double hy,double hz,
                                     double *djx0,double *djy0,double *djz0,double *drho0,int nsorts,int iFullStep,int nstep);

double CUDA_WRAP_print_plasma_values(int Np,int num_attr,char *s);

int CUDA_WRAP_print_layer(cudaLayer *d_l,int np,char *name,int Ny,int Nz);

#endif
