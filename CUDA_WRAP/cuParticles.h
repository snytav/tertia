#ifndef CUDA_WRAP_PARTICLES_H
#define CUDA_WRAP_PARTICLES_H

#include "cuCell.h"

#define NUMBER_ATTRIBUTES                  10
#define CUDA_WRAP_PARTICLE_START           6
#define CUDA_WRAP_PARTICLE_START_INDEX     (sizeof(double)*CUDA_WRAP_PARTICLE_START* NUMBER_ATTRIBUTES)
#define CUDA_WRAP_CONTROL_VALUES           170
#define PARTICLE_TOLERANCE                 1e-15
#define DELTA_TOLERANCE                    1e-10

extern double *d_partRho, *d_partJx,*d_partJy,*d_partJz;

int CUDA_WRAP_fill_particle_attributes(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray);

int CUDA_WRAP_move_particles(int Ny,int Nz,int part_per_cell_max,double hx,double hy,double hz,double djx0,double djy0, double djz0,double drho0,int i_fs,double *buf);

int CUDA_WRAP_load_fields(int Ny,int Nz);

int CUDA_WRAP_check_particle_attributes(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray);

int CUDA_WRAP_alloc_particle_values(int Ny,int Nz,int num_attr,int ppc_max,double **h_p,double **d_p);

int CUDA_WRAP_write_particle_value(int Ny,int i,int j,int num_attr,int ppc_max,int k,int n,double *h_p,double t);

double CUDA_WRAP_check_particle_values(int iLayer,int Ny,int Nz,int num_attr,int ppc_max,double *h_p,double *d_p);

int CUDA_WRAP_load_fields(int Ny,int Nz,double *rEx,double *rEy,double *rEz,double *rBx,double *rBy,double *rBz);

double CUDA_WRAP_getArraysToCompare(char *where,Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray);

int CUDA_WRAP_copy_particle_currents(int Nx,int Ny,int Nz,int iLayer);

int CUDA_WRAP_copy_particle_density(int Nx,int Ny,int Nz,int iLayer_jx,int iLayer_rho);

int CUDA_WRAP_check_on_device(int Ny,int Nz,Mesh *mesh,int i_layer,Cell *p_CellArray);

int CUDA_WRAP_write_particle_attributes_fromDevice(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray);

int CUDA_WRAP_setMaximalPICnumberOnDevice(int s);

int CUDA_WRAP_getMaximalPICnumberOnDevice();

int CUDA_WRAP_check_hidden_currents(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *P,double *d_array,char *name);

int CUDA_WRAP_ClearCurrents(int,int);

int CUDA_WRAP_getHiddenCurrents(char *where,Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray);

int allocCurrents(int Ny,int Nz,int iFullStep);

int CUDA_WRAP_EMERGENCY_HIDDEN_CURRENTS_COPY(int Ny,int Nz,Mesh *mesh,Cell *p_CellArray,int i_layer);

int CUDA_WRAP_PrintBoundaryValuesFromHost(int Ny,int Nz,Mesh *mesh,Cell *p_CellArray,int i_layer,char *where);

int CUDA_WRAP_check_all_hidden_fields(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *C,Cell *P,cudaLayer *h_cl,cudaLayer *h_pl);

int CUDA_WRAP_check_hidden_currents3D(Mesh *mesh,Cell *p_layer,int i_layer,int Ny,int Nz,double *d_array,char *name);

#endif



