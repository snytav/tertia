#ifndef CUDA_WRAP_NORMALIZATION_H
#define CUDA_WRAP_NORMALIZATION_H

int normalizationLoop(
int ny,
int nz,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz,
double *d_rJx,
double *d_rJy,
double *d_rJz
);

int normalizationIterateLoop(
int ny,
int nz,
double *d_rRho,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz
);

double CUDA_WRAP_getMaxDeviation(
int ny,int nz,
double *d_rEx,
double *d_rEy,
double *d_rEz,
double *d_rBx,
double *d_rBy,
double *d_rBz  
);

int CUDA_WRAP_alloc_backup_fields(int size);

int CUDA_WRAP_backUpFields(int size,double *ex,double *ey,double *ez,double *hx,double *hy,double *hz);

#endif