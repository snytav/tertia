#ifndef SURF_H
  __device__ void surf2Dread
(
    double *x_re,
                           double *in_surfaceT,
                           int nx,int ny,
                           int ny1
);

__device__ void surf2Dwrite
(
                           double *in_surfaceT,
                           int nx,int ny,
                           int ny1,
                           double t
 );



#endif
