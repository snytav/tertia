#ifndef CUDA_WRAP_TURN_H
#define CUDA_WRAP_TURN_H

int CUDA_WRAP_turnMatrix_from_hostCOMPLEX(int n1,int n2,double *m,double *phi,double *ktime);

int CUDA_WRAP_turnMatrix_from_deviceCOMPLEX(int n1,int n2,double *d_m,double *d_phi,double *ktime);

int CUDA_WRAP_create_alpha_surfaceCOMPLEX(int width,int height,double *h_data_in);

#endif