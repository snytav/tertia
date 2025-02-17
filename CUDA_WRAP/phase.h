#ifndef CUDA_PHASE_H
#define CUDA_PHASE_H
int CUDA_WRAP_ComputePhaseShift_onDevice(int n,double **d_alp,double **d_omg);

int ComputePhaseShift(int n,double *alp,double *omg);

#endif