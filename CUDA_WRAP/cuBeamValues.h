#ifndef BEAM_VALUES_H
#define BEAM_VALUES_H

extern double *h_beam_values,*d_beam_values;

int CUDA_WRAP_alloc_beam_values(int Np,int num_attr,double **h_p,double **d_p);


int CUDA_WRAP_write_beam_value(int i,int num_attr,int n,double *h_p,double t);

int write_beam_value(int i,int num_attr,int n,double *d_p,double t);


double CUDA_WRAP_check_beam_values(int Np,int num_attr,double *h_p,double *d_p,int bsx,int bsy,char *fname,char *beam_or_plasma,int nstep);

#endif
