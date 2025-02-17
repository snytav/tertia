#ifndef HALF_INTEGER_H
#define HALF_INTEGER_H

#define N 8

int ComputePhaseShift(int n,double *alp,double *omg);

int fourierOnePi1Dcomplex(int n,double *f,double *f_im,double *res_real,double *res_imag);

int get_exp_form(int n,double*re,double *im,double *ph,double *amp);

int phase_shift_after_pi_k_div_2N (int n,int sign,double*re,double *im,double *re_new,double *im_new);

int phase_shift(int n,int sign,double phi,double*re,double *im,double *re_new,double *im_new);

int fourierHalfInteger1D(int n,double *f,double *f_im,double *res_re,double *res_im);

#endif