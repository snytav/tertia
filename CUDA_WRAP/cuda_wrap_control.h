#ifndef CUDA_WRAP_CONTROL_H
#define CUDA_WRAP_CONTROL_H

#define TOLERANCE_IDEAL 1e-15
#define TOLERANCE_RUDE  1e-15
#define DETAILS         -33013
#define NO_DETAILS      -33014

extern double last_wrong,last_fourier,last_ordinary,last_hidden,last_delta;
extern int last_max_delta_value;

//extern double *fft_of_JxP, *fft_of_JyP, *fft_of_JzP, *fft_of_RhoP;

void CUDA_WRAP_emergency_exit(char *where);

int CUDA_WRAP_compare_device_array(int n,double *h_m,double *d_m,double *frac_ideal,double *frac_rude,char *legend,char *where,int details_flag);

int timeBegin(int num_test);

int timeEnd(int num_test);

int timePrint();

int timeInit();

double CUDA_WRAP_verify_all_vectors_on_host(
int a_size,char *where,int details_flag,
double *a1,double *d_a1,char *s1,
double *a2,double *d_a2,char *s2,
double *a3,double *d_a3,char *s3,
double *a4,double *d_a4,char *s4,
double *a5,double *d_a5,char *s5,
double *a6,double *d_a6,char *s6,
double *a7,double *d_a7,char *s7,
double *a8,double *d_a8,char *s8,
double *a9,double *d_a9,char *s9,
double *a10,double *d_a10,char *s10,
double *a11,double *d_a11,char *s11,
double *a12,double *d_a12,char *s12,
double *a13,double *d_a13,char *s13,
double *a14,double *d_a14,char *s14,
double *a15,double *d_a15,char *s15,
double *a16,double *d_a16,char *s16,
double *a17,double *d_a17,char *s17,
double *a18,double *d_a18,char *s18,
double *a19,double *d_a19,char *s19,
double *a20,double *d_a20,char *s20,
double *a21,double *d_a21,char *s21
);

double CUDA_WRAP_verify_all_vectors_on_hostReal(
int a_size,char *where,int details_flag,
double *a1,double *d_a1,char *s1,
double *a2,double *d_a2,char *s2,
double *a3,double *d_a3,char *s3,
double *a4,double *d_a4,char *s4,
double *a5,double *d_a5,char *s5,
double *a6,double *d_a6,char *s6,
double *a7,double *d_a7,char *s7,
double *a8,double *d_a8,char *s8,
double *a9,double *d_a9,char *s9,
double *a10,double *d_a10,char *s10,
double *a11,double *d_a11,char *s11,
double *a12,double *d_a12,char *s12
);

#endif