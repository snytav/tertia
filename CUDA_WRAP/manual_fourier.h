#ifndef MANUAL_FOURIER_H
#define MANUAL_FOURIER_H


#define N1 8
#define N2 3

int manual1DfourierSin(int n1,double *source,double *res);
int manual1DfourierCos(int n1,double *source,double *res);

int manual2DfourierCos(int n1,int n2,double *source,double *res_re);
int manual2DfourierSin(int n1,int n2,double *source,double *res_re);

#endif