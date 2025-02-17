#ifndef CUDA_WRAP_LINEARIZED_H
#define CUDA_WRAP_LINEARIZED_H

int CUDA_WRAP_linearized_loop(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,
double *d_fft_of_RhoP,
double *d_fft_of_JxP,
double *d_fft_of_JyP,
double *d_fft_of_JzP,
double *d_fft_of_Ex,
double *d_fft_of_Ey,
double *d_fft_of_Ez,
double *d_fft_of_ExP,
double *d_fft_of_EyP,
double *d_fft_of_EzP,
double *d_fft_of_Jx,
double *d_fft_of_Jy,
double *d_fft_of_Jz,
double *d_fft_of_Bx,
double *d_fft_of_By,
double *d_fft_of_Bz,
double *d_fft_of_JxBeam,
double *d_fft_of_RhoBeam,
double *d_fft_of_JxBeamP, // 18
double *d_fft_of_RhoBeamP  // 19
);

int CUDA_WRAP_linearizedIterateloop(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,
double *d_fft_of_RhoP,
double *d_fft_of_JxP,
double *d_fft_of_JyP,
double *d_fft_of_JzP,
double *d_fft_of_Ex,
double *d_fft_of_Ey,
double *d_fft_of_Ez,
double *d_fft_of_EyP,
double *d_fft_of_EzP,
double *d_fft_of_Jx,
double *d_fft_of_Jy,
double *d_fft_of_Jz,
double *d_fft_of_Bx,
double *d_fft_of_By,
double *d_fft_of_Bz,
double *d_fft_of_JxBeam,
double *d_fft_of_RhoBeam,
double *d_fft_of_JxBeamP,
double *d_fft_of_RhoBeamP
);


#endif