#ifndef CUDA_WRAP_VECTORS_LIST
#define CUDA_WRAP_VECTORS_LIST


extern double *d_fft_of_Ex, *d_fft_of_Ey, *d_fft_of_Ez, *d_fft_of_Bx, *d_fft_of_By, *d_fft_of_Bz, // 6
              *d_fft_of_Jx, *d_fft_of_Jy, *d_fft_of_Jz, *d_fft_of_Rho,                            // 4 (10)
              *d_fft_of_JxP, *d_fft_of_JyP, *d_fft_of_JzP, *d_fft_of_RhoP,                        // 4 (14)
              *d_fft_of_JxBeam, *d_fft_of_JyBeam, *d_fft_of_JzBeam, *d_fft_of_RhoBeam,            // 4 (18)
              *d_fft_of_JxBeamP, *d_fft_of_JyBeamP, *d_fft_of_JzBeamP, *d_fft_of_RhoBeamP,        // 4 (22)
              *d_fft_of_ExRho, *d_fft_of_EyRho, *d_fft_of_EzRho;                                  // 3 (25)
              
extern double *d_rEx,*d_rEy,*d_rEz,*d_rBx,*d_rBy,*d_rBz,*d_rJx,*d_rJy,*d_rJz,*d_rRhoBeam,*d_rJxBeam,*d_rRho;

extern double *d_fft_of_ExP, *d_fft_of_EyP, *d_fft_of_EzP, *d_fft_of_BxP, *d_fft_of_ByP, *d_fft_of_BzP; 


int CUDA_WRAP_EMERGENCY_COPY(int ny,int nz,double *d_x,double *x);
       
int CUDA_WRAP_device_alloc(
int a_size,
double **d_a1,
double **d_a2,
double **d_a3,
double **d_a4,
double **d_a5,
double **d_a6,
double **d_a7,
double **d_a8,
double **d_a9,
double **d_a10,
double **d_a11,
double **d_a12,
double **d_a13,
double **d_a14,
double **d_a15,
double **d_a16,
double **d_a17,
double **d_a18,
double **d_a19,
double **d_a20,
double **d_a21,
double **d_a22,
double **d_a23,
double **d_a24,
double **d_a25 
);


int CUDA_WRAP_copy_all_vectors_to_device(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *a12,
double *a13,
double *a14,
double *a15,
double *a16,
double *a17,
/*
double *a18,
double *a19,
double *a20,
double *a21,
double *a22,
double *a23,
double *a24,
double *a25
*/
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17
/*,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_a24,
double *d_a25 */
);

int CUDA_WRAP_copy_all_vectors_to_host(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *a12,
double *a13,
double *a14,
double *a15,
double *a16,
double *a17,
double *a18,
double *a19,
double *a20,
double *a21,
double *a22,
double *a23,
double *a24,
double *a25,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_a24,
double *d_a25
);

int CUDA_WRAP_device_free(
		double *d_a1,
		double *d_a2,
		double *d_a3,
		double *d_a4,
		double *d_a5,
		double *d_a6,
		double *d_a7,
		double *d_a8,
		double *d_a9,
		double *d_a10,
		double *d_a11,
		double *d_a12,
		double *d_a13,
		double *d_a14,
		double *d_a15,
		double *d_a16,
		double *d_a17,
		double *d_a18,
		double *d_a19,
		double *d_a20,
		double *d_a21,
		double *d_a22,
		double *d_a23,
		double *d_a24,
		double *d_a25
);




int compare_vector_from_device(int n,double *h_v,double *d_v,char *s);

int CUDA_WRAP_copy_all_real_vectors_to_device(
int a_size,
double *a1,
double *a2,
double *a3,
double *a4,
double *a5,
double *a6,
double *a7,
double *a8,
double *a9,
double *a10,
double *a11,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11
);

int CUDA_WRAP_device_real_alloc(
int a_size,
double **d_a1,
double **d_a2,
double **d_a3,
double **d_a4,
double **d_a5,
double **d_a6,
double **d_a7,
double **d_a8,
double **d_a9,
double **d_a10,
double **d_a11
);


void CUDA_WRAP_free(double *d);

int CUDA_WRAP_deviceSetZero(
int a_size,
double *d_a1,
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9,
double *d_a10,
double *d_a11,
double *d_a12,
double *d_a13,
double *d_a14,
double *d_a15,
double *d_a16,
double *d_a17,
double *d_a18,
double *d_a19,
double *d_a20,
double *d_a21,
double *d_a22,
double *d_a23,
double *d_b1,
double *d_b2,
double *d_b3,
double *d_b4,
double *d_b5,
double *d_b6,
double *d_b7,
double *d_b8, 
double *d_b9
/*
double **d_b10,
double **d_b11*/
);

#endif
