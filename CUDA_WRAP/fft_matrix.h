#ifndef FFT_MATRIX_H
#define FFT_MATRIX_H

int HalfInteger(int n1,int n2,double *fft1d_tab);

int CUDA_WRAP_create_output_surfaceCOMPLEX_fromDevice(int width,int height,int depth,
                                                      double **surf,double *array);

//int CUDA_WRAP_create_alpha_surfaceCOMPLEX(int width,int height,surface<void,2> surf,cudaArray *array);

int CUDA_WRAP_create3Dsurface(int width,int height,int depth,double *surf,double *array);

#endif
