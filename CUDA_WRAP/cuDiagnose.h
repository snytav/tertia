#ifndef CU_DIAGNOSE_H

int CUDA_WRAP_diagnose_host_layers(int Nx,int Ny,int Nz,
                                   double hx,double hy,double hz,
                                   cudaLayer **h_layers,
                                   int step);

#endif
