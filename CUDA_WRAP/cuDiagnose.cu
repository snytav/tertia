#include "cuLayers.h"


int copyFieldsFomLayersTo3D(cudaLayer **h_layers,int Nx,int Ny,int Nz,
                            double **Jx)//, double**Jy, double **Jz)
{
  *Jx = (double *)malloc(Nx*Ny*Nz*sizeof(double));

  for(int i = 0;i< Nx;i++)
  {
      for(int j = 0;j < Ny;j++)
      {
          for(int k = 0;k< Nz;k++)
          {
              *Jx[i*Ny*Nz+j*Nz +k] = (*h_layers[i]).Jx[j*Nz +k];
          }
      }
  }
  return 0;
}

int CUDA_WRAP_diagnose_host_layers(int Nx,int Ny,int Nz,
                                   double hx,double hy,double hz,
                                   cudaLayer **h_layers,
                                   int step)
{
    double *h_ex;
    char fname[100];
    FILE *f;
    double *Jx;

   // printf("in writesection %d \n",GetRank());

    h_ex  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);

    copyFieldsFomLayersTo3D(h_layers,Nx,Ny,Nz,&Jx);

    sprintf(fname,"field_%s_XY_%03d_rank%03d.dat","Jx",step,0);

    f = fopen(fname,"wt");

    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          // long nccc = mesh->GetN(i,j,k);
          // Cell &ccc = p_CellArray[nccc];
	   double t;
	   int k = Nz/2;
    	   fprintf(f," %e %e %15.5e \n",i*hx,j*hy,h_ex[i*Ny*Nz+k*Ny+j]);

       }
   //    puts("=======================================================");
    }
    double xi  = 0.0;

    fclose(f);

    free(h_ex);

   // printf("end writesection %d \n",GetRank());

    return 0;
}




