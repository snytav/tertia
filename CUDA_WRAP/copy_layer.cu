#include "../cell3d.h"
#include "../mesh.h"
#include "cuLayers.h"

int CUDA_WRAP_get_particles_number(Mesh *mesh,Cell *p_CellArray)
{
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
//   beamParticle *bp;
   cudaLayer *h_dl;
   int np = 0,Ny,Nz;

   int err = cudaGetLastError();
   printf("in getLayerParticles begin err %d \n",err);

   Ny = mesh->GetMy();
   Nz = mesh->GetMz();
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
          long ncc = mesh->GetNyz(j,  k);
          Cell &ccc = p_CellArray[ncc];

	      Particle *p  = ccc.GetParticles();

	      for(;p;np++)
	      {
		  p = p->p_Next;
	      }

      }
   }

   return np;

}

int LayerAlloc(cudaLayer **cl,int Ny,int Nz, int Np)
{
    *cl = (cudaLayer*)malloc(sizeof(cudaLayer));

    (*cl)->Ny = Ny;
    (*cl)->Nz = Nz;
    (*cl)->Ny = Ny;
    (*cl)->Np = Np;
    (*cl)->particles = (beamParticle *)malloc(Np*sizeof(beamParticle));

    int size;
    (*cl)->Ex = (double *)malloc(size);
    (*cl)->Ey = (double *)malloc(size);
    (*cl)->Ez = (double *)malloc(size);

    (*cl)->Bx = (double *)malloc(size);
    (*cl)->By = (double *)malloc(size);
    (*cl)->Bz = (double *)malloc(size);

    (*cl)->Jx = (double *)malloc(size);
    (*cl)->Jy = (double *)malloc(size);
    (*cl)->Jz = (double *)malloc(size);

    (*cl)->Rho = (double *)malloc(size);
    (*cl)->JxBeam = (double *)malloc(size);
    (*cl)->fftJxBeamHydro = (double *)malloc(size);

    (*cl)->fftRhoBeamHydro = (double *)malloc(size);





    return 0;
}

int CUDA_WRAP_copy_from_CellArray2Layer(Mesh *mesh,Cell *p_CellArray,cudaLayer **cl)
{
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   cudaLayer *h_dl;
   int Ny,Nz,Np;







   Ny = mesh->GetMy();
   Nz = mesh->GetMz();

   Np = CUDA_WRAP_get_particles_number(mesh,p_CellArray);

   LayerAlloc(cl,Ny,Nz,Np);


   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
          long ncc = mesh->GetNyz(j,  k);
          Cell &ccc = p_CellArray[ncc];

	      Particle *p  = ccc.GetParticles();



      }
   }

   return 0;

}
