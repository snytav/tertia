#include<stdlib.h>
#include "../cell3d.h"
#include "../mesh.h"
#include "cuLayers.h"
#include "cuBeam.h"

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

int Particle2beamParticle(beamParticle *bp,Particle *p,int i,int j,int k)
{
    bp->f_X      = p->f_X;
    bp->f_Y      = p->f_Y;
    bp->f_Z      = p->f_Z;
    bp->f_Px     = p->f_Px;
    bp->f_Py     = p->f_Py, 
    bp->f_Pz     = p->f_Pz, 
    bp->f_Weight = p->f_Weight; 
    bp->f_Q2m    = p->f_Q2m;
    bp->isort    = p->i_Sort;
    bp->i_X      = i;
    bp->i_Y      = j;
    bp->i_Z      = k;

	return 0;
}

int copyParticleHost2Host(beamParticle *p1,beamParticle *p2)
{
   p1->f_Px     = p2->f_Px;
   p1->f_Py     = p2->f_Py;
   p1->f_Pz     = p2->f_Pz;
   p1->f_Q2m    = p2->f_Q2m;
   p1->f_Weight = p2->f_Weight;
   p1->f_X      = p2->f_X;
   p1->f_Y      = p2->f_Y;
   p1->f_Z      = p2->f_Z;
   p1->f_Pz     = p2->f_Pz;
   p1->i_X      = p2->i_X;
   p1->i_Y      = p2->i_Y;
   p1->i_Z      = p2->i_Z;

  p1->isort    = p2->isort;


}

int LayerAlloc(cudaLayer **cl,int Ny,int Nz, int Np)
{
    *cl = (cudaLayer*)malloc(sizeof(cudaLayer));

    (*cl)->Ny = Ny;
    (*cl)->Nz = Nz;
    (*cl)->Ny = Ny;
    (*cl)->Np = Np;
    (*cl)->particles = (beamParticle *)malloc(Np*sizeof(beamParticle));

    int size = Ny*Nz*sizeof(double);
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
    (*cl)->RhoBeam = (double *)malloc(size);
    (*cl)->fftJxBeamHydro = (double *)malloc(size);

    (*cl)->fftRhoBeamHydro = (double *)malloc(size);





    return 0;
}

int CUDA_WRAP_copy_from_CellArray2Layer(Mesh *mesh,Cell *p_CellArray,cudaLayer **cl,int iLayer)
{
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   cudaLayer *h_dl;
   int Ny,Nz,Np;
   CUDA_MALLOC_TEST("Cell2Layer begin"); 






   Ny = mesh->GetMy();
   Nz = mesh->GetMz();

   Np = CUDA_WRAP_get_particles_number(mesh,p_CellArray);
   Np = 40960;
   printf(" Ny %d Nz %d Np %d\n",Ny,Nz,Np);
    
   LayerAlloc(cl,Ny,Nz,Np);
   CUDA_MALLOC_TEST("LayerAlloc");
   int np = 0;

   long Mx,My,Mz,dMx,dMy,dMz;
   mesh->GetSizes(Mx,My,Mz,dMx,dMy,dMz);
   for (int k=0; k<Mz; k++)
   {
      for (int j=0; j<My; j++)
      {
          long ncc = mesh->GetNyz(j,  k);
          Cell &ccc = p_CellArray[ncc];

	      Particle *p  = ccc.GetParticles();

          for(;p;np++)
	      {
		     p = p->p_Next;
		     if(p != NULL)
		     {
				beamParticle bp;
				Particle2beamParticle(&bp,p,iLayer,j,k);
                copyParticleHost2Host(&((*cl)->particles[np]),&bp);
             }
	      }

	  int n =  get2D_index(*cl,k,j); 
         // add copy arrays!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  //printf("k %5d j %5d ncc %10d Ny*Nz %10d \n",k,j,n,Ny*Nz);
         (*cl)->Ex[n] = ccc.GetEx();
         (*cl)->Ey[n] = ccc.GetEy();
         (*cl)->Ez[n] = ccc.GetEz();




      }
   }
  // exit(0);
   CUDA_MALLOC_TEST("END 2aRRAY");

   return 0;

}
