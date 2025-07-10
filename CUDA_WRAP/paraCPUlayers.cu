
#include "../particles.h"
#include "../cells.h"
#include "../mesh.h"
#include "cuCell.h"
#include "beam_copy.h"
#include "cuBeamValues.h"
#include "../run_control.h"
#include "split_layer.h"
#include "cuLayers.h"
#include <math.h>
#include "../para.h"

void CUDA_WRAP_getBeamFFT(double *jx,double *rho,int n);

//void CUDA_WRAP_setBeamFFT(double *jx,double *rho,int n);
double *fftJxBeamBuffer,*fftRhoBeamBuffer;

// int CUDA_WRAP_getLayerFromMesh(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer **host_layer)
// {
//    double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
//    double *JxBeamP,*RhoBeamP;
//    beamParticle *bp;
//    cudaLayer *h_dl;
//    int np = 0;
//
//    int err = cudaGetLastError();
//    printf("in getLayerFromMesh begin err %d \n",err);
//
//    for (int k=0; k<Nz; k++)
//    {
//       for (int j=0; j<Ny; j++)
//       {
//               long ncc = mesh->GetN(iLayer, j,k);
// 	     // long ncc = mesh->GetNyz(j,  k);
//               Cell &ccc = p_CellArray[ncc];
//
// 	      Particle *p  = ccc.GetParticles();
//
// 	      for(;p;np++)
// 	      {
// 		  p = p->p_Next;
// 	      }
//
//       }
//    }
//    int err1 = cudaGetLastError();
//    printf("in getLayerFromMesh Layer %d count err %d \n",iLayer,err1);
//
//    Ex = (double *)malloc(sizeof(double)*Ny*Nz);
//    Ey = (double *)malloc(sizeof(double)*Ny*Nz);
//    Ez = (double *)malloc(sizeof(double)*Ny*Nz);
//    Bx = (double *)malloc(sizeof(double)*Ny*Nz);
//    By = (double *)malloc(sizeof(double)*Ny*Nz);
//    Bz = (double *)malloc(sizeof(double)*Ny*Nz);
//    Jx = (double *)malloc(sizeof(double)*Ny*Nz);
//    Jy = (double *)malloc(sizeof(double)*Ny*Nz);
//    Jz = (double *)malloc(sizeof(double)*Ny*Nz);
//    Rho = (double *)malloc(sizeof(double)*Ny*Nz);
//    JxBeamP  = (double *)malloc(sizeof(double)*Ny*Nz);
//    RhoBeamP = (double *)malloc(sizeof(double)*Ny*Nz);
//
//    int err22 = cudaGetLastError();
//    printf("in getLayerFromMesh after alloc Layer %d count err %d \n",iLayer,err22);
//    //exit(0);
//
//
//    CUDA_WRAP_getBeamFFT(JxBeamP,RhoBeamP,Ny*Nz);
//
//    bp = (beamParticle *)malloc(np*sizeof(beamParticle));
//    if(bp == NULL) puts("bp NULL");
//    int err2 = cudaGetLastError();
//    printf("in copyLayerToDevice alloc err %d np before list composition %d \n",err2,np);
//
// //   CUDA_WRAP_allocLayer(dl,Ny,Nz,np);
//
//    np = 0;
//
//    FILE *f = fopen("layer_being_formedCPU.dat","wt");
//
//    for (int k=0; k<Nz; k++)
//    {
//       for (int j=0; j<Ny; j++)
//       {
//               long nccc = mesh->GetN(iLayer,j,  k);
//               Cell &ccc = p_CellArray[nccc];
// 	      Particle *p  = ccc.GetParticles();
//
// 	      if(p == NULL) printf("cell %5d %5d %5d has no particles \n",iLayer,j,k);
// 	      else
// 	      {
//
// 		// printf("cell %5d %5d %5d \n",iLayer,j,k);
// 	      }
//
// 	      for(;p;np++)
// 	      {
// 		 Particle *pc = bp + np;
// 		 pc->f_X      = p->f_X;
// 		 pc->f_Y      = p->f_Y;
// 		 pc->f_Z      = p->f_Z;
// 		 pc->f_Px     = p->f_Px;
// 		 pc->f_Py     = p->f_Py;
// 		 pc->f_Pz     = p->f_Pz;
// 		 pc->f_Weight = p->f_Weight;
// 		 pc->f_Q2m    = p->f_Q2m;
// 		 //if(total_np < 16) printf("Pz %d %25.15e \n",total_np,p_cuda->f_Pz);
// 		 //pc->i_X      = iLayer;
// #ifdef  CUDA_WRAP_PARALLEL_DEBUG
// 		 printf("i_X %d Layer %d \n",pc->i_X,iLayer);
// #endif
// 		 /*pc->i_Y      = j;
// 		 pc->i_Z      = k;
// 		 pc->isort    = p->GetSort()*/;
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//
// 		 fprintf(f,"%10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",np,pc->i_X,pc->i_Y,pc->i_Z,p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Py,p->f_Pz);
// 		 printf("getLayer rank %3d %10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",GetRank(),np,pc->i_X,pc->i_Y,pc->i_Z,pc->f_X,pc->f_Y,pc->f_Z,pc->f_Px,pc->f_Py,pc->f_Pz);
// #endif
//
//
// 		 p = p->p_Next;
// 	      }
// 	      long n = j + Ny*k;
//
//               Rho[n] = ccc.GetDens(); // - dens;
//
//               Ex[n]  = ccc.GetEx();
//               Ey[n]  = ccc.GetEy();
//               Ez[n]  = ccc.GetEz();
//
//               Bx[n]  = ccc.GetBx();
//               By[n]  = ccc.GetBy();
//               Bz[n]  = ccc.GetBz();
//
//               Jx[n] = ccc.GetJx();
//               Jy[n] = ccc.GetJy();
//               Jz[n] = ccc.GetJz();
// #ifdef  CUDA_WRAP_PARALLEL_DEBUG
// 	      printf("getLayerFields rank %2d %5d %5d E (%10.3e,%10.3e,%10.3e) B (%10.3e,%10.3e,%10.3e) J (%10.3e,%10.3e,%10.3e) Rho %10.3e \n",GetRank(),j,k,
// 		                                ccc.GetEx(),ccc.GetEy(),ccc.GetEz(),
// 		                                ccc.GetBx(),ccc.GetBy(),ccc.GetBz(),
// 		                                ccc.GetJx(),ccc.GetJy(),ccc.GetJz(),ccc.GetDens());
// #endif
//
//       }
//    }
//    int err3 = cudaGetLastError();
//    printf("in copyLayerToDevice particle list err %d \n",err3);
//
//
//    fclose(f);
//
//    h_dl = (cudaLayer *)malloc(sizeof(cudaLayer));
//
// //   cudaMemcpy(h_dl,*dl,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
//
//    h_dl->Ex = Ex;
//    h_dl->Ey = Ey;
//    h_dl->Ez = Ez;
//
//    h_dl->Bx = Bx;
//    h_dl->By = By;
//    h_dl->Bz = Bz;
//
//    h_dl->Jx = Jx;
//    h_dl->Jy = Jy;
//    h_dl->Jz = Jz;
//    h_dl->Rho = Rho;
//
//    h_dl->fftJxBeamHydro  = JxBeamP;
//    h_dl->fftRhoBeamHydro = RhoBeamP;
//
//    h_dl->Np = np;
//    h_dl->Ny = Ny;
//    h_dl->Nz = Nz;
//    h_dl->particles = bp;
//    int err4 = cudaGetLastError();
//
//    *host_layer = h_dl;
//    printf("1st particle % \n");
//    printf("1st particle %e \n",(*host_layer)->particles[0].f_Y);
//    return np;
// }



int CUDA_WRAP_setLayerToMesh(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer *host_layer,int first_flag)
{
    beamParticle *pc;
    Particle *new_p;
    int del_np = 0;
    double *JxBeamP,*RhoBeamP;
    
    
    
    if(first_flag == 1)
    {
       for (int k=0; k<Nz; k++)
       {
          for (int j=0; j<Ny; j++)
          {
	      
              //////////////////////////////////////////////////
              long nccc = mesh->GetN(iLayer,j,  k);
              Cell &ccc = p_CellArray[nccc];	    
	      Particle *p  = ccc.GetParticles();  
	      
	      while(p)
	      {
	            p = p->p_Next;
	            Particle *del_p = p;
	            delete del_p; 
	            
	            del_np++;
	      }
	      ccc.SetParticlesToNULL();
	  }
       }
    }
    printf("deleted %d particles \n",del_np);
    
            for(int i = 0;i < host_layer->Np;i++)   
            {
                 //pc = host_layer->particles + i;
                 //int L      = pc->i_X;
                 int j      = pc->i_Y;
                 int k      = pc->i_Z;
              
                 long nccc = mesh->GetN(iLayer,j,  k);
                 Cell &ccc = p_CellArray[nccc];	    
	         Particle *p  = ccc.GetParticles();
		
	         new_p = new Particle;	
		 
		 new_p->f_X       = pc->f_X;
		 new_p->f_Y       = pc->f_Y;
		 new_p->f_Z       = pc->f_Z;
		 new_p->f_Px      = pc->f_Px;
		 new_p->f_Py      = pc->f_Py;
		 new_p->f_Pz      = pc->f_Pz;
		 new_p->f_Weight  = pc->f_Weight;
		 new_p->f_Q2m     = pc->f_Q2m;
		 //if(total_np < 16) printf("Pz %d %25.15e \n",total_np,p_cuda->f_Pz);
		 new_p->i_Sort     = pc->isort;
		 
		 ccc.AddParticle(new_p);
#ifdef  CUDA_WRAP_PARALLEL_DEBUG		 
		 printf("%2d setLayer %10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",GetRank(),i,pc->i_X,pc->i_Y,pc->i_Z,pc->f_X,pc->f_Y,pc->f_Z,pc->f_Px,pc->f_Py,pc->f_Pz);
#endif		 
		 //p  = ccc.GetParticles();
		 //printf("%2d setDoubl %10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",GetRank(),i,pc->i_X,pc->i_Y,pc->i_Z,p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Py,p->f_Pz);
	      }  


   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
	      
              //////////////////////////////////////////////////
              long nccc = mesh->GetN(iLayer,j,  k);
              Cell &ccc = p_CellArray[nccc];	    
	      Particle *p  = ccc.GetParticles();      
	      double f[10];
	      
	      long n = j + Ny*k;

	      f[0] = host_layer->Ex[n];
	      f[1] = host_layer->Ey[n];
	      f[2] = host_layer->Ez[n];

	      f[3] = host_layer->Bx[n];
	      f[4] = host_layer->By[n];
	      f[5] = host_layer->Bz[n];
	      
	      
	      f[6] = host_layer->Jx[n];
	      f[7] = host_layer->Jy[n];
	      f[8] = host_layer->Jz[n];
	      f[9] = host_layer->Rho[n];
	      
	      ccc.SetAll(f);
#ifdef  CUDA_WRAP_PARALLEL_DEBUG	      
	      printf("setLayerFields %2d Layer %5d %5d %5d E (%10.3e,%10.3e,%10.3e) B (%10.3e,%10.3e,%10.3e) J (%10.3e,%10.3e,%10.3e) Rho %10.3e %10.3e %10.3e \n",GetRank(),iLayer,j,k,
		                                ccc.GetEx(),ccc.GetEy(),ccc.GetEz(),
		                                ccc.GetBx(),ccc.GetBy(),ccc.GetBz(),
		                                ccc.GetJx(),ccc.GetJy(),ccc.GetJz(),ccc.GetDens(),
		                                host_layer->fftJxBeamHydro[n],host_layer->fftRhoBeamHydro[n]);
#endif		                                
      }
   }
   
   fftJxBeamBuffer  = (double *)malloc(sizeof(double)*Ny*Nz);
   fftRhoBeamBuffer = (double *)malloc(sizeof(double)*Ny*Nz);
   for(int n = 0;n < Ny*Nz;n++)
   {
      fftJxBeamBuffer[n]  = host_layer->fftJxBeamHydro[n];
      fftRhoBeamBuffer[n] = host_layer->fftRhoBeamHydro[n];
     // printf("setLayerFieldsFFT %5d %15.5e %15.5e  \n ",n,host_layer->fftRhoBeamHydro[n],host_layer->fftJxBeamHydro[n]);
   }
   //CUDA_WRAP_setBeamFFT(host_layer->fftJxBeamHydro,host_layer->fftRhoBeamHydro,Ny*Nz);

    return 0;
}

void CUDA_WRAP_setBeamFFT(double *jx,double *rho,int ncomplex)
{
  
   for (int n=0; n<ncomplex; n++) 
   {
      jx[n]  = fftJxBeamBuffer[n];
      rho[n] = fftRhoBeamBuffer[n];
     // printf("setBeamFFT %5d %15.5e %15.5e  \n ",n,fftRhoBeamBuffer[n],fftJxBeamBuffer[n]);
   }    
}



int CUDA_WRAP_getLayerParticlesNumber(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,char *where)
{ 
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   cudaLayer *h_dl;
   int np = 0;
   
   int err = cudaGetLastError();
   printf("in getLayerParticles begin err %d \n",err);
      
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
              long ncc = mesh->GetN(iLayer, j,k);
	     // long ncc = mesh->GetNyz(j,  k);
              Cell &ccc = p_CellArray[ncc];
	      
	      Particle *p  = ccc.GetParticles();
		
	      for(;p;np++)
	      {
		  p = p->p_Next;
	      }
	      
      }
   }
   
   printf("Layer %3d AT %s np %d \n",iLayer,where,np);
   
   return np;
}     


