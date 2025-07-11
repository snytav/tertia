#include "cuCell.h"

#include <stdio.h>
#include <stdlib.h>
#include "../para.h"
#include "../particles.h"




 int CUDA_WRAP_createNewLayer(cudaLayer **h_l,cudaLayer *h_d_l)//,int Ny,int Nz,int Np)
 {
     int Ny,Nz,Np;
     cudaLayer *h_loc;

     puts("CUDA_WRAP_createNewLayer");

     h_loc = (cudaLayer *)malloc(sizeof(cudaLayer));
     if(h_loc == NULL)
     {
         puts("CUDA_WRAP_createNewLayer - no memory for layer");
 	exit(1);
     }

    // cudaMemcpy(*h_l,d_l,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
     printf("Ny %d Nz %d Np %d \n",h_d_l->Ny,h_d_l->Nz,h_d_l->Np);
     h_loc->Np = h_d_l->Np;
 //    return 0;
     h_loc->Ny = h_d_l->Ny;
     h_loc->Nz = h_d_l->Nz;

     Ny = h_loc->Ny;
     Nz = h_loc->Nz;
     Np = h_loc->Np;
     printf("creating new Layer Ny %d Nz %d Np %d \n",Ny,Nz,Np);
     h_loc->Bx =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->By =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Bz =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Ex =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Ey =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Ez =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Jx =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Jy =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Jz =               (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->fftRhoBeamHydro  = (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->fftJxBeamHydro   = (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->Rho =              (double *)malloc(Ny*Nz*sizeof(double));
     h_loc->particles = (beamParticle *)malloc(Np*sizeof(beamParticle));

     if(
         (h_loc->Bz == NULL) ||
         (h_loc->Ey == NULL) ||
         (h_loc->Jz == NULL) ||
         (h_loc->fftRhoBeamHydro == NULL)
       )
     {
         printf("MEMORY ERROR \n");
         exit(0);
     }
     *h_l = h_loc;

     return 0;
 }

int CUDA_WRAP_createNewLayerOnDevice(cudaLayer **h_dst_l,cudaLayer* h_l)
{
    int Ny,Nz,Np;
    
    printf("rank %d in createOnDev \n",GetRank());
//    exit(0);
    
    *h_dst_l = (cudaLayer *)malloc(sizeof(cudaLayer));
    
  //  cudaMemcpy(*h_l,d_l,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
    
    Ny = (h_l)->Ny;
    printf("rank %d in createOnDev A0 \n",GetRank());
    Nz = (h_l)->Nz;
    Np = (h_l)->Np;

    printf("rank %d in createOnDev A\n",GetRank());
//    exit(0);
    
    cudaMalloc(&((*h_dst_l)->Bx),Ny*Nz*sizeof(double));
    
    cudaMalloc(&((*h_dst_l)->By),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Bz),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Ex),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Ey),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Ez),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Jx),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Jy),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Jz),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->JxBeam),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->RhoBeam),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->Rho),Ny*Nz*sizeof(double));
    cudaMalloc(&((*h_dst_l)->particles),Np*sizeof(beamParticle));
    
    (*h_dst_l)->Np = Np;
    (*h_dst_l)->Ny = Ny;
    (*h_dst_l)->Nz = Nz;
    
    printf("rank %d in createOnDev B \n",GetRank());
//    exit(0);
    
    return 0;    
}

int CUDA_WRAP_fillLayer(cudaLayer *h_dst_l,int Ny,int Nz,int Np)
{
    
//    *h_dst_l = (cudaLayer *)malloc(sizeof(cudaLayer));
    
  //  cudaMemcpy(*h_l,d_l,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
    
    cudaMalloc(&((h_dst_l)->Bx),Ny*Nz*sizeof(double));
    
    cudaMalloc(&((h_dst_l)->By),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Bz),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Ex),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Ey),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Ez),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Jx),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Jy),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Jz),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->JxBeam),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->RhoBeam),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->Rho),Ny*Nz*sizeof(double));
    cudaMalloc(&((h_dst_l)->particles),Np*sizeof(beamParticle));
    
    (h_dst_l)->Np = Np;
    (h_dst_l)->Ny = Ny;
    (h_dst_l)->Nz = Nz;
    
    return 0;    
}


// int CUDA_WRAP_copyLayerFromDevice(cudaLayer **h_l,cudaLayer *d_l)
// {
//     int Ny,Nz,Np;
//
//     printf("in copyLayer \n");
//     CUDA_WRAP_createNewLayer(h_l,d_l);
//     printf("in copyLayer: layer created \n");
//
//     Ny = (*h_l)->Ny;
//     Nz = (*h_l)->Nz;
//     Np = (*h_l)->Np;
//     printf("in copyLayer:dims  \n");
//
//     cudaMemcpy((*h_l)->Bx,d_l->Bx,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->By,d_l->By,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Bz,d_l->Bz,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Ex,d_l->Ex,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Ey,d_l->Ey,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Ez,d_l->Ez,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Jx,d_l->Jx,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Jy,d_l->Jy,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Jz,d_l->Jz,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->JxBeam,d_l->JxBeam,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->RhoBeam,d_l->RhoBeam,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->Rho,d_l->Rho,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
//     cudaMemcpy((*h_l)->particles,d_l->particles,Np*sizeof(beamParticle),cudaMemcpyDeviceToHost);
//     printf("in copyLayer: host arrays alloced  \n");
//
//     return 0;
// }

int CUDA_WRAP_copyToNewLayerOnDevice(cudaLayer **h_dst_l,cudaLayer *h_l)
{
    int Ny,Nz,Np;
    
    CUDA_WRAP_createNewLayerOnDevice(h_dst_l,h_l);
    
    Ny = (*h_dst_l)->Ny;
    Nz = (*h_dst_l)->Nz;
    Np = (*h_dst_l)->Np;
    
    cudaMemcpy((*h_dst_l)->Bx,h_l->Bx,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->By,h_l->By,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Bz,h_l->Bz,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Ex,h_l->Ex,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Ey,h_l->Ey,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Ez,h_l->Ez,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Jx,h_l->Jx,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Jy,h_l->Jy,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Jz,h_l->Jz,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->JxBeam,h_l->JxBeam,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->RhoBeam,h_l->RhoBeam,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->Rho,h_l->Rho,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((*h_dst_l)->particles,h_l->particles,Np*sizeof(beamParticle),cudaMemcpyHostToDevice);
    
    return 0;
}

int CUDA_WRAP_copyToLayerOnDevice(cudaLayer *h_dst_l,cudaLayer *h_l)
{
    int Ny,Nz,Np;
    
    
    printf("rank %d in cpLayerOnDev \n",GetRank());
    
//    CUDA_WRAP_createNewLayerOnDevice(h_dst_l,h_l);
    
    Ny = (h_dst_l)->Ny;
    Nz = (h_dst_l)->Nz;
    Np = (h_dst_l)->Np;

    printf("rank %d in cpLayerOnDev A \n",GetRank());

    
    cudaMemcpy((h_dst_l)->Bx,h_l->Bx,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->By,h_l->By,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Bz,h_l->Bz,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Ex,h_l->Ex,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Ey,h_l->Ey,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Ez,h_l->Ez,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Jx,h_l->Jx,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Jy,h_l->Jy,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Jz,h_l->Jz,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->JxBeam,h_l->JxBeam,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->RhoBeam,h_l->RhoBeam,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->Rho,h_l->Rho,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy((h_dst_l)->particles,h_l->particles,Np*sizeof(beamParticle),cudaMemcpyHostToDevice);
    
    printf("rank %d in cpLayerOnDev B \n",GetRank());
//    exit(0);
    
    
    return 0;
}
