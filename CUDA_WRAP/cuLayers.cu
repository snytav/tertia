//#include "cuPrintf.cu"
//#include "diagnostic_print.h"
#include "../para.h"
//#include "cuPrintf.cu"

#include "cuCell.h"
#include "beam_copy.h"
//#include "assign_currents.cu"
//#include "cuBeamValues.h"
#include "../run_control.h"
#include "split_layer.h"
#include "cuLayers.h"
#include <math.h>


cudaLayer *tmpLayerC,*tmpLayerP;

void setLayersPC(cudaLayer *c,cudaLayer*p)
{
    tmpLayerC = c;
    tmpLayerP = p;
}

void getLayersPC(cudaLayer **c,cudaLayer **p)
{
    *c = tmpLayerC;
    *p = tmpLayerP;
}



// typedef struct {
//         double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho,*RhoBeam,*JxBeam,*fftRhoBeamHydro,*fftJxBeamHydro;
// 	beamParticle *particles;
// 	int Np,Ny,Nz;
// } cudaLayer;


int copyLayerStructureFromHostToDevice(cudaLayer **dl_dp,cudaLayer *hl_dp)
{
	cudaMalloc(dl_dp,sizeof(cudaLayer));
	cudaMemcpy(*dl_dp,hl_dp,sizeof(cudaLayer),cudaMemcpyHostToDevice);
	return 0;
}

int copyLayerFromHostToDevice(cudaLayer **hl_dp,cudaLayer *hl_hp)
{
    *hl_dp = (cudaLayer *)malloc(sizeof(cudaLayer));

    (*hl_dp)->Np = hl_hp->Np;
    (*hl_dp)->Ny = hl_hp->Ny;
    (*hl_dp)->Nz = hl_hp->Nz;
    int size = sizeof(double)*hl_hp->Ny*hl_hp->Nz;
    cudaMalloc(&((*hl_dp)->Ex),size);
    cudaMalloc(&((*hl_dp)->Ey),size);
    cudaMalloc(&((*hl_dp)->Ez),size);
    cudaMemcpy((*hl_dp)->Ex,hl_hp->Ex,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Ey,hl_hp->Ey,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Ez,hl_hp->Ez,size,cudaMemcpyHostToDevice);
    cudaMalloc(&((*hl_dp)->Bx),size);
    cudaMalloc(&((*hl_dp)->By),size);
    cudaMalloc(&((*hl_dp)->Bz),size);
    cudaMemcpy((*hl_dp)->Bx,hl_hp->Bx,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->By,hl_hp->By,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Bz,hl_hp->Bz,size,cudaMemcpyHostToDevice);
    cudaMalloc(&((*hl_dp)->Jx),size);
    cudaMalloc(&((*hl_dp)->Jy),size);
    cudaMalloc(&((*hl_dp)->Jz),size);
    cudaMemcpy((*hl_dp)->Jx,hl_hp->Jx,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Jy,hl_hp->Jy,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Jz,hl_hp->Jz,size,cudaMemcpyHostToDevice);

    cudaMalloc(&((*hl_dp)->JxBeam),size);
    cudaMalloc(&((*hl_dp)->Rho),size);
    cudaMemcpy((*hl_dp)->JxBeam,hl_hp->JxBeam,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Rho,hl_hp->Rho,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Jy,hl_hp->Jy,size,cudaMemcpyHostToDevice);
    cudaMemcpy((*hl_dp)->Jz,hl_hp->Jz,size,cudaMemcpyHostToDevice);
    cudaMalloc(&((*hl_dp)->RhoBeam),size);
    cudaMemcpy((*hl_dp)->RhoBeam,hl_hp->RhoBeam,size,cudaMemcpyHostToDevice);
    cudaMalloc(&((*hl_dp)->fftJxBeamHydro),size);
    cudaMemcpy((*hl_dp)->fftJxBeamHydro,hl_hp->fftJxBeamHydro,size,cudaMemcpyHostToDevice);
    cudaMalloc(&((*hl_dp)->fftRhoBeamHydro),size);
    cudaMemcpy((*hl_dp)->fftRhoBeamHydro,hl_hp->fftJxBeamHydro,size,cudaMemcpyHostToDevice);

    cudaMalloc(&((*hl_dp)->particles),hl_hp->Np*sizeof(beamParticle));
    cudaMemcpy((*hl_dp)->particles,hl_hp->particles,hl_hp->Np*sizeof(beamParticle),cudaMemcpyHostToDevice);

    return 0;

}


int CUDA_WRAP_copyArraysDevice(
int a_size,
double *d_a1,   // d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6,
double *d_a7,
double *d_a8,
double *d_a9
)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif  
   int err[10],i;
   //puts("BEGIN COPY ======================================================================================");
   //exit(0);
   err[0] = cudaMemcpy(d_a1,tmpLayerP->Ex,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
//   compare_vector_from_device(a_size,a1,d_a1,"copy1");
 //  puts("in devcpy0.5");
//   exit(0);

   err[1] = cudaMemcpy(d_a2,tmpLayerP->Ey,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a2,d_a2,"copy2");
   err[2] = cudaMemcpy(d_a3,tmpLayerP->Ez,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a3,d_a3,"copy3");
   err[3] = cudaMemcpy(d_a4,tmpLayerP->Jx,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a4,d_a4,"copy4");
   err[4] = cudaMemcpy(d_a5,tmpLayerP->Jy,a_size*sizeof(double),cudaMemcpyDeviceToDevice);  
   //compare_vector_from_device(a_size,a5,d_a5,"copy5");

   //puts("in devcpy0.51");
   //exit(0);

   
   err[5] = cudaMemcpy(d_a6,tmpLayerP->Jz,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   err[6] = cudaMemcpy(d_a7,tmpLayerC->JxBeam,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   err[7] = cudaMemcpy(d_a8,tmpLayerC->RhoBeam,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   err[8] = cudaMemcpy(d_a9,tmpLayerP->Rho,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 

#ifdef CUDA_WRAP_PARALLEL_DEBUG		 
   for(i = 0;i < 9;i++)
   {
       printf("copy arrays device error %d err %d \n",i,err[i]);
   }    
#endif
   
   return 0;
}

int CUDA_WRAP_copyArraysDeviceC(
int a_size,
double *d_a1,   // d_rEx,d_rEy,d_rEz,d_rJx,d_rJy,d_rJz,d_rJxBeam,d_rRhoBeam,d_rRho
double *d_a2,
double *d_a3,
double *d_a4,
double *d_a5,
double *d_a6
)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
   int err[10],i;
   //puts("BEGIN COPY ======================================================================================");
   //exit(0);
   err[0] = cudaMemcpy(d_a1,tmpLayerC->Ex,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
//   compare_vector_from_device(a_size,a1,d_a1,"copy1");
 //  puts("in devcpy0.5");
//   exit(0);

   err[1] = cudaMemcpy(d_a2,tmpLayerC->Ey,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a2,d_a2,"copy2");
   err[2] = cudaMemcpy(d_a3,tmpLayerC->Ez,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a3,d_a3,"copy3");
   err[3] = cudaMemcpy(d_a4,tmpLayerC->Bx,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 
   //compare_vector_from_device(a_size,a4,d_a4,"copy4");
   err[4] = cudaMemcpy(d_a5,tmpLayerC->By,a_size*sizeof(double),cudaMemcpyDeviceToDevice);  
   //compare_vector_from_device(a_size,a5,d_a5,"copy5");

   //puts("in devcpy0.51");
   //exit(0);

   
   err[5] = cudaMemcpy(d_a6,tmpLayerP->Bz,a_size*sizeof(double),cudaMemcpyDeviceToDevice); 

#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   for(i = 0;i < 6;i++)
   {
       printf("copy arrays deviceC error %d err %d \n",i,err[i]);
   }    
#endif

   
   return 0;
}


int CUDA_WRAP_allocLayerOnHost(cudaLayer **hl,int Ny,int Nz,int Np)
{
   double *d_Ex,*d_Ey,*d_Ez,*d_Bx,*d_By,*d_Bz,*d_Jx,*d_Jy,*d_Jz,*d_Rho;
   cudaLayer *l,*h_l = (cudaLayer*)malloc(sizeof(cudaLayer));
   beamParticle *p;
   //cudaMalloc((void**)&l,sizeof(cudaLayer));
   
/*#ifdef CUDA_WRAP_FFTW_ALLOWED
   return 0;
#endif*/
   
#ifdef CUDA_WRAP_PARALLEL_DEBUG	     
   printf("in alloc layer Ny %d Nz %d Np %d =============================================================\n",Ny,Nz,Np);     
#endif   

   int err = cudaGetLastError();
   err = cudaMalloc(&d_Ex,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Ex %d \n",err);
   err = cudaMalloc(&d_Ey,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Ey %d \n",err);
   err = cudaMalloc(&d_Ez,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Ez %d \n",err);
   err = cudaMalloc(&d_Bx,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Bx %d \n",err);
   err = cudaMalloc(&d_By,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error By %d \n",err);
   err = cudaMalloc(&d_Bz,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Bz %d \n",err);
   err = cudaMalloc(&d_Jx,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Jx %d \n",err);
   err = cudaMalloc(&d_Jy,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Jy %d \n",err);
   err = cudaMalloc(&d_Jz,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Jz %d \n",err);
   err = cudaMalloc(&d_Rho,sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error Rho %d \n",err);
   printf("CUDA_WRAP_allocLayerOnHost d_Rho %p \n",d_Rho);
   
   cudaMemset(d_Ex,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Ey,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Ez,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Bx,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_By,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Bz,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Jx,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Jy,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Jz,0,sizeof(double)*Ny*Nz);
   cudaMemset(d_Rho,0,sizeof(double)*Ny*Nz);
   
   err = cudaMalloc(&(h_l->JxBeam),sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error JxBeam %d \n",err);
   err = cudaMalloc(&(h_l->RhoBeam),sizeof(double)*Ny*Nz);
   if(err != cudaSuccess) printf("alloc layer error RhoBeam %d \n",err);

   cudaMemset(h_l->JxBeam,0,sizeof(double)*Ny*Nz);
   cudaMemset(h_l->RhoBeam,0,sizeof(double)*Ny*Nz);
   
   err = cudaMalloc(&p,Np*sizeof(beamParticle));
//#ifdef  CUDA_WRAP_PARALLEL_DEBUG   
   printf("alloc layer particles error %d size beamParticle %d p %p Np %d \n",err,sizeof(beamParticle),p,Np);
//#endif   

   h_l->Ex = d_Ex; 
   h_l->Ey = d_Ey; 
   h_l->Ez = d_Ez; 
   h_l->Bx = d_Bx; 
   h_l->By = d_By; 
   h_l->Bz = d_Bz; 
   h_l->Jx = d_Jx; 
   h_l->Jy = d_Jy; 
   h_l->Jz = d_Jz;
   h_l->Rho = d_Rho;
   printf("CUDA_WRAP_allocLayerOnHost  h_l->Rho %p    \n ",h_l->Rho);
   //exit(0);
   h_l->particles = p;
   printf(" h_l->particles %p  \n ", h_l->particles);
   CUDA_WRAP_printLayerParticles(h_l,"IN ALLOC");
   
   ////////////// TEST WRITE
   beamParticle bp_test;
   bp_test.f_X = -999.0;
   bp_test.f_Y = -1313.0;
   printf("%e %e %e %e %e %e %e\n ",bp_test.f_X,bp_test.f_Y,bp_test.f_Z,bp_test.f_Px,bp_test.f_Py,bp_test.f_Pz);

   printf(" h_l->particles %p &bp_test  %p Np %d \n ",h_l->particles,&bp_test,Np);
   //exit(0);
   
   cudaMemcpy(h_l->particles,&bp_test,sizeof(beamParticle)  ,cudaMemcpyHostToDevice);
   printf("after test write\n");
   CUDA_WRAP_printLayerParticles(h_l,"IN TEST");
   //exit(0);
   /////////////////////////
   
   h_l->Ny = Ny;
   h_l->Nz = Nz;
   h_l->Np = Np;
   
//   cudaMemcpy(l,h_l,sizeof(cudaLayer),cudaMemcpyHostToDevice);
   
   *hl = h_l;
   
   return 0;
}

// int CUDA_WRAP_allocHostLayer(cudaLayer **h_l,int Ny,int Nz,int Np)
// {
//    double *d_Ex,*d_Ey,*d_Ez,*d_Bx,*d_By,*d_Bz,*d_Jx,*d_Jy,*d_Jz,*d_Rho;
//    cudaLayer *l;
//    printf("in CUDA_WRAP_allocHostLayer\n   ");
//    //exit(0);
//    *h_l = (cudaLayer*)malloc(sizeof(cudaLayer));
//
//
//    beamParticle *p;
//    //cudaMalloc((void**)&l,sizeof(cudaLayer));
//
//
// //#ifdef CUDA_WRAP_PARALLEL_DEBUG
//    printf("in alloc host layer Ny %d Nz %d Np %d =============================================================\n",Ny,Nz,Np);
// //#endif
//
//    (*h_l)->Ex = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Ey = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Ez = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Bx = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->By = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Bz = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Jx = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Jy = (double *)malloc(sizeof(double)*Ny*Nz);
//    printf("in alloc layer Jy %p \n", (*h_l)->Jy);
//    (*h_l)->Jz = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->Rho = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->particles = (beamParticle *)malloc(sizeof(double)*Np);
//    (*h_l)->fftRhoBeamHydro = (double *)malloc(sizeof(double)*Ny*Nz);
//    (*h_l)->fftJxBeamHydro  = (double *)malloc(sizeof(double)*Ny*Nz);
//    CUDA_WRAP_printLayerParticles(*h_l,"IN ALLOC");
//
//    (*h_l)->Ny = Ny;
//    (*h_l)->Nz = Nz;
//    (*h_l)->Np = Np;
//
// //   cudaMemcpy(l,h_l,sizeof(cudaLayer),cudaMemcpyHostToDevice);
//
//    //*hl = h_l;
//
//    return 0;
// }


int CUDA_WRAP_copyLayerFrom3D(int iLayer,int Ny,int Nz,int Np,cudaLayer **h_cl)
{

    printf("in  CUDA_WRAP_copyLayerFrom3D\n  ");
    //exit(0); 
   
// #ifdef CUDA_WRAP_FFTW_ALLOWED
//    CUDA_WRAP_allocHostLayer(h_cl,Ny,Nz,Np);
//      return 0;
// #else
//     CUDA_WRAP_allocLayerOnHost(h_cl,Ny,Nz,Np);     
// #endif
     printf("h_cl %p\n",h_cl);
     printf("h_cl->Jy %p \n ",(*h_cl)->Jy);
     printf("h_cl->Rho %p \n ",(*h_cl)->Rho);



    
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Rho3D,    (*h_cl)->Rho);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Ex3D,     (*h_cl)->Ex);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Ey3D,     (*h_cl)->Ey);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Ez3D,     (*h_cl)->Ez);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Bx3D,     (*h_cl)->Bx);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_By3D,     (*h_cl)->By);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Bz3D,     (*h_cl)->Bz);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jx3D,     (*h_cl)->Jx);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jy3D,     (*h_cl)->Jy);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jz3D,     (*h_cl)->Jz);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_RhoBeam3D,(*h_cl)->RhoBeam);
     CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_JxBeam3D, (*h_cl)->JxBeam);
     
    printf("copyLayer Ny %d Nz %d Np %d \n",(*h_cl)->Ny,(*h_cl)->Nz,(*h_cl)->Np);
//    exit(0);
    return 0;
}

int CUDA_WRAP_copyLayerDeviceToDevice(int Ny,int Nz,int Np,cudaLayer *h_dl,cudaLayer *h_sl)
{ 
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho,*t;
   beamParticle *bp;
   int np = 0;
   
/*   cudaMalloc(&t,sizeof(double)*Ny*Nz);
   h_dl->Ex = t;
   cudaMalloc(&(h_dl->Ey),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Ez),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Bx),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->By),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Bz),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Jy),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Jx),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Jz),sizeof(double)*Ny*Nz);
   cudaMalloc(&(h_dl->Rho),sizeof(double)*Ny*Nz);
*/
   int err = cudaGetLastError();
   t = h_sl->Ex;
   cudaMemcpy(h_dl->Ex,t,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Ey,h_sl->Ey,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Ez,h_sl->Ez,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Bx,h_sl->Bx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->By,h_sl->By,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Bz,h_sl->Bz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Jx,h_sl->Jx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Jy,h_sl->Jy,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Jz,h_sl->Jz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(h_dl->Rho,h_sl->Rho,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);

  // h_dl->Np = Np;
   h_dl->Ny = Ny;
   h_dl->Nz = Nz;   

  // cudaMalloc(&(h_dl->particles),Np*sizeof(beamParticle));

 //  cudaMemcpy(h_dl->particles,h_sl->particles,Np*sizeof(beamParticle),cudaMemcpyDeviceToDevice);
   
   return np;
}

int CUDA_WRAP_storeArraysToDeviceC(
int Ny,int Nz,
double *d_Ex,
double *d_Ey,
double *d_Ez,
double *d_Bx,
double *d_By,
double *d_Bz,
double *d_rRho
)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
   cudaMemcpy(tmpLayerC->Ex,d_Ex,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
#ifndef CPU_COMPUTING   
   CUDA_DEBUG_printDdevice_matrix(Ny,Nz,tmpLayerC->Ex,"Ex:copy");
#endif   
   cudaMemcpy(tmpLayerC->Ey,d_Ey,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(tmpLayerC->Ez,d_Ez,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(tmpLayerC->Bx,d_Bx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(tmpLayerC->By,d_By,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(tmpLayerC->Bz,d_Bz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy(tmpLayerC->Rho,d_rRho,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
  
    return 0;
}

int CUDA_WRAP_storeArraysToDevice(
int Ny,int Nz,
double *d_Ex,
double *d_Ey,
double *d_Ez,
double *d_Bx,
double *d_By,
double *d_Bz,
double *d_Jx,
double *d_Jy,
double *d_Jz,
double *d_Rho
)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif  
   int err[10],i;
   
   err[0] = cudaMemcpy(tmpLayerC->Ex,d_Ex,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
#ifndef CPU_COMPUTING   
   CUDA_DEBUG_printDdevice_matrix(Ny,Nz,tmpLayerC->Ex,"Ex:copy");
#endif   
   err[1] = cudaMemcpy(tmpLayerC->Ey,d_Ey,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[2] = cudaMemcpy(tmpLayerC->Ez,d_Ez,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[3] = cudaMemcpy(tmpLayerC->Bx,d_Bx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[4] = cudaMemcpy(tmpLayerC->By,d_By,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[5] = cudaMemcpy(tmpLayerC->Bz,d_Bz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[6] = cudaMemcpy(tmpLayerC->Jx,d_Jx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[7] = cudaMemcpy(tmpLayerC->Jy,d_Jy,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[8] = cudaMemcpy(tmpLayerC->Jz,d_Jz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   err[9] = cudaMemcpy(tmpLayerC->Rho,d_Rho,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);  

#ifdef CUDA_WRAP_PARALLEL_DEBUG		 
  for(i = 0;i < 10;i++)
  {
      printf("store arrays errors %d err %d \n",i,err[i]);
  }
#endif
  
    return 0;
}



void __global__ interpolateLayerKernel(int iSplit,int NxSplit,int Ny,int Nz,double *RhoBeamC,double *RhoBeamP,double *JxBeamC,double *JxBeamP,
                                                                            double *RhoBeamL,double *RhoBeamR,double *JxBeamL,double *JxBeamR
)
{
//     cuPrintf("in inter\n");
//     return;
     unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
     unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
     unsigned int n = k*Ny + j;
     
     
     RhoBeamC[n] = (RhoBeamL[n]*(NxSplit - iSplit) + RhoBeamR[n]*iSplit)/((double)NxSplit);
     JxBeamC[n]  = (JxBeamL[n]*(NxSplit - iSplit) +  JxBeamR[n]*iSplit)/((double)NxSplit);

     RhoBeamP[n] = (RhoBeamL[n]*(NxSplit - iSplit - 1.0) + RhoBeamR[n]*(iSplit + 1.0))/((double)NxSplit);
     JxBeamP[n]  = (JxBeamL[n]*(NxSplit - iSplit  - 1.0) + JxBeamR[n]*(iSplit + 1.0))/((double)NxSplit);
    // printf("inter %5d %5d %25.15e %25.15e %25.15e \n",j,k,JxBeamL[n],JxBeamP[n],JxBeamR[n]);
     
}


int CUDA_WRAP_interpolateLayers(int iSplit,int NxSplit,int Ny,int Nz,cudaLayer *h_cl,cudaLayer *h_pl,cudaLayer *h_left,cudaLayer *h_right)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
   dim3 dimGrid(Ny/16,Nz/16),dimBlock(16,16); 
 
   int err0 = cudaGetLastError();
   
#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("before interpolate error %d \n",err0);    
#endif   

   //cudaPrintfInit();
 
   interpolateLayerKernel<<<dimGrid, dimBlock>>>(iSplit,NxSplit,Ny,Nz,h_cl->RhoBeam,h_pl->RhoBeam,h_cl->JxBeam,h_pl->JxBeam,
						                      h_left->RhoBeam,h_right->RhoBeam,h_left->JxBeam,h_right->JxBeam);
						                      
   //cudaPrintfDisplay(stdout, true);
   //cudaPrintfEnd();   
   int err11 = cudaGetLastError();

#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("after interpolate error %d \n",err11);    
#endif   

   return 0;
}

int CUDA_WRAP_printLayerParticles(cudaLayer *h_l,char *s)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
#ifndef CUDA_WRAP_PRINT_10_PARTICLES
    return 1;
#endif    
    puts("in print part");
    int err_last = cudaGetLastError();
    beamParticle *bp = (beamParticle *)malloc(LAYER_PARTICLE_PRINT_NUMBER*sizeof(beamParticle));
    
    int err = cudaMemcpy(bp,h_l->particles,LAYER_PARTICLE_PRINT_NUMBER*sizeof(beamParticle),cudaMemcpyDeviceToHost);
    
#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("print layer particles error %d last error %d\n",err,err_last);    
   if(err != cudaSuccess) exit(0);
#endif   
    
    
    for(int i = 0;i < LAYER_PARTICLE_PRINT_NUMBER;i++)
    {
        printf("%s %25.15e %25.15e %25.15e %e %e %e \n",s,bp[i].f_X,bp[i].f_Y,bp[i].f_Z,bp[i].f_Px,bp[i].f_Py,bp[i].f_Pz);
    }
    
    free(bp);
    
    return 0;
}

int CUDA_WRAP_printParticles(beamParticle *d_bp,char *s)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
#ifndef CUDA_WRAP_PRINT_BEAM_10_PARTICLES
    return 1;
#endif    
    int N = 10;
    printf("rank %d in PRINT PARTICLES \n",GetRank());
    //return 0;
  
    beamParticle *bp = (beamParticle *)malloc(N*sizeof(beamParticle));
    
    int err = cudaMemcpy(bp,d_bp,N*sizeof(beamParticle),cudaMemcpyDeviceToHost);

#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("print particles error %d \n",err);    
#endif   
    
    for(int i = 0;i < N;i++)
    {
        printf("%s %25.15e %25.15e %25.15e %e %e %e \n",s,bp[i].f_X,bp[i].f_Y,bp[i].f_Z,bp[i].f_Px,bp[i].f_Py,bp[i].f_Pz);
    }
    
    return 0;
}





int CUDA_WRAP_copyLayerParticles(cudaLayer *h_dst,cudaLayer *h_src)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
    int err = cudaMemcpy(h_dst->particles,h_src->particles,h_src->Np*sizeof(beamParticle),cudaMemcpyDeviceToDevice);
#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("copy layer particles error %d \n",err);    
#endif   
    
    h_dst->Np = h_src->Np;
    
    return 0;
}

__global__ void addOne(double *dst,double *src,int Ny)
{
   unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
   unsigned int k = blockIdx.y * blockDim.y + threadIdx.y;
   unsigned int n = k*Ny + j;
   
   dst[n] = src[n]+1.0;
}

int CUDA_WRAP_copyArraysDeviceIterate(int l_My,int l_Mz,double *d_rJx,double *d_rJy,double *d_rJz,double *d_rJxBeam,double *d_rRhoBeam,double *d_rRho)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
    dim3 dimGrid(l_My/16,l_Mz/16),dimBlock(16,16);
    int err0,err1,err2,err3,err4;
    
#ifndef CUDA_WRAP_FFTW_ALLOWED
    
    addOne<<<dimGrid,dimBlock>>>(d_rRho,tmpLayerC->Rho,l_My);
    
    err0 = cudaMemcpy(d_rJx,tmpLayerC->Jx,          l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);
    err1 = cudaMemcpy(d_rJy,tmpLayerC->Jy,          l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);
    err2 = cudaMemcpy(d_rJz,tmpLayerC->Jz,          l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);
    err3 = cudaMemcpy(d_rJxBeam,tmpLayerC->JxBeam,  l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);
    err4 = cudaMemcpy(d_rRhoBeam,tmpLayerC->RhoBeam,l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);

#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
   printf("copy arrays iterate error 0 %d 1 %d 2 %d 3 %d 4 %d \n",err0,err1,err2,err3,err4);    
#endif
   
#endif   
    
    return 0;
}


int CUDA_WRAP_clearCurrents(int Ny,int Nz,int rho_flag)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif
     
    cudaMemset(tmpLayerC->Jx,0,Ny*Nz*sizeof(double));
    cudaMemset(tmpLayerC->Jy,0,Ny*Nz*sizeof(double));
    cudaMemset(tmpLayerC->Jz,0,Ny*Nz*sizeof(double));
    if(rho_flag)
    {
       cudaMemset(tmpLayerC->Rho,0,Ny*Nz*sizeof(double));
    }
}


int cuLayerPrintCentre(cudaLayer *h_cl,int iLayer,Mesh *mesh,Cell *p_CellArray,char *where)
{
    int Ny,Nz,Np;
    double ex0,ey0,ez0,bx0,by0,bz0,jx0,jy0,jz0,rho0,jxb0,rhb0;
    beamParticle p;
    int err;
    
//#ifndef CUDA_WRAP_PRINT_CENTRE    
//    return 0;
//#endif

    
    
//    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,h_cl->Ey,"Ey");

// #ifndef CUDA_WRAP_FFTW_ALLOWED
    Ny = h_cl->Ny;
    Nz = h_cl->Nz;
    Np = h_cl->Np;

    printf("in  cuLayerPrintCentre %s %s \n ",where,"C-loop 3");
    printf("Ny %d Nz %d Np %d \n",Ny,Nz,Np);

    
    cudaMemcpy(&ex0,h_cl->Ex+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&ey0,h_cl->Ey+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&ez0,h_cl->Ez+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);

    cudaMemcpy(&bx0,h_cl->Bx+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&by0,h_cl->By+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&bz0,h_cl->Bz+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);

    cudaMemcpy(&jx0,h_cl->Jx+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&jy0,h_cl->Jy+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&jz0,h_cl->Jz+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);

    cudaMemcpy(&rho0,h_cl->Rho+Ny*Nz/2+200,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&jxb0,h_cl->JxBeam+Ny*Nz/2+Ny/2,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(&rhb0,h_cl->RhoBeam+Ny*Nz/2+Ny/2,sizeof(double),cudaMemcpyDeviceToHost);
    
    //int err = cudaMemcpy(&p,h_cl->particles,sizeof(beamParticle),cudaMemcpyDeviceToHost);
// #else
    printf("IN CPU PART layer %d \n",iLayer);
    
    Ny = mesh->GetMy() + 1;
    Nz = mesh->GetMz() + 1;
//    Np = h_cl->Np;
    printf("IN CPU PART layer %d Ny %3d Nz %3d \n",iLayer,Ny,Nz);
    
    //Cell ccc;
    //Particle *p_host;
    long nccc;
    if(iLayer < 0)
    {  
        nccc = mesh->GetNyz(Ny/2,Nz/2 + 5);
    }
    else
    {
        nccc = mesh->GetN(iLayer, Ny/2,Nz/2 + 5);
    }
    Cell &ccc = p_CellArray[nccc];	    
    Particle *p_host  = ccc.GetParticles();
    
    
    ex0 = ccc.GetEx();
    ey0 = ccc.GetEy();
    ez0 = ccc.GetEz();
    
    bx0 = ccc.GetBx();
    by0 = ccc.GetBy();
    bz0 = ccc.GetBz();
    
    jx0 = ccc.GetJx();
    jy0 = ccc.GetJy();
    jz0 = ccc.GetJz();
    
    rho0 = ccc.GetDens();
    rhb0 = ccc.GetRhoBeam();
    jxb0 = ccc.GetJxBeam();
    
    if(p_host != NULL)
    {
       p.f_X  = p_host->f_X;
       p.f_Y  = p_host->f_Y;
       p.f_Z  = p_host->f_Z;
       p.f_Px = p_host->f_Px;
       p.f_Py = p_host->f_Py;
       p.f_Pz = p_host->f_Pz;
    }
    
// #endif
    
    printf("rank %d Layer %3d %s============================================================================================================================\n",
	   GetRank(),iLayer,where);
    
    printf("E   (%15.5e,%15.5e,%15.5e) B (%15.5e,%15.5e,%15.5e) J (%15.5e,%15.5e,%15.5e) \n",ex0,ey0,ez0,bx0,by0,bz0,jx0,jy0,jz0);
    printf("Rho %15.5e JxB %25.15e RhoB %25.15e \n",rho0,jxb0,rhb0);
    printf("particle0 err %d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e \n",err,p.f_X,p.f_Y,p.f_Z,p.f_Px,p.f_Py,p.f_Pz);
    
    for(int i = 0;i < Ny;i++)
    {
        for(int j = 0;j < Nz;j++)
	{
	    long nccc = mesh->GetN(iLayer, i,j);
            Cell &ccc = p_CellArray[nccc];	
            //printf("%d %d %d %e %e\n",iLayer,i,j,ccc.GetRhoBeam(),ccc.GetEy());	  
	}
    }
    printf("rank %d Layer %3d END==========================================================================================================================\n",GetRank(),iLayer);

    return 0;
}

int CUDA_WRAP_printBeamDensity3D(Mesh *p_M,int step,char *where)
{
#ifdef CUDA_WRAP_BEAM_3D_DENSITY_PRINT	
    char fname[100];
    FILE *f;
    long Mx,My,Mz,dMx,dMy,dMz;
    
    p_M->GetSizes(Mx,My,Mz,dMx,dMy,dMz);
    
    sprintf(fname,"beam_%s_%03d_rank%03d.dat",where,step,GetRank());
    
    f = fopen(fname,"wt");

    for (int i = -dMx; i < Mx + dMx - 1; i++) {
      for (int k = -dMz; k < Mz + dMz - 1; k++) {
         for (int j = -dMy; j < My + dMy - 1; j++) {
            Cell& ccc = p_M->GetCell(i,j,k);
	    fprintf(f,"%3d %3d %3d %25.15e %25.15e\n",i,j,k,ccc.GetJxBeam(),ccc.GetRhoBeam());
/*	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();
	    
	    ccc_c.SetFields(fds);
*/	    
         }
      }
    }
    fclose(f);
#endif	    

   return 0; 
}

int CUDA_WRAP_printBeamParticles(Mesh *p_M,int step,char *where)
{
#ifdef CUDA_WRAP_BEAM_PARTICLES_PRINT	
    char fname[100];
    FILE *f;
    long Mx,My,Mz,dMx,dMy,dMz;
    
    p_M->GetSizes(Mx,My,Mz,dMx,dMy,dMz);
    
    sprintf(fname,"beamParticles_%s_%03d_rank%03d.dat",where,step,GetRank());
    
    f = fopen(fname,"wt");

    for (int i = -dMx; i < Mx + dMx - 1; i++) {
      for (int k = -dMz; k < Mz + dMz - 1; k++) {
         for (int j = -dMy; j < My + dMy - 1; j++) {
            Cell& ccc = p_M->GetCell(i,j,k);
	    Particle *p = ccc.GetBeamParticles();
	    int num = 0;
	    while(p)
	    {
	       fprintf(f,"%3d %3d %3d %d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",i,j,k,num,p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Pz,p->f_Pz,p->f_Weight,p->f_Q2m);
	       
	       p = p->p_Next;
	       num++;
	    }
	     
/*	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();
	    
	    ccc_c.SetFields(fds);
*/	    
         }
      }
    }
    fclose(f);   
#endif	 
   

   return 0; 
}

int get2D_index(cudaLayer *cl,int i,int j)
{
	return cl->Ny*i + j;
}

int CUDA_WRAP_printPlasmaParticles(Mesh *p_M,int step,char *where)
{
#ifdef CUDA_WRAP_BEAM_PARTICLES_PRINT
    char fname[100];
    FILE *f;
    long Mx,My,Mz,dMx,dMy,dMz;

    p_M->GetSizes(Mx,My,Mz,dMx,dMy,dMz);

    sprintf(fname,"plasmaParticles_%s_%03d_rank%03d.dat",where,step,GetRank());

    f = fopen(fname,"wt");

    for (int i = -dMx; i < Mx + dMx - 1; i++) {
      for (int k = -dMz; k < Mz + dMz - 1; k++) {
         for (int j = -dMy; j < My + dMy - 1; j++) {
            Cell& ccc = p_M->GetCell(i,j,k);
	    Particle *p = ccc.GetParticles();
	    int num = 0;
	    while(p)
	    {
	       fprintf(f,"%3d %3d %3d %5d %2d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",i,j,k,num,p->GetSort(),
                   p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Pz,p->f_Pz,p->f_Weight,p->f_Q2m);

	       p = p->p_Next;
	       num++;
	    }

	     

/*	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();

	    ccc_c.SetFields(fds);
*/
         }
      }
    }
    fclose(f);
#endif


   return 0;
}
