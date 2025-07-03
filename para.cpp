
// #include <mpi.h>
#include <stdlib.h>


#include "run_control.h"

#include "CUDA_WRAP/cuLayers.h"
#include "CUDA_WRAP/paraLayers.h"
#include "CUDA_WRAP/sendBeam.h"

#include "para.h"



#include "CUDA_WRAP/cuCell.h"


int rank,size,locMx;
int mpi_initialized = 0;
double X_min,X_max;

int beamNp = 0;


int getBeamNp()
{
    return 0;
}

int setBeamNp(int n)
{
    beamNp = n;
}


int GetRank(){return 0;}
int GetSize(){return 1;}

double GetXmin(){return 0.0;}
double GetXmax(){return X_max;}

int ParallelExit()
{
//    MPI_Barrier(MPI_COMM_WORLD);
//    MPI_Finalize();
   exit(0);
   
   return 0;
}


int ParallelInit(int argc,char *argv[])
{
//     MPI_Init(&argc,&argv);
//
//     MPI_Comm_size(MPI_COMM_WORLD,&size);
//     MPI_Comm_rank(MPI_COMM_WORLD,&rank);
//
//     printf("rank %d size %d \n",GetRank(),size);
    
  
    return 0;
}

int SetXSize(long *l_Mx,double x_size)
{
//    *l_Mx = (*l_Mx - 1)/size + 1;
  
    X_min = 0.0;
    X_max = (x_size/(double)size);
/*#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d l_Mx %d min %e max %e \n",rank,*l_Mx,X_min,X_max);
#endif   */
}

int Set_l_Mx(int *l_Mx)
{
//     *l_Mx = (*l_Mx - 1)/size + 1;
//
// //    X_min = (x_size/(double)size)*rank;
// //    X_max = (x_size/(double)size)*(rank+1);
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     printf("rank %d l_Mx \n",rank,*l_Mx);
// #endif
}

int ParallelFinalize()
{
//    return MPI_Finalize();
    return 0;
}


// int PackLayer(cudaLayer *h_l,double **lp,int Ny,int Nz,int Np)
// {
//     double *pack;
//     int size = Ny*Nz,sizep = Np;
//
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     printf("in Pack Ny %d Nz %d Np %d \n",Ny,Nz,Np);
// #endif
//
//     pack = (double*)malloc(sizeof(double)*(Ny*Nz*LAYER_ATTRIBUTE_NUMBER+Np*BEAM_PARTICLE_ATTRIBUTE_NUMBER));
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     puts("in Pack A");
// #endif
//
//     for(int i = 0;i < Ny*Nz;i++)
//     {
//         pack[i]         = h_l->Bx[i];
// 	pack[i+size]    = h_l->By[i];
// 	pack[i+2*size]  = h_l->Bz[i];
// 	pack[i+3*size]  = h_l->Ex[i];
// 	pack[i+4*size]  = h_l->Ey[i];
// 	pack[i+5*size]  = h_l->Ez[i];
// 	pack[i+6*size]  = h_l->Jx[i];
// 	pack[i+7*size]  = h_l->Jy[i];
// 	pack[i+8*size]  = h_l->Jz[i];
// 	pack[i+9*size]  = h_l->Rho[i];
// 	pack[i+10*size] = h_l->fftJxBeamHydro[i];
// 	pack[i+11*size] = h_l->fftRhoBeamHydro[i];
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//         printf("rank %d in Pack A-B i E (%e,%e,%e) B (%e,%e,%e) J (%e,%e,%e) Rho %e hydro %e %e   \n",GetRank(),i,
//                h_l->Ex[i],h_l->Ey[i],h_l->Ez[i],
//                h_l->Bx[i],h_l->By[i],h_l->Bz[i],
//                h_l->Jx[i],h_l->Jy[i],h_l->Jz[i],
//                h_l->Rho[i],pack[i+10*size],pack[i+11*size]
//         );
// #endif
//
//     }
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     printf("in Pack B size %d sizep %d \n",size,sizep);
// #endif
//
//     for(int i = 0;i < Np;i++)
//     {
//         pack[i+12*size]                    = h_l->particles[i].f_Px;
//         ////printf("Pack rank %d i %d index %d \n",rank,i,i+12*size +    sizep);
//         pack[i+12*size +    sizep]         = h_l->particles[i].f_Py;
//         pack[i+12*size +  2*sizep]         = h_l->particles[i].f_Pz;
//         pack[i+12*size +  3*sizep]         = h_l->particles[i].f_X;
//         pack[i+12*size +  4*sizep]         = h_l->particles[i].f_Y;
//         pack[i+12*size +  5*sizep]         = h_l->particles[i].f_Z;
//         pack[i+12*size +  6*sizep]         = h_l->particles[i].f_Q2m;
//         pack[i+12*size +  7*sizep]         = h_l->particles[i].f_Weight;
//         pack[i+12*size +  8*sizep]         = h_l->particles[i].i_X;
//         pack[i+12*size +  9*sizep]         = h_l->particles[i].i_Y;
//         pack[i+12*size + 10*sizep]         = h_l->particles[i].i_Z;
//         pack[i+12*size + 11*sizep]         = h_l->particles[i].isort;
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//         printf("Pack vorC rank %5d i %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",rank,i,h_l->particles[i].f_X,h_l->particles[i].f_Y,h_l->particles[i].f_Z,
// 	                                                                                      h_l->particles[i].f_Px,h_l->particles[i].f_Py,h_l->particles[i].f_Pz
// 	);
// #endif
//     }
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     puts("in Pack C");
// #endif
//
//     *lp = pack;
//
//     return (Ny*Nz*LAYER_ATTRIBUTE_NUMBER+Np*BEAM_PARTICLE_ATTRIBUTE_NUMBER);
//
// }
//
// int UnpackLayer(cudaLayer *d_l,double *lp,int Ny,int Nz,int Np)
// {
//     double *pack;
//     int size = Ny*Nz,sizep = Np;
//     static cudaLayer *h_l;
//     static int first = 1;
//
//
//     if(first == 1)
//     {
//        CUDA_WRAP_createNewLayer(&h_l,d_l);
//        first = 0;
//     }
//
//     //printf("rank %d Recv LAYER %d %d %d  \n",rank,h_l->Ny,h_l->Nz,h_l->Np);
// //    CUDA_WRAP_fillLayer(h_l,Ny,Nz,Np);
//
//
//     //printf("rank %d in UnPack \n",GetRank());
//     //pack = (double*)malloc(sizeof(double)*(Ny*Nz*LAYER_ATTRIBUTE_NUMBER+Np*BEAM_PARTICLE_ATTRIBUTE_NUMBER));
//     pack = lp;
//     //printf("rank %d in UnPack A size %d Ny %d Nz %d Np %d \n",GetRank(),size,Ny,Nz,Np);
//
//     for(int i = 0;i < Ny*Nz;i++)
//     {
//         h_l->Bx[i]      = pack[i];
// 	h_l->By[i]      = pack[i+size];
// //	//printf("Bz %e \n",h_l->Bz[i]);
// //	//printf("pack+2 %e \n",pack[i+2*size]);
// //	h_l->Bz[i]      = 1.0;//pack[i+2*size];
// 	h_l->Bz[i]      = pack[i+2*size];
// 	h_l->Ex[i]      = pack[i+3*size];
// 	h_l->Ey[i]      = pack[i+4*size];
//         h_l->Ez[i]      = pack[i+5*size];
// 	h_l->Jx[i]      = pack[i+6*size];
// 	h_l->Jy[i]      = pack[i+7*size];
// 	h_l->Jz[i]      = pack[i+8*size];
// 	h_l->Rho[i]     = pack[i+9*size];
// 	h_l->fftJxBeamHydro[i] = pack[i+10*size];
//         h_l->fftRhoBeamHydro[i]  = pack[i+11*size];
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//         printf("rank %d in UnPack A-B %d E (%e,%e,%e) B (%e,%e,%e) J (%e,%e,%e) Rho %e %e %e\n",GetRank(),i,
//                h_l->Ex[i],h_l->Ey[i],h_l->Ez[i],
//                h_l->Bx[i],h_l->By[i],h_l->Bz[i],
//                h_l->Jx[i],h_l->Jy[i],h_l->Jz[i],
//                h_l->Rho[i],h_l->fftJxBeamHydro[i],h_l->fftRhoBeamHydro[i]
//         );
// #endif
//
//     }
//
//
//     //printf("rank %d in UnPack B sizep %d \n",GetRank(),sizep);
// //    exit(0);
//
//     for(int i = 0;i < Np;i++)
//     {
//         ////printf("Unpack vorC0 rank %d i %d \n",rank,i);
//
//         h_l->particles[i].f_Px = pack[i+12*size];
// //        printf("Unpack vorC01 rank %d i %d index %d size %d sizep %d \n",rank,i,i+12*size +    sizep,size,sizep);
//         h_l->particles[i].f_Py = pack[i+12*size +    sizep];
// //        printf("Unpack vorC02 rank %d i %d \n",rank,i);
//         h_l->particles[i].f_Pz = pack[i+12*size +  2*sizep];
// //        printf("Unpack vorC03 rank %d i %d \n",rank,i);
//         h_l->particles[i].f_X  = pack[i+12*size +  3*sizep];
// //        printf("Unpack vorC04 rank %d i %d \n",rank,i);
//         h_l->particles[i].f_Y  = pack[i+12*size +  4*sizep];
// //        printf("Unpack vorC05 rank %d i %d \n",rank,i);
//         h_l->particles[i].f_Z  = pack[i+12*size +  5*sizep];
//         //printf("Unpack vorC1 rank %d i %d Np %d \n",rank,i,Np);
//
//         h_l->particles[i].f_Q2m = pack[i+12*size +  6*sizep];
//         h_l->particles[i].f_Weight =      pack[i+12*size +  7*sizep];
//         h_l->particles[i].i_X      = (int)pack[i+12*size +  8*sizep];
//         h_l->particles[i].i_Y      = (int)pack[i+12*size +  9*sizep];
//         h_l->particles[i].i_Z      = (int)pack[i+12*size + 10*sizep];
//         h_l->particles[i].isort    = (int)pack[i+12*size + 11*sizep];
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//         printf("Unpack vorC rank %5d i %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",rank,i,h_l->particles[i].f_X,h_l->particles[i].f_Y,h_l->particles[i].f_Z,
// 	                                                                                      h_l->particles[i].f_Px,h_l->particles[i].f_Py,h_l->particles[i].f_Pz
// 	);
// #endif
//     }
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     printf("rank %d in UnPack C \n",GetRank());
// #endif
// //    exit(0);
// #ifndef CUDA_WRAP_FFTW_ALLOWED
//     CUDA_WRAP_copyToLayerOnDevice(d_l,h_l);
// #else
//     d_l->Ex = h_l->Ex;
//     d_l->Ey = h_l->Ey;
//     d_l->Ez = h_l->Ez;
//
//     d_l->Bx = h_l->Bx;
//     d_l->By = h_l->By;
//     d_l->Bz = h_l->Bz;
//
//     d_l->Jx = h_l->Jx;
//     d_l->Jy = h_l->Jy;
//     d_l->Jz = h_l->Jz;
//
//     d_l->Rho = h_l->Rho;
//     d_l->fftJxBeamHydro  = h_l->fftJxBeamHydro;
//     d_l->fftRhoBeamHydro = h_l->fftRhoBeamHydro;
//     d_l->particles = h_l->particles;
//
// #endif
// #ifdef CUDA_WRAP_PARALLEL_DEBUG
//     printf("rank %d in UnPack D 1st %e \n",GetRank(),d_l->particles[0].f_Y);
// #endif
//
// //    exit(0);
//
//     return (Ny*Nz*LAYER_ATTRIBUTE_NUMBER+Np*BEAM_PARTICLE_ATTRIBUTE_NUMBER);
//
// }

int SendLayer(cudaLayer *d_l,int Ny,int Nz,int Np)
{
    cudaLayer *h_l;
    double *buf;
    int buf_size = 1,dest;

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send layer \n",rank);
#endif    
    
    
    if (GetRank() == 0) return 0;
    
#ifndef PARALLEL_ONLY     

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send layer A0 \n",rank);
#endif    

#ifndef CPU_COMPUTING   
    CUDA_WRAP_copyLayerFromDevice(&h_l,d_l);
#else
    h_l = d_l;
#endif    
    
//     buf_size = PackLayer(h_l,&buf,Ny,Nz,Np);
#endif
#ifdef PARALLEL_ONLY     
    buf_size = 10;
    buf = (double *)malloc(buf_size*sizeof(double)); 
#endif

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send layer A \n",rank);
#endif    

    
    dest = GetRank() - 1;
//    MPI_Send(&buf_size,1,MPI_INTEGER,dest,SEND_BUFSIZE_TAG,MPI_COMM_WORLD);
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send layer B size %d \n",GetRank(),buf_size);
#endif    
//     MPI_Request   request;
       
    //MPI_Send(buf,buf_size,MPI_DOUBLE_PRECISION,dest,SEND_BUFFER_TAG,MPI_COMM_WORLD);
//    MPI_Isend(buf,buf_size,MPI_DOUBLE_PRECISION,dest,SEND_BUFFER_TAG,MPI_COMM_WORLD,&request);
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send layer C \n",GetRank());
#endif    

//    if(GetRank() == 1) exit(0);


    //SendBeamParticlesDown(&beamNp);

    return 0;
}

int ReceiveLayer(cudaLayer *h_result_l,int Ny,int Nz,int Np)
{
    cudaLayer *h_l,*h_d_l;
    double *buf;
    int buf_size,src;
//     MPI_Status status;

    return 0;

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in receive layer \n",GetRank());
    printf("begin recv %d %d %d \n ",h_result_l->Ny,h_result_l->Nz,h_result_l->Np);
#endif    
    
    if(GetRank() == size-1) 
    {
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("last proc extied"); 
#endif       
       return 0;
    }
      
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in receive layer A \n",GetRank());
#endif  
    src = GetRank() + 1;
//     MPI_Recv(&buf_size,1,MPI_INTEGER,src,SEND_BUFSIZE_TAG,MPI_COMM_WORLD,&status);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in receive layer B \n",GetRank());
#endif    
    
    buf = (double *)malloc(sizeof(double)*buf_size);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in receive layer C size  %d \n",GetRank(),buf_size);
#endif    
       
//     MPI_Recv(buf,buf_size,MPI_DOUBLE_PRECISION,src,SEND_BUFFER_TAG,MPI_COMM_WORLD,&status);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in receive layer D \n",GetRank());
#endif    

//    exit(0);

#ifndef PARALLEL_ONLY      

    Np = (buf_size - Ny*Nz*LAYER_ATTRIBUTE_NUMBER)/BEAM_PARTICLE_ATTRIBUTE_NUMBER;
    h_result_l->Np = Np;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("layerToFill in recv %d %d %d Np %d \n ",h_result_l->Ny,h_result_l->Nz,h_result_l->Np,Np);
#endif    
    
    
//     UnpackLayer(h_result_l,buf,Ny,Nz,Np);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d after Unpack 1st particleY %e   \n",GetRank(),h_result_l->particles[0].f_Y);
#endif    
    
//    CUDA_WRAP_copyToNewLayerOnDevice(&h_result_l,h_d_l);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d after copyToNewLayerOnDevice \n",GetRank());
#endif    
//    exit(0);
#endif    

   // SendBeamParticlesUp(&beamNp);


    return 0;
}

int SendBeamParticles(int *Np)
{
    beamParticle *h_send_up,*h_send_down,*h_recv_up,*h_recv_down;
    static beamParticle *d_send_up,*d_send_down,*d_recv_up,*d_recv_down;
    int send_up_cnt,send_down_cnt,recv_up_cnt,recv_down_cnt;
    int up,down;
//     MPI_Status status;
    static int nstep = 0;
    static int first_call = 1;
    printf("rank %d in send particles \n",GetRank());
    
    
    if(first_call == 1)
    {
#ifndef CUDA_WRAP_FFTW_ALLOWED      
       CUDA_WRAP_allocFly(*Np,&d_send_down,&d_send_up,&d_recv_down,&d_recv_up);
#endif       
       first_call = 0;
    }
    
#ifndef PARALLEL_ONLY 

#ifdef CUDA_WRAP_PARALLEL_DEBUG
  //  printf("before Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    CUDA_WRAP_getFlyList(Np,X_min,X_max,d_send_down,&send_down_cnt,d_send_up,&send_up_cnt);

#ifdef CUDA_WRAP_PARALLEL_DEBUG
 //   printf("after  Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    
    
      
      
       copyParticlesToHost(h_send_down,d_send_down,send_down_cnt);
       copyParticlesToHost(h_send_up,d_send_up,send_up_cnt);
 
#endif 
    
#ifdef PARALLEL_ONLY
    send_down_cnt = 10 +GetRank();
    send_up_cnt   = 10 +GetRank();
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rankA %d send_down_cnt %d send_up_cnt %d nstep %d \n",GetRank(),send_down_cnt,send_up_cnt,nstep);
#endif    
  /*   if(nstep == 1)
    {
      printf("%d EXITING \n",rank);
      exit(0);
    }*/
//    if(size > 1)
//    {
       h_send_down = (beamParticle *)malloc(sizeof(beamParticle)*send_down_cnt);
   
       nstep++;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankB %d send_down_cnt %d send_up_cnt %d \n",GetRank(),send_down_cnt,send_up_cnt);
#endif       
       h_send_up   = (beamParticle *)malloc(sizeof(beamParticle)*send_up_cnt);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankC %d send_down_cnt %d send_up_cnt %d \n",GetRank(),send_down_cnt,send_up_cnt);
#endif       
//    }
#endif    
    
    down = GetRank() - 1;
    up   = GetRank() + 1;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
     printf("rank %d in send particles A \n",GetRank());
#endif     
    if((GetRank() % 2) == 0)
    {
        if(GetRank() > 0)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send down %d \n",GetRank()); 
#endif	   
//            MPI_Sendrecv(&send_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,&recv_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
	}
        if(GetRank() < size - 1)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send up %d \n",GetRank()); 
#endif	   
	  
//            MPI_Sendrecv(&send_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,&recv_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
	}
    }
    else
    {
        if(GetRank() < size -1)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send up %d \n",rank); 
#endif	   
	  
//            MPI_Sendrecv(&send_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,&recv_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
	}
        if(GetRank() > 0)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send down %d \n",rank); 
#endif	   
	  
//            MPI_Sendrecv(&send_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,&recv_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
	}
      
    }
    
 //   if(size > 1)
 //   {
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particles up %d down %d \n",rank,recv_up_cnt,recv_down_cnt);
#endif
       
       h_recv_down = (beamParticle *)malloc(sizeof(beamParticle)*recv_down_cnt);
       h_recv_up   = (beamParticle *)malloc(sizeof(beamParticle)*recv_up_cnt);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particles B\n",rank);    
#endif       
 //   }

    if((GetRank() % 2) == 0)
    {
        if(GetRank() > 0)
	{
//            /*MPI_Sendrecv(h_send_down,send_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,
// 			h_r*/ecv_down,recv_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}
        if(GetRank() < size -1)
	{
//            MPI_Sendrecv(h_send_up,send_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,
// 			h_recv_up,recv_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}
    }
    else
    {
        if(GetRank() < size -1)
	{
//            MPI_Sendrecv(h_send_up,send_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,
// 			h_recv_up,recv_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}

        if(GetRank() > 0)
	{
//            MPI_Sendrecv(h_send_down,send_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,
// 			h_recv_down,recv_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}
      
    }    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send particles C \n",rank);    
#endif
    
#ifndef PARALLEL_ONLY    
  //  if(size > 1)
  //  {
       copyParticlesToDevice(d_recv_down,h_recv_down,recv_down_cnt);
       copyParticlesToDevice(d_recv_up,h_recv_up,recv_up_cnt);
    
       addBeamParticles(Np,d_recv_down,recv_down_cnt);
       addBeamParticles(Np,d_recv_up,recv_up_cnt);
  //  }
#endif    
  
}


int SendBeamParticlesUp(int *Np)
{
    beamParticle *h_send_up,*h_send_down,*h_recv_up,*h_recv_down;
    static beamParticle *d_send_up,*d_send_down,*d_recv_up,*d_recv_down;
    int send_up_cnt,send_down_cnt,recv_up_cnt = 0,recv_down_cnt = 0;
    int up,down;
//     MPI_Status status;
    static int nstep = 0;
    static int first_call = 1;

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send particles \n",rank);
#endif    
    
    
    if(first_call == 1)
    {
#ifndef CUDA_WRAP_FFTW_ALLOWED      
       CUDA_WRAP_allocFly(*Np,&d_send_down,&d_send_up,&d_recv_down,&d_recv_up);
#endif       
       first_call = 0;
    }
    
#ifndef PARALLEL_ONLY 
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("before Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    CUDA_WRAP_getFlyList(Np,X_min,X_max,d_send_down,&send_down_cnt,d_send_up,&send_up_cnt);

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("after  Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    
       copyParticlesToHost(h_send_down,d_send_down,send_down_cnt);
       copyParticlesToHost(h_send_up,d_send_up,send_up_cnt);
 
#endif 
    
#ifdef PARALLEL_ONLY
    send_down_cnt = 10 +rank;
    send_up_cnt   = 10 +rank;

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rankA %d send_down_cnt %d send_up_cnt %d nstep %d \n",rank,send_down_cnt,send_up_cnt,nstep);
#endif    
  /*   if(nstep == 1)
    {
      printf("%d EXITING \n",rank);
      exit(0);
    }*/
//    if(size > 1)
//    {
       h_send_down = (beamParticle *)malloc(sizeof(beamParticle)*send_down_cnt);
   
       nstep++;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankB %d send_down_cnt %d send_up_cnt %d \n",rank,send_down_cnt,send_up_cnt);
#endif       
       h_send_up   = (beamParticle *)malloc(sizeof(beamParticle)*send_up_cnt);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankC %d send_down_cnt %d send_up_cnt %d \n",rank,send_down_cnt,send_up_cnt);
#endif
       
//    }
#endif    
    
    down = GetRank() - 1;
    up   = GetRank() + 1;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
     printf("rank %d in send particles A \n",GetRank());
#endif     
        if(GetRank() < size - 1)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send up %d \n",rank); 
#endif	   
	  
//            MPI_Sendrecv(&send_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,&recv_up_cnt,1,MPI_INTEGER,up,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
        }
 //   if(size > 1)
 //   {
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particles up %d down %d \n",GetRank(),recv_up_cnt,recv_down_cnt);
#endif       
       h_recv_down = (beamParticle *)malloc(sizeof(beamParticle)*recv_down_cnt);
       h_recv_up   = (beamParticle *)malloc(sizeof(beamParticle)*recv_up_cnt);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particles B\n",GetRank());    
#endif       
 //   }

        if(GetRank() < size -1)
	{
//            MPI_Sendrecv(h_send_up,send_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,
// 			h_recv_up,recv_up_cnt*sizeof(beamParticle),MPI_BYTE,up,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}
#ifdef CUDA_WRAP_PARALLEL_DEBUG
 printf("rank %d in send particles C \n",GetRank());    
#endif 
 
#ifndef PARALLEL_ONLY    
  //  if(size > 1)
  //  {
#ifdef CUDA_WRAP_PARALLEL_DEBUG
        printf("rank %d before write particles Np %d \n",rank,*Np);    
#endif	

       copyParticlesToDevice(d_recv_down,h_recv_down,recv_down_cnt);
       copyParticlesToDevice(d_recv_up,h_recv_up,recv_up_cnt);
    
       addBeamParticles(Np,d_recv_down,recv_down_cnt);
       addBeamParticles(Np,d_recv_up,recv_up_cnt); 
  //  }
#endif    
  
}


int SendBeamParticlesDown(int *Np)
{
    beamParticle *h_send_up,*h_send_down,*h_recv_up,*h_recv_down;
    static beamParticle *d_send_up,*d_send_down,*d_recv_up,*d_recv_down;
    int send_up_cnt,send_down_cnt,recv_up_cnt = 0,recv_down_cnt = 0;
    int up,down;
//     MPI_Status status;
    static int nstep = 0;
    static int first_call = 1;
    
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rank %d in send particles \n",rank);
#endif    
    
    
    if(first_call == 1)
    {
       CUDA_WRAP_allocFly(*Np,&d_send_down,&d_send_up,&d_recv_down,&d_recv_up);
       first_call = 0;
    }
    
#ifndef PARALLEL_ONLY 
    
#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("before Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    CUDA_WRAP_getFlyList(Np,X_min,X_max,d_send_down,&send_down_cnt,d_send_up,&send_up_cnt);

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("after  Np %d X_min %e X_max %e down %d up %d\n",*Np,X_min,X_max,send_down_cnt,send_up_cnt);
#endif    
    
    
      
      
       copyParticlesToHost(h_send_down,d_send_down,send_down_cnt);
       copyParticlesToHost(h_send_up,d_send_up,send_up_cnt);
 
#endif 
    
#ifdef PARALLEL_ONLY
    send_down_cnt = 10 +rank;
    send_up_cnt   = 10 +rank;

#ifdef CUDA_WRAP_PARALLEL_DEBUG
    printf("rankA %d send_down_cnt %d send_up_cnt %d nstep %d \n",rank,send_down_cnt,send_up_cnt,nstep);
#endif    
  /*   if(nstep == 1)
    {
      printf("%d EXITING \n",rank);
      exit(0);
    }*/
//    if(size > 1)
//    {
       h_send_down = (beamParticle *)malloc(sizeof(beamParticle)*send_down_cnt);
   
       nstep++;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankB %d send_down_cnt %d send_up_cnt %d \n",rank,send_down_cnt,send_up_cnt);
#endif       
       h_send_up   = (beamParticle *)malloc(sizeof(beamParticle)*send_up_cnt);

#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rankC %d send_down_cnt %d send_up_cnt %d \n",rank,send_down_cnt,send_up_cnt);
#endif       
//    }
#endif    
    
    down = GetRank() - 1;
    up   = GetRank() + 1;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
     printf("rank %d in send particles A \n",GetRank());
#endif     
        if(GetRank() > 0)
	{
#ifdef CUDA_WRAP_PARALLEL_DEBUG
	   printf("send down %d \n",GetRank()); 
#endif	   
//            MPI_Sendrecv(&send_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,&recv_down_cnt,1,MPI_INTEGER,down,PARTICLE_NUMBER_TAG,MPI_COMM_WORLD,&status);
	}
 //   if(size > 1)
 //   {
	  
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particlesDown up %d down %d \n",rank,recv_up_cnt,recv_down_cnt);
#endif       
       h_recv_down = (beamParticle *)malloc(sizeof(beamParticle)*recv_down_cnt);
       h_recv_up   = (beamParticle *)malloc(sizeof(beamParticle)*recv_up_cnt);
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d in send particles B\n",rank);    
#endif       
 //   }

        if(GetRank() > 0)
	{
//            MPI_Sendrecv(h_send_down,send_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,
// 			h_recv_down,recv_down_cnt*sizeof(beamParticle),MPI_BYTE,down,PARTICLE_TAG,MPI_COMM_WORLD,&status);
	}
#ifdef CUDA_WRAP_PARALLEL_DEBUG
 printf("rank %d in send particles C \n",rank);    
#endif
 
#ifndef PARALLEL_ONLY    
  //  if(size > 1)
  //  {
#ifdef CUDA_WRAP_PARALLEL_DEBUG
       printf("rank %d before write particles Np %d \n",rank,*Np);    
#endif       
  
       copyParticlesToDevice(d_recv_down,h_recv_down,recv_down_cnt);
       copyParticlesToDevice(d_recv_up,h_recv_up,recv_up_cnt);
    
       addBeamParticles(Np,d_recv_down,recv_down_cnt);
       addBeamParticles(Np,d_recv_up,recv_up_cnt);
       
  //  }
#endif    
  
}

