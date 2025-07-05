//#include "../cells.h"
//#include "../mesh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "diagnostic_print.h"

#include "../para.h"



double *d_Rho3D,*d_Jx3D,*d_Jy3D,*d_Jz3D,*d_JyBeam3D,*d_JzBeam3D;
double *d_Ex3D,*d_Ey3D,*d_Ez3D,*d_Bx3D,*d_By3D,*d_Bz3D;

double *h_Rho3D,*h_Jx3D,*h_Jy3D,*h_Jz3D;


int CUDA_WRAP_3Dto2D(int iLayer,int Ny,int Nz,double *d_3d,double *d_2d)
{
//    double *h_copy = (double *)malloc(sizeof(double)*Ny*Nz);
    
    cudaMemcpy(d_2d,d_3d+iLayer*Ny*Nz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
    
  //  cudaMemcpy(h_copy,d_3d+iLayer*Ny*Nz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToHost);	
    
    return 0;
}

int CUDA_WRAP_2Dto3D(int iLayer,int Nx,int Ny,int Nz,double *d_3d,double *d_2d)
{
    if(iLayer < Nx - 1)
    {
        cudaMemcpy(d_3d+iLayer*Ny*Nz,d_2d,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
    }
    
    return 0;
}

int CUDA_WRAP_write3Darray(int Nx,int Ny,int Nz,double *d_3d)
{
    double *h_3d = (double *)malloc(Nx*Ny*Nz*sizeof(double));
    
    cudaMemcpy(h_3d,d_3d,Nx*Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
    
    FILE *f = fopen("ctrlField3d.dat","wt");
    
    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          for(int k = 0;k < Nz;k++)
          {    
	      fprintf(f,"%5d %5d %5d %15.5e\n",i,j,k,h_3d[i*Ny*Nz+k*Ny+j]);
	  }
       }
    }
    fclose(f);
    free(h_3d);
}



int CUDA_WRAP_alloc3DArray(int Nx,int Ny,int Nz,double **d_x)
{
 int err = cudaGetLastError();  
    int err05 = cudaMalloc((void **)d_x,sizeof(double)*Nx*Ny*Nz);
 int err1 = cudaGetLastError();    
    cudaMemset(*d_x,0,Nx*Ny*Nz*sizeof(double));
 int err2 = cudaGetLastError();      
    return err05;
}

int CUDA_WRAP_alloc3Dfields(int Nx,int Ny,int Nz)
{
    int err_ex,err_ey,err_ez,err_bx,err_by,err_bz;
    
    int err = cudaGetLastError();
    err_ex = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Ex3D);   
    if(err_ex != cudaSuccess) printf("alloc3D error ex %d \n",err_ex);
    
    int err1 = cudaGetLastError();
    err_ey = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Ey3D);   
    //if(err_ey != cudaSuccess) 
    printf("alloc3D error ey %d \n",err_ey);

    int err2 = cudaGetLastError();
    err_ez = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Ez3D);   
    //if(err_ez != cudaSuccess) 
    printf("alloc3D error ez %d \n",err_ez);
    
    int err3 = cudaGetLastError();
    
    err_bx = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Bx3D);   
    //if(err_bx != cudaSuccess) 
    printf("alloc3D error bx %d \n",err_bx);
    
    int err4 = cudaGetLastError();
    
    err_by = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_By3D);   
    if(err_by != cudaSuccess) printf("alloc3D error by %d \n",err_by);
    
    int err5 = cudaGetLastError();
    
    err_bz = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Bz3D);   
    if(err_bz != cudaSuccess) printf("alloc3D error bz %d \n",err_bz);
    
    int err6 = cudaGetLastError();
    
    
    return err6;
}

int CUDA_WRAP_alloc3Dcurrents(int Nx,int Ny,int Nz)
{
    CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Rho3D);   
    CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Jx3D);   
    CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Jy3D);   
    CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_Jz3D);   
    
    return 0;
}

int CUDA_WRAP_copyLayerCurrents(int iLayer,int Nx,int Ny,int Nz,double *rho,double *jx,double *jy,double *jz)
{
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Rho3D,rho); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Jx3D,jx); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Jy3D,jy); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Jz3D,jz); 
    //CUDA_DEBUG_printDdevice_matrix(Ny,Nz,jy,"Rho saving to 3D");  
    return 0;
}

int CUDA_WRAP_copyLayerFields(int iLayer,int Nx,int Ny,int Nz,double *ex,double *ey,double *ez,double *bx,double *by,double *bz)
{
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Ex3D,ex); 
 //   CUDA_WRAP_write3Darray(Nx,Ny,Nz,d_Ex3D);
    
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Ey3D,ey); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Ez3D,ez); 

    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Bx3D,bx); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_By3D,by); 
    CUDA_WRAP_2Dto3D(iLayer,Nx,Ny,Nz,d_Bz3D,bz);
    
    if(GetRank() == 1)
    {
        double *dm;
        cudaMalloc(&dm,sizeof(double)*Ny*Nz);
        CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Ey3D,dm);

        CUDA_DEBUG_printDdevice_matrix(Ny,Nz,ey,"ey saving to 3D");  
        CUDA_DEBUG_printDdevice_matrix(Ny,Nz,dm,"ey3D saving to 3D");  
    }
        
    return 0;
}

int CUDA_WRAP_restoreLayerCurrents(int iLayer,int Nx,int Ny,int Nz,double *rho,double *jx,double *jy,double *jz)
{
    CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Rho3D,rho); 
    CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jx3D,jx);
    CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jy3D,jy); 
    CUDA_WRAP_3Dto2D(iLayer,Ny,Nz,d_Jz3D,jz); 
    //CUDA_DEBUG_printDdevice_matrix(Ny,Nz,rho,"Rho restoring from 3D");  
    return 0;
}

int CUDA_WRAP_setCurrentsToZero(int Ny,int Nz,double *jx,double *jy,double *jz)
{
    cudaMemset(jx,0,sizeof(double)*Ny*Nz);
    cudaMemset(jy,0,sizeof(double)*Ny*Nz);
    cudaMemset(jz,0,sizeof(double)*Ny*Nz);
    
    return 0;
}

