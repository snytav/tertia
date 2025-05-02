#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>

//#include "cuPrintf.cu"
#include "cuCell.h"

#include "../particles.h"
//#include "../cells.h"
//#include "../mesh.h"
#include "cuda_wrap_vector_list.h"
#include "cuda_wrap_control.h"
#include "cuParticles.h"
#include "diagnostic_print.h"
#include "../run_control.h"
#include "beam_copy.h"

#include "surf.h"

double last_wrong,last_delta;
int    last_max_delta_value; 


FILE *f_part_device = NULL;

int MaximalNumberOfParticlesPerCellonDevice = -1;

double *h_vx,*d_vx;

double *h_aqq,*d_aqq;

double *d_partRho = NULL, *d_partJx = NULL,*d_partJy = NULL,*d_partJz = NULL;

//surface<void, 2> partSurfInFloat; 
double *partSurfIn;
double *cuInputArrayX;

double *SurfEx;

//surface<void,2>SurfEy,SurfEz,SurfBx,SurfBy,SurfBz; 
//cudaArray       *cuExInputArray,*cuEyInputArray,*cuEzInputArray,*cuBxInputArray,*cuByInputArray,*cuBzInputArray; 

double *cuOutputArrayX;
double    *partSurfOut;

double *d_particleResult = NULL;
double *d_params = NULL;

int fillAttributesFirstCall = 1;
int write_values_first_call = 1;



int iFullStep;

__device__ void surf2Dwrite
(                          double t,
                           double *in_surfaceT,
                           int nx,int ny,
                           int ny1

 )
{
         in_surfaceT[nx*ny1 + ny] = t;
//          *x_re = t;
}

// __device__ void surf2Dread
// (double *x_re,
//                            double *in_surfaceT,
//                            int nx,int ny,
//                            int NY)
// {
//          double t = in_surfaceT[nx*NY + ny];
//          *x_re = t;
// }



__device__ void cuDepositRhoInCell(int l_My,int l_Mz,int part_number,double *buf,
                      double *d_partJx,double *d_partRho,
                      int isort,
                      int i, int nx, int ny,
                      double Vx, double Vy, double Vz,
                      double x, double y, double z,
                      double djx, double djy, double djz, double drho);


__device__ void cuDepositCurrentsInCell(int l_My,int l_Mz,int part_number,double *buf,
                      double *d_partJy,double *d_partJz,
                      int isort,
                      int i, int nx, int ny,
                      double Vx, double Vy, double Vz,
                      double x, double y, double z,
                      double djx, double djy, double djz, double drho);



int CUDA_WRAP_setMaximalPICnumberOnDevice(int s)
{
    if(MaximalNumberOfParticlesPerCellonDevice < 0) MaximalNumberOfParticlesPerCellonDevice = s;
}

int CUDA_WRAP_getMaximalPICnumberOnDevice()
{
    return MaximalNumberOfParticlesPerCellonDevice;
}



__device__ int ncell2(int nx,int ny)
{
    if(nx < 0) nx = 0;
    if(ny < 0) ny = 0;
    
    int n = (nx*(gridDim.y*blockDim.y) + ny);
    int size = (gridDim.y*blockDim.y)*(gridDim.x*blockDim.x);
    if(n < 0) n = 0;
    if(n >= size) n = size - 1;
    
    return n;
}

__device__ int ncell(int nx,int ny)
{
    int n = (nx*(gridDim.y*blockDim.y) + ny);
    
    return n;
}

//special core to assign currents (taking into account the periodical boundaries)
__device__ int ncellDeposit(int ny,int nx)
{
    int nx0 = nx,ny0 = ny;
    
    if(nx < 0) nx = gridDim.x*blockDim.x - 1;
    if(ny < 0) ny = gridDim.y*blockDim.y - 1;
    
     
    if(nx == gridDim.x*blockDim.x) nx = 0;
    if(ny == gridDim.y*blockDim.y) ny = 0;
    
   

    int n = (nx*(gridDim.y*blockDim.y) + ny);
    ////cuPrintf("n %d nx %d ny %d \n",n,nx,ny);
    //cuPrintf("ncellD n %d nx0 %d ny0 %d nx %d ny %d \n",n,nx0,ny0,nx,ny);
    
    return n;
}

__device__ void addToMatrix(double *m,double t,int nx,int ny)
{
    int n0 = nx*(gridDim.y*blockDim.y) + ny;
    int n  = ncellDeposit(nx,ny);
    double t0 = m[n];
    
    m[n] += t;
    
    //cuPrintf("addToMatrix: nx %d ny %d to %15.5e adding %15.5e result %15.5e position real %d naiv %d \n",nx,ny,t0,t,m[n],n,n0);
    
}

__device__ void addToMatrixRho(double *m,double t,int nx,int ny)
{
    int n0 = nx*(gridDim.y*blockDim.y) + ny;
    int n  = ncellDeposit(nx,ny);
    double t0 = m[n];
    
    m[n] += t;
    
    
    ////cuPrintf("addToMatrixRho: nx %d ny %d to %15.5e adding %15.5e result %15.5e position real %d naiv %d \n",nx,ny,t0,t,m[n],n,n0);
      //cuPrintf("addToMatrixRho: nx %d ny %d thx %d thy %d to %15.5e adding %15.5e result %15.5e position real %d naiv %d \n",nx,ny,threadIdx.x,threadIdx.y,t0,t,m[n],n,n0);  
}

//special core to assign fields (taking into account the periodical boundaries). In future may be different from currents
__device__ int ncellDepositField(int ny,int nx)
{
    int nx0 = nx,ny0 = ny;
  
    if(nx < 0) nx = gridDim.x*blockDim.x - 1;
    if(ny < 0) ny = gridDim.y*blockDim.y - 1;

    if(nx == gridDim.x*blockDim.x) nx = 0;
    if(ny == gridDim.y*blockDim.y) ny = 0;

    int n = (nx*(gridDim.y*blockDim.y) + ny);
    
    printf("nx0 %d ny0 %d gridDim.x*blockDim.x %d gridDim.y*blockDim.y %d nx %d ny %d \n",nx0,ny0,gridDim.x*blockDim.x,gridDim.y*blockDim.y,nx,ny);    
    
    return n;
}


__device__ int ncell1(int nx,int ny)
{
    
    int n = (nx*(gridDim.y*blockDim.y) + ny);
    int size = (gridDim.y*blockDim.y)*(gridDim.x*blockDim.x);
    if(n < 0) n = 0;
    if(n >= size) n = size - 1;
    
    return n;
}

// __device__ void controlWriteSurface(int cell_number,double x,double y,double z,double px,double py,double pz,double weight,double f_Q2m)
// {
//             surf2Dwrite(x,      partSurfOut, 0,   cell_number );
//             surf2Dwrite(y,      partSurfOut, 1*8, cell_number );
//             surf2Dwrite(z,      partSurfOut, 2*8, cell_number );
//             surf2Dwrite(px,     partSurfOut, 3*8, cell_number );
//             surf2Dwrite(py,     partSurfOut, 4*8, cell_number );
//             surf2Dwrite(pz,     partSurfOut, 5*8, cell_number );
//             surf2Dwrite(weight, partSurfOut, 6*8, cell_number );
//             surf2Dwrite(f_Q2m,  partSurfOut, 7*8, cell_number );
// }

//writing a value to the control array for a definite particle in a definite cell
__device__ int write_particle_value(int Ny,int i,int j,int num_attr,int ppc_max,int k,int n,double *d_p,double t,int surf_height)
{
	int cell_number = i*Ny + j;
	
	d_p [cell_number*num_attr*ppc_max + k*num_attr + n] =t;
	
	return 0;
}

//writing particle coordinates to global-memory array for comparison...
__device__ void controlWrite(int cell_number,int part_per_cell_max,int k,double x,double y,double z,double px,double py,double pz,double weight,double f_Q2m,double *d_data_in)
//__device__ void controlWrite(int cell_number,int part_per_cell_max,int k,double x,double *d_data_in)
{
        int num = cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES;
//	//cuPrintf("%d %d %d num %d \n",cell_number,part_per_cell_max,k,num);
        d_data_in [num + 0] = x;
	d_data_in [num + 1] = y;
	d_data_in [num + 2] = z;
		      
	d_data_in [num + 3] = px;
	d_data_in [num + 4] = py;
	d_data_in [num + 5] = pz;
	d_data_in [num + 6] = weight;
	d_data_in [num + 7] = f_Q2m;
}


__global__ void testSurfKernel(double *t)
{
//        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
  //      unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
	double ccc;
	printf("test !!!!!!!!!!!!!!!!!!!!!! \n");
/*	
	*t = -10.0;
	surf2Dread(&ccc, SurfEx,      0, 1 ); 
	//*t = -10.0;
	*t = ccc;
	//cuPrintf("test %e \n",ccc);
  */
}

//reading particle coordinates from a surface
__device__ void getParticle(double *partSurfIn,
   int cn,int pn,double *x,double *y,double *z,double *px,double *py,double *pz,double *w,double *qm,
                            int Ny,int Nz)
{
//            //cuPrintf("getParticle cn %d \n",cn);
            pn += CUDA_WRAP_PARTICLE_START_INDEX;  

            surf2Dread(x, partSurfIn, pn,     cn,Ny*Nz );
            surf2Dread(y, partSurfIn, pn + 1, cn,Ny*Nz );
            surf2Dread(z, partSurfIn, pn + 2, cn,Ny*Nz );
            surf2Dread(px,partSurfIn, pn + 3, cn,Ny*Nz );
            surf2Dread(py,partSurfIn, pn + 4, cn,Ny*Nz );
            surf2Dread(pz,partSurfIn, pn + 5, cn,Ny*Nz );
            surf2Dread(w, partSurfIn, pn + 6, cn,Ny*Nz );
            surf2Dread(qm,partSurfIn, pn + 7, cn,Ny*Nz );
	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT	    
	     //cuPrintf("I read %e %e %e %e %e %e pn %d cell %d \n",*x,*y,*z,*px,*py,*pz,pn,cn);
#endif	     
}

// reading particle attributes as an array
void __global__ getParticleFromCell(double *partSurfIn,
   int ny,int nz,double *part)
{
   double  w,q;
   getParticle(partSurfIn,ncell(ny,nz),0,&part[0],&part[1],&part[2],&part[3],&part[4],&part[5],&w,&q,ny,nz);
   
}

double __device__ getField(int nx,int ny,int attr,double *d_F)
{
   if(((ny < 0) || (ny == gridDim.y*blockDim.y)) && (attr > 3)) return 0.0;
   else return d_F[ncellDepositField(nx,ny)];
}

//writing each of all the field components including necessary shifts
void __device__ setFieldComponent(
   double *partSurfIn,
   int nx,int ny,int attr,double *d_F,int surf_height)
{
    int     nccc,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm;
    double  accc,acpc,accp,acpp,acmc,accm,acmm,acmp,acpm;
    printf("attr %3d\n",attr);
    
    nccc = ncellDepositField(nx,ny); //
    ncpc = ncellDepositField(nx+1,ny);
    nccp = ncellDepositField(nx,ny+1); //
    ncpp = ncellDepositField(nx+1,ny+1); //
    ncmc = ncellDepositField(nx-1,ny);
    nccm = ncellDepositField(nx,ny-1);
    ncmm = ncellDepositField(nx-1,ny-1);//
    ncmp = ncellDepositField(nx-1,ny+1);
    ncpm = ncellDepositField(nx+1,ny-1); 
    
    accc = d_F[nccc];
    accc = getField(nx,ny,attr,d_F);
    acpc = d_F[ncpc];
    acpc = getField(nx+1,ny,attr,d_F);
    accp = d_F[nccp];
    accp = getField(nx,ny+1,attr,d_F);
    acpp = d_F[ncpp];
    acpp = getField(nx+1,ny+1,attr,d_F);
    acpm = d_F[ncpm];
    acpm = getField(nx+1,ny-1,attr,d_F);
    acmp = d_F[ncmp];
    acmp = getField(nx-1,ny+1,attr,d_F);
    acmc = d_F[ncmc];
    acmc = getField(nx-1,ny,attr,d_F);
    accm = d_F[nccm];
    accm = getField(nx,ny-1,attr,d_F);
    acmm = d_F[ncmm];
    acmm = getField(nx-1,ny-1,attr,d_F);
    
    surf2Dwrite(accc, partSurfIn, attr*NUMBER_ATTRIBUTES + 0, nccc,surf_height);

    surf2Dwrite(acpc, partSurfIn, attr*NUMBER_ATTRIBUTES + 1, nccc,surf_height);
    surf2Dwrite(accp, partSurfIn, attr*NUMBER_ATTRIBUTES + 2, nccc,surf_height);
    surf2Dwrite(acpp, partSurfIn, attr*NUMBER_ATTRIBUTES + 3, nccc,surf_height);
    surf2Dwrite(acpm, partSurfIn, attr*NUMBER_ATTRIBUTES + 4, nccc,surf_height);
    surf2Dwrite(acmp, partSurfIn, attr*NUMBER_ATTRIBUTES + 5, nccc,surf_height);
            
//	    return;
    surf2Dwrite(acmc, partSurfIn, attr*NUMBER_ATTRIBUTES + 6, nccc,surf_height);
    surf2Dwrite(accm, partSurfIn, attr*NUMBER_ATTRIBUTES + 7, nccc,surf_height);
    surf2Dwrite(acmm, partSurfIn, attr*NUMBER_ATTRIBUTES + 8, nccc,surf_height);
    
  //  printf("nx %3d ny %3d attr %3d accc %10.3e acpc %25.15e accp %10.3e acpp %10.3e acpm %10.3e acmp %10.3e acmc %10.3e accm %10.3e acmm %10.3e \n",
//	    nx,ny,attr,accc,acpc,accp,acpp,acpm,acmp,acmc,accm,acmm);
}

//writing all the field components
__global__ void SetField(
   double *partSurfIn,
   double *ex,double *ey,double *ez,double *bx,double *by,double *bz,int surf_height)
{
    unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;   
    
    setFieldComponent(partSurfIn,nx,ny,0,ex,surf_height);
    setFieldComponent(partSurfIn,nx,ny,1,ey,surf_height);
    setFieldComponent(partSurfIn,nx,ny,2,ez,surf_height);
    
    setFieldComponent(partSurfIn,nx,ny,3,bx,surf_height);
    setFieldComponent(partSurfIn,nx,ny,4,by,surf_height);
    setFieldComponent(partSurfIn,nx,ny,5,bz,surf_height);
}

//reading one field component for a particle in a cell   
__device__ void getFieldForParticle(
   double *partSurfIn,
   int nx,int ny,int attr,double *ccc,double *cpc,double *ccp,double *cpp,double *cpm,double *cmp,double *cmc,double *ccm,double *cmm, int surf_height)
{
   int nccc;//,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm;
   double t;
   nccc = ncell(ny,nx);
            
   surf2Dread(ccc, partSurfIn, attr*NUMBER_ATTRIBUTES, nccc,surf_height);
	    
   surf2Dread(cpc, partSurfIn, attr*NUMBER_ATTRIBUTES+1,nccc,surf_height);
   surf2Dread(ccp, partSurfIn, attr*NUMBER_ATTRIBUTES + 2, nccc,surf_height);
   surf2Dread(cpp, partSurfIn, attr*NUMBER_ATTRIBUTES + 3, nccc,surf_height);
   surf2Dread(cpm, partSurfIn, attr*NUMBER_ATTRIBUTES + 4, nccc,surf_height);
   surf2Dread(cmp, partSurfIn, attr*NUMBER_ATTRIBUTES + 5, nccc,surf_height);
            
//	    return;
   surf2Dread(cmc, partSurfIn, attr*NUMBER_ATTRIBUTES + 6, nccc,surf_height);
   surf2Dread(ccm, partSurfIn, attr*NUMBER_ATTRIBUTES + 7, nccc,surf_height);
   surf2Dread(cmm, partSurfIn, attr*NUMBER_ATTRIBUTES + 8, nccc,surf_height);
}

__device__ void getFieldForParticleDirect(double *d_F,int nx,int ny,int attr,double *ccc,double *cpc,double *ccp,double *cpp,double *cpm,double *cmp,double *cmc,double *ccm,double *cmm)
{
    int     nccc,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm;
    double  accc,acpc,accp,acpp,acmc,accm,acmm,acmp,acpm;
    
    nccc = ncellDepositField(nx,ny); //
    ncpc = ncellDepositField(nx+1,ny);
    nccp = ncellDepositField(nx,ny+1); //
    ncpp = ncellDepositField(nx+1,ny+1); //
    ncmc = ncellDepositField(nx-1,ny);
    nccm = ncellDepositField(nx,ny-1);
    ncmm = ncellDepositField(nx-1,ny-1);//
    ncmp = ncellDepositField(nx-1,ny+1);
    ncpm = ncellDepositField(nx+1,ny-1); 
    
    accc = d_F[nccc];
    *ccc = getField(nx,ny,attr,d_F);
    acpc = d_F[ncpc];
    *cpc = getField(nx+1,ny,attr,d_F);
    accp = d_F[nccp];
    *ccp = getField(nx,ny+1,attr,d_F);
    acpp = d_F[ncpp];
    *cpp = getField(nx+1,ny+1,attr,d_F);
    acpm = d_F[ncpm];
    *cpm = getField(nx+1,ny-1,attr,d_F);
    acmp = d_F[ncmp];
    *cmp = getField(nx-1,ny+1,attr,d_F);
    acmc = d_F[ncmc];
    *cmc = getField(nx-1,ny,attr,d_F);
    accm = d_F[nccm];
    *ccm = getField(nx,ny-1,attr,d_F);
    acmm = d_F[ncmm];
    *cmm = getField(nx-1,ny-1,attr,d_F);  
}


// writing new particle coordinates to the surface
__device__ void writeParticle(double *partSurfOut,
                              int cell_number,int pn,double x,double y, double z,double px,double py, double pz,double weight,double f_Q2m,int surf_height)
{
            pn += NUMBER_ATTRIBUTES*8; 
            surf2Dwrite(x,      partSurfOut, pn,       cell_number,surf_height );
            surf2Dwrite(y,      partSurfOut, pn + 1*8, cell_number,surf_height );
            surf2Dwrite(z,      partSurfOut, pn + 2*8, cell_number,surf_height );
            surf2Dwrite(px,     partSurfOut, pn + 3*8, cell_number,surf_height );
            surf2Dwrite(py,     partSurfOut, pn + 4*8, cell_number,surf_height );
            surf2Dwrite(pz,     partSurfOut, pn + 5*8, cell_number,surf_height );
            surf2Dwrite(weight, partSurfOut, pn + 6*8, cell_number,surf_height );
            surf2Dwrite(f_Q2m,  partSurfOut, pn + 7*8, cell_number,surf_height );
	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT	    
	    //cuPrintf("O write %e %e %e %e %e %e pn %d cell %d\n",x,y,z,px,py,pz,pn,cell_number);
#endif	    
	    
}

//writing particles from output surface to input surface
void __global__ copyParticlesToInputSurface(double *partSurfIn,
                                            double *partSurfOut,
                                            int surf_height)
{
            unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
            unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;
	    unsigned int pn = threadIdx.z*NUMBER_ATTRIBUTES*8;
	    double x,y,z,px,py,pz,w,qm;
	    
            int cell_number = ncell(nx,ny);
	    
	    int pn1 = pn + NUMBER_ATTRIBUTES*8; 
            surf2Dread(&x,      partSurfOut, pn1,       cell_number,surf_height );
            surf2Dread(&y,      partSurfOut, pn1 + 1, cell_number,surf_height );
            surf2Dread(&z,      partSurfOut, pn1 + 2, cell_number,surf_height );
            surf2Dread(&px,     partSurfOut, pn1 + 3, cell_number,surf_height );
            surf2Dread(&py,     partSurfOut, pn1 + 4, cell_number,surf_height );
            surf2Dread(&pz,     partSurfOut, pn1 + 5, cell_number,surf_height );
            surf2Dread(&w,      partSurfOut, pn1 + 6, cell_number,surf_height );
            surf2Dread(&qm,     partSurfOut, pn1 + 7, cell_number,surf_height );
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT	    
	    //cuPrintf("O2I: write %e %e %e %e %e %e\n",x,y,z,px,py,pz);
#endif	    
            int pn2 = pn + CUDA_WRAP_PARTICLE_START_INDEX;  
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT	    
	    //cuPrintf("O2I pn1 %d pn2 %d cellnumber %d \n",pn1,pn2,cell_number);
#endif	    
            surf2Dwrite(x, partSurfIn, pn2,     cell_number ,surf_height);
            surf2Dwrite(y, partSurfIn, pn2 + 1, cell_number ,surf_height);
            surf2Dwrite(z, partSurfIn, pn2 + 2, cell_number ,surf_height);
            surf2Dwrite(px,partSurfIn, pn2 + 3, cell_number ,surf_height);
            surf2Dwrite(py,partSurfIn, pn2 + 4, cell_number ,surf_height);
            surf2Dwrite(pz,partSurfIn, pn2 + 5, cell_number ,surf_height);
            surf2Dwrite(w, partSurfIn, pn2 + 6, cell_number ,surf_height);
            surf2Dwrite(qm,partSurfIn, pn2 + 7, cell_number ,surf_height);
}

//transverse currents copy from computation array to array accessible by Poisson solver
int CUDA_WRAP_copy_particle_currents(int Nx,int Ny,int Nz,int iLayer)
{
    //cudaMemcpy(d_rR,d_partJx,Ny*Nz*sizeof(double),cudaMemcpyDeviceToDevice);
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_rJy,"rJy before copy from 3D");
    if(iLayer == Nx - 1)
    {
       cudaMemcpy(d_rJy,d_partJy,Ny*Nz*sizeof(double),cudaMemcpyDeviceToDevice);
       cudaMemcpy(d_rJz,d_partJz,Ny*Nz*sizeof(double),cudaMemcpyDeviceToDevice);
    }
    else
    {
       CUDA_WRAP_3Dto2D(iLayer ,Ny,Nz,d_Jy3D,d_rJy);
       CUDA_WRAP_3Dto2D(iLayer ,Ny,Nz,d_Jz3D,d_rJz);
    }
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_rJy,"rJy after copy from 3D");
    return 0;
}

//longitudinal current and density copy from computation array to array accessible by Poisson solver
int CUDA_WRAP_copy_particle_density(int Nx,int Ny,int Nz,int iLayer_jx,int iLayer_rho)
{
    if(iLayer_jx >= Nx - 2)
    {
       cudaMemcpy(d_rJx,d_partJx,Ny*Nz*sizeof(double),cudaMemcpyDeviceToDevice);  
       CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"in copy");
       //cudaMemset(d_partRho,0,sizeof(double)*Ny*Nz);
       cudaMemcpy(d_rRho,d_partRho,Ny*Nz*sizeof(double),cudaMemcpyDeviceToDevice);
    }
    else
    {
       CUDA_WRAP_3Dto2D(iLayer_rho,Ny,Nz,d_Rho3D,d_rRho);
       CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_rRho,"in copy");
       CUDA_WRAP_3Dto2D(iLayer_jx,Ny,Nz,d_Jx3D ,d_rJx);
    }
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_rRho,"end copy");
    return 0;
}

//main kernel - particle movement
__global__ void moveKernel(
double*partSurfIn,
double *partSurfOut,
   int width, int height,int part_per_cell_max,int l_My,int l_Mz,int iFullStep,double *params,double *result,double *buf,
		           double *d_partRho,double *d_partJx,double *d_partJy,double *d_partJz,double *Ex)
{
        // Calculate surface coordinates 
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y; 
        unsigned int cell_number,part_number = threadIdx.z*NUMBER_ATTRIBUTES*8;
        double hx, hy, hz, djx0, djy0, djz0, drho0;
	
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("Begin moverKernel real 12 \n");
	//cuPrintf("rho-1 %e \n",d_partRho[10]);
#endif	
	
	//return;      
        cell_number = ncell(nx,ny);
        
       
	//transmitting global parameters
        hx    = params[0];
        hy    = params[1];
        hz    = params[2];
        djx0  = params[3];
        djy0  = params[4];
        djz0  = params[5];
        drho0 = params[6];

        ////cuPrintf("x %d y %d cell %d my %d mz %d fs %d \n",nx,ny,cell_number,l_My,l_Mz,iFullStep);


        ////cuPrintf("hx %f hy %f hz %f \n",hx,hy,hz);
        
        ////cuPrintf("%f %f %f %f \n",djx0,djy0,djz0,drho0);
        

        // if the cell number is not accidentally wrong
        if (cell_number < width) 
        { 
            float  fx,fy,fz,fu,fv,fw,fweight,fl_q2m;
            double x,y,z,px,py,pz,weight,f_Q2m;
	    //double copy_x,copy_y,copy_z,copy_px,copy_py,copy_pz,copy_weight,copy_f_Q2m;
            double ccc = 0,cpc = 0,ccp = 0, cpp = 0,cpm = 0,cmp = 0,cmc = 0,ccm = 0,cmm = 0;
            
	    double ex, ey, ez,bx,by,bz;
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT
            //cuPrintf("cell %d part %d\n",cell_number,part_number);
#endif	    
            
	    getParticle(partSurfIn,
                    cell_number,part_number,&x,&y,&z,&px,&py,&pz,&weight,&f_Q2m,l_My,l_Mz);
	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	   
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,30,buf,x,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,31,buf,y,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,32,buf,z,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,33,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,34,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,35,buf,pz,l_My*l_Mz);
#endif	    
	    controlWrite(cell_number,part_per_cell_max,part_number,x,y,z,px,py,pz,weight,f_Q2m,result);

#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-2 %e \n",d_partRho[10]);
#endif	
	    
            double ys = y - 0.5;
            double zs = z - 0.5;  

            double ayc = 1. - ys*ys;
            double aym = 0.5*(ys-1)*ys;
            double ayp = 0.5*(1+ys)*ys;
            double azc = 1. - zs*zs;
            double azm = 0.5*(zs-1)*zs;
            double azp = 0.5*(1+zs)*zs;	 
	    double acc = ayc*azc;
            double apc = ayp*azc;
            double acp = ayc*azp;
            double app = ayp*azp;
            double apm = ayp*azm;
            double amp = aym*azp;
            double amc = aym*azc;
            double acm = ayc*azm;
            double amm = aym*azm;

	    // only for printing
            fx = (float)x;
            fy = (float)y;
            fz = (float)z;
            fu = (float)px;
            fv = (float)py;
            fw = (float)pz;
	    fweight = (float)weight;
	    fl_q2m = (float)f_Q2m;
	    
#ifdef CUDA_WRAP_SINGLE_PARTICLE_TRACE           
            //cuPrintf("trace %d %d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n",nx,ny,x,y,z,px,py,pz);
#endif	    
///////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
           nx,ny,0,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);
	   // write_particle_value(int Ny,int i,int j,int num_attr,int ppc_max,int k, int n, double *d_p,double t)
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,110,buf,acc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,111,buf,apc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,112,buf,acp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,113,buf,app,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,114,buf,amp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,115,buf,apm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,116,buf,amc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,117,buf,acm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,118,buf,amm,l_My*l_Mz);

	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,20,buf,ccc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,21,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,22,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,23,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,24,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,25,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,26,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,27,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,28,buf,cmm,l_My*l_Mz);
#endif	    
	    //getFieldForParticle1(nx,ny,0,params);

	    ex = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;
		 
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("Ex %e\n",ex);
#endif	    
	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-3 %e \n",d_partRho[10]);
#endif	
	    
/////////////////////////////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
                            nx,ny,1,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,140,buf,ccc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,141,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,142,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,143,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,144,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,145,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,146,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,147,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,148,buf,cmm,l_My*l_Mz);
#endif	    

	    ey = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;

#ifdef CUDA_WRAP_CUPRINTF_ALL		 
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("Ey %e\n",ey);
#endif	    
/////////////////////////////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
           nx,ny,2,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);

	    ez = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,150,buf,ccc,l_My*l_Mz);

	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,151,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,152,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,153,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,154,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,155,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,156,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,157,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,158,buf,cmm,l_My*l_Mz);
#endif		 
		 
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("Ez %e\n",ey);
#endif	    
//***************************************************************************************************************
///////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
                            nx,ny,3,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,160,buf,ccc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,161,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,162,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,163,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,164,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,165,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,166,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,167,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,168,buf,cmm,l_My*l_Mz);
#endif	    

	    bx = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;
		 
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("Bx %e\n",bx);
#endif	    
/////////////////////////////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
                            nx,ny,4,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,120,buf,ccc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,121,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,122,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,123,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,124,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,125,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,126,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,127,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,128,buf,cmm,l_My*l_Mz);
#endif	    
	    by = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;

		 
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("By %e\n",ey);
#endif	    
/////////////////////////////////////////////////////////////////////////////////////////////////////
	    getFieldForParticle(partSurfIn,
                            nx,ny,5,&ccc,&cpc,&ccp,&cpp,&cpm,&cmp,&cmc,&ccm,&cmm,l_My*l_Mz);

#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,130,buf,ccc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,131,buf,cpc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,132,buf,ccp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,133,buf,cpp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,134,buf,cmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,135,buf,cpm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,136,buf,cmc,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,137,buf,ccm,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,138,buf,cmm,l_My*l_Mz);
#endif	    
	    
	    
	    bz = acc*ccc + apc*cpc + acp*ccp + app*cpp +
                 apm*cpm + amp*cmp + amc*cmc + acm*ccm + amm*cmm;
		 
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("fields %e %e %e %e %e %e %e %e %e\n",ccc,cpc,ccp, cpp,cpm,cmp, cmc,ccm,cmm);
	    //cuPrintf("Bz %e\n",ey);	    
#endif	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-3 %e \n",d_partRho[10]);
#endif	
	    
	//    return;
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    
            int j = nx, k = ny;
	    double res_sqrt = sqrt(1. + px*px + py*py + pz*pz);
            double Vx = px; // / sqrt(1. + px*px + py*py + pz*pz);
            printf("device-Vx %d %d %25.15e \n",nx,ny,Vx);
            double q2m = f_Q2m*hx*0.5;
//            double f_GammaMax = 1.0;
            double djx,djy,djz,drho;
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,81,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,82,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,83,buf,pz,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,84,buf,res_sqrt,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,85,buf,Vx,l_My*l_Mz);
	    
	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	     write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,0,buf,Vx,l_My*l_Mz);
            
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,1,buf,ex,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,2,buf,ey,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,3,buf,ez,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,4,buf,bx,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,5,buf,by,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,6,buf,bz,l_My*l_Mz);
#endif	
	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT	    
            //cuPrintf("Ex %e %e %e \n",ccc,apc,ccp); 
#endif	    
//	    return;

	    
	    //// Ey
	    
////////////////////////////////////////////////////////////////////////////////	    

/*            double ex1 = ex;
            double ey1 = ey;
            double ez1 = ez;*/
#ifdef CUDA_WRAP_CUPRINTF_ALL
            //cuPrintf("fields interpolated \n");            
#endif	    
//            return;
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,62,buf,ex,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,63,buf,ey,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,64,buf,ez,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,65,buf,q2m,l_My*l_Mz);
#endif
            ex *= q2m;
            ey *= q2m;
            ez *= q2m;
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES            
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,60,buf,ey,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,61,buf,Vx,l_My*l_Mz);
#endif	    
            px += ex/(1.-Vx);
            py += ey/(1.-Vx);
            pz += ez/(1.-Vx);
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
	    //cuPrintf("pxyz %e %e %e \n",px,py,pz);
#endif	
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,48,buf,ex,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,49,buf,q2m,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,7,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,8,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,9,buf,pz,l_My*l_Mz);
#endif	    
//	    return;
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-4 %e \n",d_partRho[10]);
#endif	
	    

            double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!

  //          if (f_GammaMax < gamma)
    //           f_GammaMax = gamma;

            double gamma_r = 1./gamma;																	 //!!!!!!
    
            //bx += bXext;
            //by += bYext;
            //bz += bZext;
            ///////////////////////////////////////////////////////////////////////
            //////////////////////////////////////////////////////////////////////// 
/*            double bx1 = bx;
            double by1 = by;
            double bz1 = bz;*/
            ////cuPrintf("INI-BB by-bz %10.3e\n",by-bz);
            bx = bx*gamma_r*q2m/(1.-Vx);
            by = by*gamma_r*q2m/(1.-Vx);
            bz = bz*gamma_r*q2m/(1.-Vx);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,40,buf,bx,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,41,buf,by,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,42,buf,bz,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,43,buf,gamma,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,44,buf,gamma_r,l_My*l_Mz);
	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,53,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,54,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,55,buf,pz,l_My*l_Mz);
#endif	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
//	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,30,buf,bz,l_My*l_Mz);
#endif	    
	    
            ////cuPrintf("BB by-bz %10.3e\n",by-bz);
            double co = 2./(1. + (bx*bx) + (by*by) + (bz*bz));
	    
	    ////cuPrintf("DIFF pz-py %e bz-by %e \n",pz-py,bz-by);
	    
	    ////cuPrintf("PZ pz %e bx %e px %e by %e px %e bz %e py %e bx %e \n",pz,bx,px,by,px,bz,py,bx);
	    ////cuPrintf("SOURCE pz*bx-px*by %e px*bz-py*bx %e \n",pz*bx-px*by,px*bz-py*bx);
            ////cuPrintf("VECTORPRODUCT %10.3e py-pz %10.3e \n",(pz*bx - px*bz)-(px*by - py*bx),py-pz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
//	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,57,buf,px);   
//	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,58,buf,py);   
//	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,59,buf,pz);   
#endif		    
            double p3x = py*bz - pz*by + px;
            double p3y = pz*bx - px*bz + py;
            double p3z = px*by - py*bx + pz;
	    
	    ////cuPrintf("P3 py-pz %10.3e\n",p3y-p3z);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
//            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,31,buf,p3z);
#endif	    
            p3x *= co;
            p3y *= co;
            p3z *= co;
	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,37,buf,p3x,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,38,buf,p3y,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,39,buf,p3z,l_My*l_Mz);
#endif
            double px_new = p3y*bz - p3z*by;
            double py_new = p3z*bx - p3x*bz;
            double pz_new = p3x*by - p3y*bx;
	    //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,32,buf,pz_new,l_My*l_Mz);
            ////cuPrintf("NEW py-pz %10.3e\n",py_new-pz_new);     
            px += ex/(1.-Vx) + px_new;
            py += ey/(1.-Vx) + py_new;
            pz += ez/(1.-Vx) + pz_new;
	    ////cuPrintf("PY py-pz %10.3e\n",py-pz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,50,buf,ex,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,51,buf,Vx,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,52,buf,px_new,l_My*l_Mz);
	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,45,buf,px_new,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,46,buf,py_new,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,47,buf,pz_new,l_My*l_Mz);
	    
	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,10,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,11,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,12,buf,pz,l_My*l_Mz);
#endif	    
            //return;
            

            /////////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            gamma = 1./sqrt(1. + px*px + py*py + pz*pz);
            Vx = px*gamma;
            double Vy = py*gamma;
            double Vz = pz*gamma;


            djx = weight*djx0*Vx;
            djy = weight*djy0*Vy;
            djz = weight*djz0*Vz;
            drho = weight*drho0;
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,91,buf,y,l_My*l_Mz);
#endif	    
            double xtmp = 1.;
            double ytmp = y;
            double ztmp = z;

//            double full = 1.;
            double part_step = 1.;

            int itmp = 0;
            int jtmp = nx;
            int ktmp = ny;

            djx *= 1./(1.-Vx);
            djy *= 1./(1.-Vx);
            djz *= 1./(1.-Vx);
            drho *= 1./(1.-Vx);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,92,buf,ytmp,l_My*l_Mz);

            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,33,buf,djz,l_My*l_Mz);
#endif	    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-4 %e \n",d_partRho[10],l_My*l_Mz);
#endif	

            double dy = py*gamma/(1.-Vx)*hx/hy;
            double dz = pz*gamma/(1.-Vx)*hx/hz;
            ////cuPrintf("GaMMA dy-dz %10.3e py-pz %10.3e\n",dy-dz,py-pz);

            ////cuPrintf("almost computed \n");            
           // return;

	    dy = dy/2*(!iFullStep) + iFullStep * dy;
	    dz = dz/2*(!iFullStep) + iFullStep * dz;
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
            //cuPrintf("MULT dy-dz %10.3e \n",dy-dz);
#endif	    

            double partdx = 0.;
            double step = 1.;
            // --- first half-step

            if (fabs(dy)>1. || fabs(dz)>1.) {
               if (fabs(dy) > fabs(dz)) {
                  step = partdx = fabs(dy);
               } else {
                  step = partdx = fabs(dz);
               };
            }
            ////cuPrintf("IF dy-dz %10.3e \n",dy-dz);
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES            
            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,13,buf,dy,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,14,buf,dz,l_My*l_Mz);
#endif
	    
#ifdef CUDA_WRAP_CUPRINTF_ALL	    
            //cuPrintf("after if\n");
#endif	    
//            return;

            if (partdx < 1.) {
               partdx = step = 1.;
            }
#ifdef CUDA_WRAP_CUPRINTF_ALL
            //cuPrintf("while \n");  
#endif
	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES	    
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,66,buf,partdx,l_My*l_Mz);
            write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,67,buf,xtmp,l_My*l_Mz);
#endif	    

#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5 %e \n",d_partRho[10]);
#endif	
	    
             write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,93,buf,ytmp,l_My*l_Mz);

            while (partdx>0.) 
	    {
                  if (partdx > 1.) 
		  {
                      partdx -= 1.;
                      part_step = 1./step;
                  } 
                  else 
		  {
                      part_step = partdx/step;
                      partdx = 0.;
                  }
                  ////cuPrintf("part_step in while %e\n",part_step);
                  xtmp = 0.;
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,94,buf,ytmp,l_My*l_Mz);

                  ytmp += dy*part_step;
                  ztmp += dz*part_step;
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,95,buf,ytmp,l_My*l_Mz);
		  
		  
		  
		  ////cuPrintf("ytmp-ztmp %10.3e dy-dz %10.3e \n",ytmp-ztmp,dy-dz);
		  
                  int j_jump = ytmp;
                  int k_jump = ztmp;
                  if (ytmp < 0.) 
		  {
                      j_jump--;
                  };
                  if (ztmp < 0.) 
		  {
                      k_jump--;
                  };
                  jtmp = j + j_jump;
                  ktmp = k + k_jump;
                  if (jtmp < 0) jtmp = -1;
                  if (jtmp > l_My-1) jtmp = l_My;
                  if (ktmp < 0) ktmp = -1;
                  if (ktmp > l_Mz-1) ktmp = l_Mz;
                  ytmp -= j_jump;
                  ztmp -= k_jump;
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,96,buf,ytmp,l_My*l_Mz);

/*                  if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) 
		  {
                     double checkpoint21 = 0.;
                  };*/

                  int ntmp = ncell(jtmp,ktmp);
		  int isort = 0;

/*                  if (fabs(djy) > 0.) 
		  {
                      int check = 0;
                  };*/
                  double part_djx = djx*part_step;
                  double part_djy = djy*part_step;
                  double part_djz = djz*part_step;
                  double part_drho = drho*part_step;
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES		  
		//  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,34,buf,part_djz);
		//  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,35,buf,Vz);
#endif		  
		  
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES		  
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,56,buf,px,l_My*l_Mz);
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,57,buf,py,l_My*l_Mz);
                  write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,58,buf,pz,l_My*l_Mz);
#endif
		  
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5.1 %e \n",d_partRho[10]);
#endif            
	          if (!iFullStep) 
		  {
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5.2 %e \n",d_partRho[10]);
#endif


// __device__ void cuDepositCurrentsInCell(int l_My,int l_Mz,int part_number,double *buf,
//                       double *d_partJy,double *d_partJz,
//                       int isort,
//                       int i, int nx, int ny,
//                       double Vx, double Vy, double Vz,
//                       double x, double y, double z,
//                       double djx, double djy, double djz, double drho)

            cuDepositCurrentsInCell(l_My,l_Mz,part_number,buf,d_partJy,d_partJz,isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp,
                         part_djx, part_djy, part_djz, part_drho);
            //write currents section
           // CUDA_WRAP_writeXYSection(l_My,l_My,l_Mz,hx,hy,part_djx,"Jx",step,mesh,p_CellArray,0);
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5.3 %e \n",d_partRho[10]);
#endif		    
	
                  } 
                  else 
		  {
		    
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5A %e \n",d_partRho[10]);


#endif


// __device__ void cuDepositRhoInCell(int l_My,int l_Mz,int part_number,double *buf,
//                       double *d_partJx,double *d_partRho,
//                       int isort,
//                       int i, int nx, int ny,
//                       double Vx, double Vy, double Vz,
//                       double x, double y, double z,
//                       double djx, double djy, double djz, double drho)



   cuDepositRhoInCell(l_My,l_Mz,part_number,buf, d_partJx,d_partRho,isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp,
                         part_djx, part_djy, part_djz, part_drho);
   //write rho section, m.b. currents too

#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-5B %e \n",d_partRho[10]);
#endif		     
                  } 
            }
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-A6 %e \n",d_partRho[10]);
#endif            
#ifdef CUDA_WRAP_CUPRINTF_ALL            
            //cuPrintf("after while %e %e \n",d_partJy[ncell(nx,ny)],d_partJz[ncell(nx,ny)]);            
#endif
	    
#ifdef CUDA_WRAP_CHECK_PARTICLE_VALUES            
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,15,buf,xtmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,16,buf,ytmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,17,buf,ztmp,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,29,buf,px,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,18,buf,py,l_My*l_Mz);
	    write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,19,buf,pz,l_My*l_Mz);
#endif	    
            //return;

            if (iFullStep) {
               xtmp = 0.;
	       writeParticle(partSurfOut,
                         cell_number,part_number,xtmp,ytmp,ztmp,px,py,pz,weight,f_Q2m,l_My*l_Mz);
	       write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,86,buf,xtmp,l_My*l_Mz);
	       write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,87,buf,ytmp,l_My*l_Mz);
	       write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,88,buf,ztmp,l_My*l_Mz);
	       
/*               long nnew = GetN(i-1,jtmp,ktmp);
               Cell &cnew = p_CellArray[nnew];
               p->p_Next = cnew.p_Particles;
               cnew.p_Particles = p;
               p->l_Cell = nnew;
               ccc.p_Particles = p_next; */
            }
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
	//cuPrintf("rho-6 %e \n",d_partRho[10]);
#endif	
            
            
//            p = p_next;


            

                /////////////////////////////////////////////////////////////////////////////////////////////////////////////
                
        }
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT        
            //cuPrintf("finished \n");            
#endif	    

}


int CUDA_WRAP_create_particle_surface(double *surf,
                                      double **surf_array,int width,int height,double *h_data_in)
{
        int size = width*height*sizeof(double);
	int err = cudaGetLastError();
	
// 	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
        
	err = cudaGetLastError();
		
    err = cudaMalloc(surf_array, width*height*sizeof(double));

    err = cudaMemcpy(*surf_array, h_data_in, size, cudaMemcpyHostToDevice);
        
        // Bind the arrays to the surface references 
        //err = cudaBindSurfaceToArray( surf, *surf_array);

        return 0;
}

int CUDA_WRAP_get_particle_surface(double *surf,
                                   double *surf_array,int width,int height,double *h_data_in)
{
        int size = width*height*sizeof(double);


        cudaMemcpy(h_data_in,surf_array, size, cudaMemcpyDeviceToHost);
        
        // Bind the arrays to the surface references 
        //cudaBindSurfaceToArray( surf, surf_array); 

        return 0;
}

int CUDA_WRAP_create_surface_fromDevice(
   double * surf,
   double *surf_array,int width,int height,double *d_data_in)
{
        int size = width*height*sizeof(double);

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned); 
        
    cudaMalloc(&surf_array, size);

    cudaMemcpy(surf_array,d_data_in, size, cudaMemcpyDeviceToDevice);
        
        // Bind the arrays to the surface references 
//         cudaBindSurfaceToArray( surf, surf_array);

        return 0;
}


int findMaxNumberOfParticlesPerCell(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
    int max = 0,num,np;
    
    for (int i = 0;i < Ny;i++)
    {
        for (int j = 0;j < Nz;j++)
	{
	    num = mesh->GetN(i_layer,i,j);
	    Cell & ccc = p_CellArray[num];
	    Particle *p  = ccc.GetParticles();
	    np = 0;
	    while (p)
	    {
	        np++;
		p = p->p_Next;
	    }
	    if(np > max) max = np;
	}
    }
  
    return max;
}

int WriteParticleAttribute(int cell_number,int part_per_cell_max,int partNumber,int attr_number, double *h_data_in,double t)
{
//    printf("WriteParticleAttribute cn %d pmax %d  pn %d atn %d total %d \n",cell_number,part_per_cell_max,partNumber,attr_number,
//       cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + partNumber*NUMBER_ATTRIBUTES + attr_number
    //);
    printf("%d cell %d pn %d attr %d\n",cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + partNumber*NUMBER_ATTRIBUTES + attr_number,cell_number,partNumber,attr_number);
    //                       NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START)*width       
    h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + partNumber*NUMBER_ATTRIBUTES + attr_number] = t;
    return 0;
}

int writeFieldComponent(int cell_number,int part_per_cell_max, double *h_data_in,int num,
			double ccc,double cpc,double ccp,double cpp,double cpm,double cmp,double cmc,double ccm,double cmm)
{
    WriteParticleAttribute(cell_number,part_per_cell_max,num,0,h_data_in,ccc);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,1,h_data_in,cpc);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,2,h_data_in,ccp);

    WriteParticleAttribute(cell_number,part_per_cell_max,num,3,h_data_in,cpp);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,4,h_data_in,cpm);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,5,h_data_in,cmp);

    WriteParticleAttribute(cell_number,part_per_cell_max,num,6,h_data_in,cmc);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,7,h_data_in,ccm);
    WriteParticleAttribute(cell_number,part_per_cell_max,num,8,h_data_in,cmm);
    
    printf("FieldComponent cell %d ccc %e cpc %e ccp %e cpp %e cpm %e cmp %e cmc %e ccm %e cmm %e \n");
}




int WriteCell(int cell_number,int part_per_cell_max, double *h_data_in,
	      Cell &ccc,Cell & cpc,Cell & ccp,Cell & cpp,Cell & cpm,Cell & cmp,Cell & cmc,Cell & ccm,Cell & cmm)
{
    //h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + partNumber*NUMBER_ATTRIBUTES + attr_number] = t;
    Particle *p  = ccc.GetParticles();
     
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,0,
			ccc.GetEx(),cpc.GetEx(),ccp.GetEx(),cpp.GetEx(),cpm.GetEx(),cmp.GetEx(),cmc.GetEx(),ccm.GetEx(),cmm.GetEx());
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,1,
			ccc.GetEy(),cpc.GetEy(),ccp.GetEy(),cpp.GetEy(),cpm.GetEy(),cmp.GetEy(),cmc.GetEy(),ccm.GetEy(),cmm.GetEy());
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,2,
			ccc.GetEz(),cpc.GetEz(),ccp.GetEz(),cpp.GetEz(),cpm.GetEz(),cmp.GetEz(),cmc.GetEz(),ccm.GetEz(),cmm.GetEz());
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,3,
			ccc.GetBx(),cpc.GetBx(),ccp.GetBx(),cpp.GetBx(),cpm.GetBx(),cmp.GetBx(),cmc.GetBx(),ccm.GetBx(),cmm.GetBx());
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,4,
			ccc.GetBy(),cpc.GetBy(),ccp.GetBy(),cpp.GetBy(),cpm.GetBy(),cmp.GetBy(),cmc.GetBy(),ccm.GetBy(),cmm.GetBy());
    writeFieldComponent(cell_number,part_per_cell_max,h_data_in,5,
			ccc.GetBz(),cpc.GetBz(),ccp.GetBz(),cpp.GetBz(),cpm.GetBz(),cmp.GetBz(),cmc.GetBz(),ccm.GetBz(),cmm.GetBz());

//    return 0;
    
    for(int k =  CUDA_WRAP_PARTICLE_START;k < part_per_cell_max;k++)
    {
      
        if(p != NULL)
        {
   	   //printf("cell ny %d nz %d particle %d INIT: x %.2e y %.2e z %.2e px %.2e py %.2e pz %.2e \n",i,j,k,p->f_X,p->f_Y,p->f_Z,p->f_Px, p->f_Py,p->f_Pz);
	   int pn  = k;			
  	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,0,h_data_in,p->f_X);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,1,h_data_in,p->f_Y);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,2,h_data_in,p->f_Z);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,3,h_data_in,p->f_Px);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,4,h_data_in,p->f_Py);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,5,h_data_in,p->f_Pz);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,6,h_data_in,p->f_Weight);
	   WriteParticleAttribute(cell_number,part_per_cell_max,pn,7,h_data_in,p->f_Q2m);
           p = p->p_Next;
        }
        else
        {
        }
    }   
    
    return 0;
}

int CUDA_WRAP_fill_particle_attributes(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
        int part_per_cell_max,cell_number,size;
        int width = Ny*Nz; 
        double *h_data_in;//   = (double*) malloc(NUMBER_ATTRIBUTES*(part_per_cell_max+1)*width*sizeof(double));
        
        if(fillAttributesFirstCall == 1)
	{
	    fillAttributesFirstCall = -1;
	}
	else
	{
	    return 1;
	} 
	int err = cudaGetLastError();
	
	part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	size = NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START)*width;
	h_data_in   = (double*) malloc(NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START)*width*sizeof(double));
	printf("particle host array size %d \n",NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START)*width);
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = i*Nz + j;
		
	        //int num      = mesh->GetN(i_layer,i,j);
		long nccc = mesh->GetN(i_layer,  i,  j);
                long ncpc = mesh->GetN(i_layer,  i+1,j);
                long nccp = mesh->GetN(i_layer,  i,  j+1);
                long ncpp = mesh->GetN(i_layer,  i+1,j+1);
                long ncmc = mesh->GetN(i_layer,  i-1,j);
                long nccm = mesh->GetN(i_layer,  i,  j-1);
                long ncmm = mesh->GetN(i_layer,  i-1,j-1);
                long ncmp = mesh->GetN(i_layer,  i-1,j+1);
                long ncpm = mesh->GetN(i_layer,  i+1,j-1);
	        Cell &ccc = p_CellArray[nccc];
                Cell &cpc = p_CellArray[ncpc];
                Cell &ccp = p_CellArray[nccp];
                Cell &cpp = p_CellArray[ncpp];
                Cell &cmc = p_CellArray[ncmc];
                Cell &ccm = p_CellArray[nccm];
                Cell &cmm = p_CellArray[ncmm];
                Cell &cmp = p_CellArray[ncmp];
                Cell &cpm = p_CellArray[ncpm];
		WriteCell(cell_number,part_per_cell_max+CUDA_WRAP_PARTICLE_START,h_data_in,ccc,cpc,ccp,cpp,cpm,cmp,cmc,ccm,cmm);
		/*
	        Particle *p  = ccc.GetParticles();
		
	           for(int k = 0;k < part_per_cell_max;k++)
	           {
		      if(p != NULL)
		      {
			  printf("cell ny %d nz %d particle %d INIT: x %.2e y %.2e z %.2e px %.2e py %.2e pz %.2e \n",i,j,k,p->f_X,p->f_Y,p->f_Z,p->f_Px, p->f_Py,p->f_Pz);
				
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,0,h_data_in,p->f_X);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,1,h_data_in,p->f_Y);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,2,h_data_in,p->f_Z);

			  WriteParticleAttribute(cell_number,part_per_cell_max,k,3,h_data_in,p->f_Px);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,4,h_data_in,p->f_Py);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,5,h_data_in,p->f_Pz);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,6,h_data_in,p->f_Weight);
			  WriteParticleAttribute(cell_number,part_per_cell_max,k,7,h_data_in,p->f_Q2m);
			  
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 3] = p->f_Px;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 4] = p->f_Py;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 5] = p->f_Pz;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 6] = p->f_Weight;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 7] = p->f_Q2m;
		          p = p->p_Next;
		      }
		      else
		      {
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 0] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 1] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 2] = 0.0;
		      
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 3] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 4] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 5] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 6] = 0.0;
		          h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 7] = 0.0;
		      }
		   }*/
	     }  
        }
        CUDA_WRAP_create_particle_surface(partSurfIn,&cuInputArrayX,NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START),width,h_data_in);
	
	CUDA_WRAP_create_particle_surface(partSurfOut,&cuOutputArrayX,NUMBER_ATTRIBUTES*(part_per_cell_max+CUDA_WRAP_PARTICLE_START),width,h_data_in);
	
// 	cudaDeviceSynchronize();
	
//	free(h_data_in);
  
  
        return part_per_cell_max;
}

int CUDA_WRAP_check_particle_attributes(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
        int part_per_cell_max,cell_number;
        int de_facto_particles = 0;
        int wrong_particles = 0.0;
        int width = Ny*Nz; 
//         double *h_data_in;
	
#ifdef CUDA_WRAP_PARTICLE_CONTROL_FILE
	FILE *f;
	if((f = fopen("particle_host.dat","at")) == NULL) return EOF;
#endif

#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS		
	puts("BEGIN PARTICLE CHECK =============================================================================");
#endif	
	
	part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	double *h_data_in   = (double*) malloc(NUMBER_ATTRIBUTES*part_per_cell_max*width*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	cudaMemcpy(h_data_in,d_particleResult,NUMBER_ATTRIBUTES*part_per_cell_max*width*sizeof(double),cudaMemcpyDeviceToHost);
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = i*Ny + j;
		
	        int num      = mesh->GetN(i_layer,i,j);
	        Cell & ccc   = p_CellArray[num];
	        Particle *p  = ccc.GetParticles();
		
	           for(int k = 0;(k < part_per_cell_max) && (p != NULL);k++)
	           {
		      if(p != NULL)
		      {
			  double x,y,z,px,py,pz;
			  double cu_x,cu_y,cu_z,cu_px,cu_py,cu_pz;
			  
			  de_facto_particles++;
			  
		          x     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 0];
			  cu_x  = p->f_X;
		          y     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 1];
			  cu_y  = p->f_Y;
		          z     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 2];
			  cu_z  = p->f_Z;
		      
		          px    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 3]; 
		          cu_px = p->f_Px;
		          py    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 4];
			  cu_py = p->f_Py;
		          pz    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 5];
		          cu_pz = p->f_Pz;
			  
			  if((  fabs(x-cu_x) > TOLERANCE_IDEAL) ||   (fabs(y-cu_y) > TOLERANCE_IDEAL) ||   (fabs(z-cu_z) > TOLERANCE_IDEAL) ||
			     (fabs(px-cu_px) > TOLERANCE_IDEAL) || (fabs(py-cu_py) > TOLERANCE_IDEAL) || (fabs(pz-cu_pz) > TOLERANCE_IDEAL)   )
			  {
			      wrong_particles++;
#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS			      
			      printf("cell ny %d nz %d particle %d wrong: x %.2e/%.2e y %.2e/%.2e z %.2e/%.2e px %.2e/%.2e py %.2e/%.2e pz %.2e/%.2e \n",
				     i,j,k,x,cu_x,y,cu_y,z,cu_z,px,cu_px,py,cu_py,pz,cu_pz);
#endif			      
			  }
			  else
			  {
#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS			      
			      printf("cell ny %d nz %d particle %d OK   : x %.2e/%.2e y %.2e/%.2e z %.2e/%.2e px %.2e/%.2e py %.2e/%.2e pz %.2e/%.2e \n",
				     i,j,k,x,cu_x,y,cu_y,z,cu_z,px,cu_px,py,cu_py,pz,cu_pz);
#endif			      
			    
			  }
#ifdef CUDA_WRAP_PARTICLE_CONTROL_FILE
                          fprintf(f,"%10d %5d %5d %3d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n",i_layer,i,j,k,cu_x,cu_y,cu_z,cu_px,cu_py,cu_pz); 
#endif
		          p = p->p_Next;
		      }
		   }
	     }  
        }
	
	free(h_data_in);
	
	printf("PARTICLE CHECK OK %.2f wrong %.2f ==============================================================\n",
	       1.0-((double)wrong_particles)/de_facto_particles,((double)wrong_particles)/de_facto_particles);
	
#ifdef CUDA_WRAP_PARTICLE_CONTROL_FILE  
        fclose(f);
#endif
        return 0;
}

int CUDA_WRAP_alloc_particle_values(int Ny,int Nz,int num_attr,int ppc_max,double **h_p,double **d_p)
{
	
	*h_p   = (double*) malloc(num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	cudaMalloc((void**)d_p,num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	cudaMemset(*d_p,0,num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	memset(*h_p,0,num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	return 0;
}

int CUDA_WRAP_write_particle_value(int Ny,int i,int j,int num_attr,int ppc_max,int k,int n,double *h_p,double t)
{
	int cell_number = i*Ny + j;
	
	h_p [cell_number*num_attr*ppc_max + k*num_attr + n] = t;
	
	//cudaMemcpy((void**)d_p,num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	return 0;
}


double CUDA_WRAP_check_particle_values(int iLayer,int Ny,int Nz,int num_attr,int ppc_max,double *h_p,double *d_p)
{
        int cell_number,wrong_particles = 0;
	double    *h_copy,frac_err,delta = 0.0,*wrong_array,*delta_array;
	
	wrong_array = (double *)malloc(num_attr*sizeof(double));
	delta_array = (double *)malloc(num_attr*sizeof(double));
//        int width = Ny*Nz; 
//        double *h_data_in;
	
	puts("BEGIN PARTICLE-RELATED VALUES sCHECK =============================================================================");
	
	//part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	h_copy   = (double*) malloc(num_attr*ppc_max*Ny*Nz*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	cudaMemcpy(h_copy,d_p,num_attr*ppc_max*Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);

    for(int n = 0;n < num_attr;n++)
    {
        int wpa = 0;
	double fr_attr,x,cu_x;
	
	delta = 0.0;
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = i*Ny + j;
		
	        for(int k = 0;k < ppc_max;k++)
	        {
		  
		          cu_x = h_copy[cell_number*num_attr*ppc_max + k*num_attr + n];
		          x    = h_p[cell_number*num_attr*ppc_max + k*num_attr + n];
			  
#ifdef CUDA_WRAP_PARTICLE_VALUES_DETAILS
			  if((fabs(x-cu_x) > PARTICLE_TOLERANCE)  )
			  {
			     printf("%5d %5d %5d %25.15e/%25.15e delta %15.5e \n",n,i,j,x,cu_x,fabs(cu_x - x));
			  }
#endif			
                          if(delta < fabs(cu_x - x)) delta = fabs(cu_x - x); 
			  
			  if(  fabs(x-cu_x) > PARTICLE_TOLERANCE)
			  {
			      wrong_particles++;
			      wpa++;
			  //    printf("cell ny %d nz %d particle %d wrong: x %.2e/%.2e %15.5e\n",i,j,k,x,cu_x,fabs(x-cu_x));
			  }
			  else
			  {
			    //  printf("cell ny %d nz %d particle %d OK: x %.2e/%.2e %15.5e\n",i,j,k,x,cu_x,fabs(x-cu_x));
			  }
      	        }
	     } 
#ifdef CUDA_WRAP_PARTICLE_VALUES_DETAILS	     
             if(  fabs(x-cu_x) > PARTICLE_TOLERANCE)
	     {
//	        printf("\n");
	     }
#endif	     
        }
        fr_attr = (double)wpa/(Ny*Nz*ppc_max);
        printf("\n Layer %5d value %3d OK %.2f wrong %.2f delta %15.5e \n",iLayer,n,1.0 - fr_attr,fr_attr,delta);
	
	wrong_array[n] = fr_attr;
	delta_array[n] = delta;
	
	puts("___________________________________________________________________________________________________________");
    }
	
	free(h_copy);
	
	frac_err = (double)wrong_particles/(Ny*Nz*num_attr*ppc_max);
	
	
	
	FILE *wf,*df;
	if(write_values_first_call == 1)
	{
	   if((wf = fopen("values_wrong.dat","wt")) == NULL) return 1;
	   if((df = fopen("values_delta.dat","wt")) == NULL) return 1;
	   
	   write_values_first_call = 0;
	}
        else
	{
	   if((wf = fopen("values_wrong.dat","at")) == NULL) return 1;
	   if((df = fopen("values_delta.dat","at")) == NULL) return 1;
	}
	
	fprintf(wf,"Layer %5d ",iLayer);
	fprintf(df,"Layer %5d ",iLayer);
	if(iLayer <= 477)
	{
	   int ig45 = 0;
	}
	double max_delta = 0.0;
	for(int i = 0;i < num_attr;i++)
	{
	    fprintf(wf,"%15.5e ",wrong_array[i]);
	    fprintf(df,"%15.5e ",delta_array[i]);
	    
	    if(max_delta < delta_array[i]) 
	    {
	       max_delta = delta_array[i];
	       last_max_delta_value = i;
	    }
	    
	      
	}
	fprintf(wf,"\n");
	fprintf(df,"\n");
	
	fclose(wf);
	fclose(df);
  
          
	free(wrong_array);
	free(delta_array);
	
	last_wrong = frac_err;
	last_delta = max_delta;
	
	printf("PARTICLE-RELATED CHECK OK %.4f wrong %.4f delta %15.5e =================================================\n",
	       1.0-frac_err,frac_err,max_delta);
	
        return frac_err;
}


int CUDA_WRAP_ClearCurrents(int Ny,int Nz)
{
     cudaMemset(d_partRho,0.0,sizeof(double)*Ny*Nz);
     cudaMemset(d_partJx, 0.0,sizeof(double)*Ny*Nz);
     cudaMemset(d_partJy, 0.0,sizeof(double)*Ny*Nz);
     cudaMemset(d_partJz, 0.0,sizeof(double)*Ny*Nz);
  
     return 0;
}


int allocCurrents(int Ny,int Nz,int iFullStep)
{
     int err1,err2,err3,err4,err_res,err;
     
     if(d_partRho == NULL)
     {
        int err1 = cudaMalloc((void**)&d_partRho,sizeof(double)*Ny*Nz);
	cudaMemset(d_partRho,0.0,sizeof(double)*Ny*Nz);
     }
     if(d_partJx == NULL)
     {
        int err2 = cudaMalloc((void**)&d_partJx, sizeof(double)*Ny*Nz);
	cudaMemset(d_partJx, 0.0,sizeof(double)*Ny*Nz);
     }
     if(d_partJy == NULL)
     {
        int err3 = cudaMalloc((void**)&d_partJy, sizeof(double)*Ny*Nz);
	 cudaMemset(d_partJy, 0.0,sizeof(double)*Ny*Nz);
     }
     if(d_partJz == NULL)
     {
        int err4 = cudaMalloc((void**)&d_partJz, sizeof(double)*Ny*Nz);
	 cudaMemset(d_partJz, 0.0,sizeof(double)*Ny*Nz);
     }
     
     if(iFullStep == 1)
     {
        cudaMemset(d_partRho,0.0,sizeof(double)*Ny*Nz);
        cudaMemset(d_partJx, 0.0,sizeof(double)*Ny*Nz);
     }
     else
     {
        cudaMemset(d_partJy, 0.0,sizeof(double)*Ny*Nz);
        cudaMemset(d_partJz, 0.0,sizeof(double)*Ny*Nz);
     }
     
     return 0;
}

int allocParticleAuxillaries(int width,int height)
{
     int err1,err2,err3,err4,err_res,err;
     
     if(d_particleResult == NULL)
     {
        int err_res = cudaMalloc((void**)&d_particleResult,sizeof(double)*width*height);
     }
     if(d_params == NULL)
     {
        int err = cudaMalloc((void**)&d_params,sizeof(double)*10);   
     }
     
     return 0;
}


int CUDA_WRAP_move_particles(double*partSurfIn,

                             int Ny,int Nz,int part_per_cell_max,double hx,double hy,double hz,double djx0,double djy0, double djz0,double drho0,int i_fs,double *buf)
{
    double params[10];
    int width = Ny*Nz,height = part_per_cell_max*NUMBER_ATTRIBUTES;
    
   // CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy init move ");
  
    params[0] = hx;
    params[1] = hy;
    params[2] = hz;
    params[3] = djx0;
    params[4] = djy0;
    params[5] = djz0;
    params[6] = drho0;
    
    part_per_cell_max = CUDA_WRAP_getMaximalPICnumberOnDevice();
    height = part_per_cell_max*NUMBER_ATTRIBUTES;
    //CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy b alloc ");
    allocCurrents(Ny,Nz,i_fs);
    allocParticleAuxillaries(width,height);
    //CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy a alloc ");
    
    cudaMemcpy(d_params,params,sizeof(double)*10,cudaMemcpyHostToDevice);
    
    
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy begin move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJz,"Jz begin move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJx,"Jx begin move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"Rho begin move ");    

    
    
    // Invoke kernel 
    dim3 dimBlock(8, 8,part_per_cell_max); 
    dim3 dimGrid(Ny/8 ,Nz/8, 1); 

    struct timeval tv2,tv1;
    
//    cudaPrintfInit();
    
    gettimeofday(&tv1,NULL);
    

    double d_t = 0.0; 
 //   testSurfKernel<<<dimGrid, dimBlock>>>(&d_t);
    SetField<<<dimGrid, dimBlock>>>(partSurfIn,d_rEx,d_rEy,d_rEz,d_rBx,d_rBy,d_rBz,Ny*Nz);
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"Rho begin move-1 "); 
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy setField ");
    printf("Rho begin move-1 fullstep %d \n",i_fs);
    moveKernel<<<dimGrid, dimBlock>>>(partSurfIn,partSurfOut,
                                      width, height, part_per_cell_max,Ny,Nz,i_fs,d_params,d_particleResult,buf,d_partRho,d_partJx,d_partJy,d_partJz,d_rEx); // ,hx,hy,hz,djx0,djy0,djz0,drho0);
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"Rho begin move-2 "); 
//     moveKernel<<<dimGrid, dimBlock>>>(width, height, part_per_cell_max,Ny,Nz,i_fs,d_params,d_particleResult,d_partRho,d_partJx,d_partJy,d_partJz, ,hx,hy,hz,djx0,djy0,djz0,drho0);
    if(i_fs == 1) copyParticlesToInputSurface<<<dimGrid, dimBlock>>>(
       partSurfIn,partSurfOut,Ny*Nz);
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"Rho begin move-3 "); 
    gettimeofday(&tv2,NULL);  
    
//    cudaPrintfDisplay(stdout, true);
 //   cudaPrintfEnd();
    
    //                              (int Nx,int Ny,int Nz,int iLayer)
  // CUDA_WRAP_copy_particle_currents(Ny,Nz,0);
 //  CUDA_WRAP_copy_particle_density(Ny,Nz,0);
    
 //   CUDA_WRAP_copy_particle_currents(Ny,Nz);
//    cudaMemcpy(&t,d_t,sizeof(double),cudaMemcpyDeviceToHost);
  //  printf("ttt %e \n",t);
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy end move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJz,"Jz end move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJx,"Jx en move ");
    CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"Rho end move ");    
    
    return 0;
}
/*
int CUDA_WRAP_load_fields_fromDevice(int Ny,int Nz)
{
   CUDA_WRAP_create_surface_fromDevice(SurfEx,cuExInputArray,1,Ny*Nz,d_rEx);
   CUDA_WRAP_create_surface_fromDevice(SurfEy,cuEyInputArray,1,Ny*Nz,d_rEy);
   CUDA_WRAP_create_surface_fromDevice(SurfEz,cuEzInputArray,1,Ny*Nz,d_rEz);
   
   CUDA_WRAP_create_surface_fromDevice(SurfBx,cuBxInputArray,1,Ny*Nz,d_rBx);
   CUDA_WRAP_create_surface_fromDevice(SurfBy,cuByInputArray,1,Ny*Nz,d_rBy);
   CUDA_WRAP_create_surface_fromDevice(SurfEx,cuBxInputArray,1,Ny*Nz,d_rBz);
   
   return 0;
}

int CUDA_WRAP_load_fields(int Ny,int Nz,double *rEx,double *rEy,double *rEz,double *rBx,double *rBy,double *rBz)
{
   CUDA_WRAP_create_particle_surface(SurfEx,cuExInputArray,1,Ny*Nz,rEx);
   CUDA_WRAP_create_particle_surface(SurfEy,cuEyInputArray,1,Ny*Nz,rEy);
   CUDA_WRAP_create_particle_surface(SurfEz,cuEzInputArray,1,Ny*Nz,rEz);
   
   CUDA_WRAP_create_particle_surface(SurfBx,cuBxInputArray,1,Ny*Nz,rBx);
   CUDA_WRAP_create_particle_surface(SurfBy,cuByInputArray,1,Ny*Nz,rBy);
   CUDA_WRAP_create_particle_surface(SurfEx,cuBxInputArray,1,Ny*Nz,rBz);
   
   return 0;
}*/

__device__ void cuDepositCurrentsInCell(int l_My,int l_Mz,int part_number,double *buf,
                      double *d_partJy,double *d_partJz, 
                      int isort, 
                      int i, int nx, int ny, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho)
{
   int nccc,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm;
//   cudaCell ccc,cpc,ccp,cpp,cmc,ccm,cmm,cmp,cpm;
   nccc = ncellDeposit(nx,ny);
   ncpc = ncellDeposit(nx+1,ny);
   nccp = ncellDeposit(nx,ny+1);
   ncpp = ncellDeposit(nx+1,ny+1);
   
   ncmc = ncellDeposit(nx-1,ny);
   nccm = ncellDeposit(nx,ny-1);
   ncmm = ncellDeposit(nx-1,ny-1);
   ncmp = ncellDeposit(nx-1,ny+1);
   ncpm = ncellDeposit(nx+1,ny-1); 

/* Cell &ccc = GetCell(i,j,k);
   Cell &cpc = GetCell(i,j+1,k);
   Cell &ccp = GetCell(i,j,k+1);
   Cell &cpp = GetCell(i,j+1,k+1);
   Cell &cmc = GetCell(i,j-1,k);
   Cell &ccm = GetCell(i,j,k-1);
   Cell &cmm = GetCell(i,j-1,k-1);
   Cell &cmp = GetCell(i,j-1,k+1);
   Cell &cpm = GetCell(i,j+1,k-1);   */
#ifdef CUDA_WRAP_CUPRINTF_IN_OUT   
//    cuPrintf("djy,djz %25.15e %25.15e %10.3e \n",djy,djz,djz-djy);
#endif   
   x = 0.;
   printf("in cuDepositCurrentsInCell\n");
   return;

//   double xs = x - 0.5;
   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.-x;
//   double axp = x;
/*
   double ayc = 0.5 + y - y*y;
   double aym = 0.5*(1-y)*(1-y);
   double ayp = 0.5*y*y;
   double azc = 0.5 + z - z*z;
   double azm = 0.5*(1-z)*(1-z);
   double azp = 0.5*z*z;
   */
   
   double ayc = 1. - ys*ys;
   double aym = 0.5*(ys-1.0)*ys;
   double ayp = 0.5*(1+ys)*ys;

   double azc = 1. - zs*zs;
   double azm = 0.5*(zs-1)*zs;
   double azp = 0.5*(1+zs)*zs;

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;
   double acpm = axc*ayp*azm;
   double acmp = axc*aym*azp;
   double acmc = axc*aym*azc;
   double accm = axc*ayc*azm;
   double acmm = axc*aym*azm;

/*   double apcc = axp*ayc*azc;
   double appc = axp*ayp*azc;
   double apcp = axp*ayc*azp;
   double appp = axp*ayp*azp;
   double appm = axp*ayp*azm;
   double apmp = axp*aym*azp;
   double apmc = axp*aym*azc;
   double apcm = axp*ayc*azm;
   double apmm = axp*aym*azm;
*/
   //double weight = fabs(drho);
   

//   ccc.f_Jx += djx*accc;
//   ccc.f_Jy += djy*accc;
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,81,buf,d_partJy[nccc]);
   
   //d_partJy[nccc] += djy*accc; 
   
   addToMatrix(d_partJy,djy*accc,nx,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,71,buf,djy*accc,l_My*l_Mz);
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,91,buf,d_partJy[nccc]);
   
//   ccc.f_Jz += djz*accc;
   d_partJz[nccc] += djz*accc; 

//   cmc.f_Jx += djx*acmc;
//   cmc.f_Jy += djy*acmc;
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,82,buf,d_partJy[ncmc]);
   //d_partJy[ncmc] += djy*acmc; 
   addToMatrix(d_partJy,djy*acmc,nx-1,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,72,buf,djy*acmc,l_My*l_Mz);
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,92,buf,d_partJy[ncmc]);
   
//   cmc.f_Jz += djz*acmc;
   d_partJz[ncmc] += djz*acmc; 

//   cpc.f_Jx += djx*acpc;
   //cpc.f_Jy += djy*acpc;
  // write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,83,buf,d_partJy[ncpc]);
   
  // d_partJy[ncpc] += djy*acpc; 
   addToMatrix(d_partJy,djy*acpc,nx+1,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,73,buf,djy*acpc,l_My*l_Mz);
  // write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,93,buf,d_partJy[ncpc]);

   //cpc.f_Jz += djz*acpc;
   d_partJz[ncpc] += djz*acpc; 

//   ccm.f_Jx += djx*accm;
   //ccm.f_Jy += djy*accm;
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,84,buf,d_partJy[nccm],l_My*l_Mz);
   
   //d_partJy[nccm] += djy*accm; 
   addToMatrix(d_partJy,djy*accm,nx,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,74,buf,djy*accm,l_My*l_Mz);
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,94,buf,d_partJy[nccm],l_My*l_Mz);
   
   //ccm.f_Jz += djz*accm;
   d_partJz[nccm] += djz*accm; 

//   cmm.f_Jx += djx*acmm;
//   cmm.f_Jy += djy*acmm;
   //write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,85,buf,d_partJy[ncmm],l_My*l_Mz);
   
   //d_partJy[ncmm] += djy*acmm; 
   addToMatrix(d_partJy,djy*acmm,nx-1,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,75,buf,djy*acmm,l_My*l_Mz);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,95,buf,d_partJy[ncmm],l_My*l_Mz);

//   cmm.f_Jz += djz*acmm;
   d_partJz[ncmm] += djz*acmm; 

//   cpm.f_Jx += djx*acpm;
//   cpm.f_Jy += djy*acpm;
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,86,buf,d_partJy[ncpm],l_My*l_Mz);
   
   //d_partJy[ncpm] += djy*acpm; 
   addToMatrix(d_partJy,djy*acpm,nx+1,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,76,buf,djy*acpm,l_My*l_Mz);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,96,buf,d_partJy[ncpm],l_My*l_Mz);

//   cpm.f_Jz += djz*acpm;
   d_partJz[ncpm] += djz*acpm; 

//   ccp.f_Jx += djx*accp;
//   ccp.f_Jy += djy*accp;
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,87,buf,d_partJy[nccp],l_My*l_Mz);
   //d_partJy[nccp] += djy*accp; 
   addToMatrix(d_partJy,djy*accp,nx,ny+1);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,77,buf,djy*accp,l_My*l_Mz);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,97,buf,d_partJy[nccp],l_My*l_Mz);

//   ccp.f_Jz += djz*accp;
   d_partJz[nccp] += djz*accp; 

//   cmp.f_Jx += djx*acmp;
//   cmp.f_Jy += djy*acmp;
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,88,buf,d_partJy[ncmp],l_My*l_Mz);
   
   //d_partJy[ncmp] += djy*acmp; 
   addToMatrix(d_partJy,djy*acmp,nx-1,ny+1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,78,buf,djy*acmp,l_My*l_Mz);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,98,buf,d_partJy[ncmp],l_My*l_Mz);

   
//   cmp.f_Jz += djz*acmp;
   d_partJz[ncmp] += djz*acmp; 

//   cpp.f_Jx += djx*acpp;
   //cpp.f_Jy += djy*acpp;
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,89,buf,d_partJy[ncpp],l_My*l_Mz);
   
   //d_partJy[ncpp] += djy*acpp; 
   addToMatrix(d_partJy,djy*acpp,nx+1,ny+1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,79,buf,djy*acpp,l_My*l_Mz);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,99,buf,d_partJy[ncpp],l_My*l_Mz);

   //cpp.f_Jz += djz*acpp;
   d_partJz[ncpp] += djz*acpp; 
}

__device__ void cuDepositRhoInCell(int l_My,int l_Mz,int part_number,double *buf,
                      double *d_partJx,double *d_partRho,  
                      int isort, 
                      int i, int nx, int ny, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho)
{
   int nccc,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm;
//   cudaCell ccc,cpc,ccp,cpp,cmc,ccm,cmm,cmp,cpm;
   nccc = ncellDeposit(nx,ny);
   ncpc = ncellDeposit(nx+1,ny);
   nccp = ncellDeposit(nx,ny+1);
   ncpp = ncellDeposit(nx+1,ny+1);
   ncmc = ncellDeposit(nx-1,ny);
   nccm = ncellDeposit(nx,ny-1);
   ncmm = ncellDeposit(nx-1,ny-1);
   ncmp = ncellDeposit(nx-1,ny+1);
   ncpm = ncellDeposit(nx+1,ny-1); 
   x = 0.;
   
   double xs = x - 0.5;
   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.-x;
   double axp = x;
/*
   double ayc = 0.5 + y - y*y;
   double aym = 0.5*(1-y)*(1-y);
   double ayp = 0.5*y*y;
   double azc = 0.5 + z - z*z;
   double azm = 0.5*(1-z)*(1-z);
   double azp = 0.5*z*z;
*/

   double ayc = 1. - ys*ys;
   double aym = 0.5*(ys-1)*ys;
   double ayp = 0.5*(1+ys)*ys;

   double azc = 1. - zs*zs;
   double azm = 0.5*(zs-1)*zs;
   double azp = 0.5*(1+zs)*zs;

   double accc = axc*ayc*azc;
   double acpc = axc*ayp*azc;
   double accp = axc*ayc*azp;
   double acpp = axc*ayp*azp;
   double acpm = axc*ayp*azm;
   double acmp = axc*aym*azp;
   double acmc = axc*aym*azc;
   double accm = axc*ayc*azm;
   double acmm = axc*aym*azm;

   double apcc = axp*ayc*azc;
   double appc = axp*ayp*azc;
   double apcp = axp*ayc*azp;
   double appp = axp*ayp*azp;
   double appm = axp*ayp*azm;
   double apmp = axp*aym*azp;
   double apmc = axp*aym*azc;
   double apcm = axp*ayc*azm;
   double apmm = axp*aym*azm;
/////////////////////////////////////
   
 //  //cuPrintf("ny %d nccc %2d accc %e acmc %e acpc %e accm %e ",);
   double weight = fabs(drho);
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,100,buf,weight,l_My*l_Mz);
   
   //cuPrintf("rho-alpha ny %d nx %d %e \n",ny,nx,d_partRho[nccc]);
   //cuPrintf("nccc %d ncpc %d nccp %d ncpp %d ncmc %d nccm %d ncmm %d ncmp %d ncpm %d \n",nccc,ncpc,nccp,ncpp,ncmc,nccm,ncmm,ncmp,ncpm);

   //d_partRho[nccc] += weight*accc;
   addToMatrixRho(d_partRho,weight*accc,nx,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,101,buf,accc,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d A nx %d %e %e\n",ny,nx,d_partRho[nccc],accc);
   
   //d_partRho[ncmc] +=  weight*acmc;
   addToMatrixRho(d_partRho,weight*acmc,nx-1,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,102,buf,acmc,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d B nx %d %e %e %e weight*acmc %e nccc %d ncmc %d\n",ny,nx,d_partRho[nccc],weight,acmc,weight*acmc,nccc,ncmc);
   
   //d_partRho[ncpc] +=  weight*acpc;
   addToMatrixRho(d_partRho,weight*acpc,nx+1,ny);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,103,buf,acpc,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d C nx %d %e \n",ny,nx,d_partRho[nccc],weight,acpc);

   //d_partRho[nccm] +=  weight*accm;
   addToMatrixRho(d_partRho,weight*accm,nx,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,104,buf,accm,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d D nx %d %e \n",ny,nx,d_partRho[nccc]);
   
   //d_partRho[ncmm] +=  1e-7; // weight*acmm;
   addToMatrixRho(d_partRho,weight*acmm,nx-1,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,105,buf,acmm,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d E nx %d %e \n",ny,nx,d_partRho[nccc]);

   //d_partRho[ncpm] +=  1e-9;// weight*acpm;
   addToMatrixRho(d_partRho,weight*acpm,nx+1,ny-1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,106,buf,acpm,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d F nx %d %e \n",ny,nx,d_partRho[nccc]);

   //d_partRho[ncmp] +=  1e-11; // weight*acmp;
   addToMatrixRho(d_partRho,weight*acmp,nx-1,ny+1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,107,buf,acmp,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d G nx %d %e \n",ny,nx,d_partRho[nccc]);

   //d_partRho[nccp] +=  1e-13;// weight*accp;
   addToMatrixRho(d_partRho,weight*accp,nx,ny+1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,108,buf,accp,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d H nx %d %e \n",ny,nx,d_partRho[nccc]);

   //d_partRho[ncpp] += 1e-15; // weight*acpp;
   addToMatrixRho(d_partRho,weight*acpp,nx+1,ny+1);
   
   write_particle_value(l_My,nx,ny,CUDA_WRAP_CONTROL_VALUES,1,part_number,109,buf,acpp,l_My*l_Mz);
   //cuPrintf("rho-beta ny %d J nx %d %25.15e weight %e acpp %e \n",ny,nx,d_partRho[nccc],weight,acpp);

   
   d_partJx[nccc] += djx*accc;
   d_partJx[ncmc] += djx*acmc;
   d_partJx[ncpc] += djx*acpc;
   d_partJx[nccm] += djx*accm;
   d_partJx[ncmm] += djx*acmm;
   d_partJx[ncpm] += djx*acpm;
   d_partJx[ncmp] += djx*acmp;
   d_partJx[nccp] += djx*accp;
   d_partJx[ncpp] += djx*acpp;
   
      //cuPrintf("rho-omega ny %d nx %d %e %e\n",ny,nx,d_partRho[nccc],accc);

   
}

double CUDA_WRAP_getArraysToCompare(char *where,Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
         double *h_Rho,*h_Jx,*h_Jy,*h_Jz;
         int cell_number;
	 double frac_ideal[4],frac_rude;
	 
#ifndef CUDA_WRAP_CURRENTS_CHECK
	 return 0;
#endif	 
	 
	 h_Rho = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jx  = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jy  = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jz  = (double *)malloc(Ny*Nz*sizeof(double));
	 
         for (int i = 0;i < Ny;i++)
         {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = j*Ny + i;
		
	        //int num      = mesh->GetN(i_layer,i,j);
		long nccc = mesh->GetN(i_layer,  i,  j);
	        Cell &ccc = p_CellArray[nccc];
		
		h_Rho[cell_number] = ccc.GetDens0();
		h_Jx[cell_number]  = ccc.GetJx();
		h_Jy[cell_number]  = ccc.GetJy();
		h_Jz[cell_number]  = ccc.GetJz();
	    }
	 }
	// CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"rho");
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJx,"Jx");
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy");
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJz,"Jz");
	 
	 CUDA_WRAP_compare_device_array(Ny*Nz,h_Rho,d_partRho,&frac_ideal[0],&frac_rude,"Rho",where,DETAILS);
	 CUDA_WRAP_compare_device_array(Ny*Nz,h_Jx, d_partJx,&frac_ideal[1], &frac_rude,"Jx",  where,DETAILS);	
	 CUDA_WRAP_compare_device_array(Ny*Nz,h_Jy, d_partJy,&frac_ideal[2], &frac_rude,"Jy",  where,DETAILS);	
	 CUDA_WRAP_compare_device_array(Ny*Nz,h_Jz, d_partJz,&frac_ideal[3], &frac_rude,"Jz",  where,DETAILS);	
    
    free(h_Rho);
    free(h_Jx);
    free(h_Jy);
    free(h_Jz);
	 
    return ((frac_ideal[0] + frac_ideal[1] + frac_ideal[2] + frac_ideal[3])*0.25);
}


int CUDA_WRAP_getHiddenCurrents(char *where,Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
         double *h_Rho,*h_Jx,*h_Jy,*h_Jz;
         int cell_number;
	 double frac_ideal,frac_rude;
	 
#ifndef CUDA_WRAP_CURRENTS_CHECK
	 return 0;
#endif	 
	 printf("%s :: BEGIN CHECK DENSITIES  ============================================================\n",where);
 
         for (int i = 0;i < Ny;i++)
         {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = j*Ny + i;
		
	        //int num      = mesh->GetN(i_layer,i,j);
		long nccc = mesh->GetN(i_layer,  i,  j);
	        Cell &ccc = p_CellArray[nccc];
		
		printf("Hidden host rho %5d %5d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",i,j,ccc.GetDens0(),ccc.GetJx(),ccc.GetJy(),ccc.GetJz(),
		                                                                        ccc.GetDens1(),ccc.GetDens2(),ccc.GetDens() );
	    }
	 }
	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partRho,"hidden device rho");
	// printExplicitRho(Ny,Nz,"explicit Rho from hidden");
	 
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJx,"Jx");
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJy,"Jy");
//	 CUDA_DEBUG_printDdevice_matrix(Ny,Nz,d_partJz,"Jz");
	 
	 printf("%s :: END CHECK DENSITIES ============================================================== \n",where);	 
    return 0;
}


int CUDA_WRAP_EMERGENCY_HIDDEN_CURRENTS_COPY(int Ny,int Nz,Mesh *mesh,Cell *p_CellArray,int i_layer)
{
#ifndef CUDA_WRAP_EMERGENCY_HIDDEN
         return 0;
#endif	 

  
         double *h_Rho,*h_Jx,*h_Jy,*h_Jz;
         int cell_number;
	 double frac_ideal,frac_rude;

	 h_Rho = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jx  = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jy  = (double *)malloc(Ny*Nz*sizeof(double));
	 h_Jz  = (double *)malloc(Ny*Nz*sizeof(double));
	 
	 //printf("%s :: BEGIN CHECK DENSITIES  ============================================================\n",where);
 
         for (int i = 0;i < Ny;i++)
         {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = j*Ny + i;
		
	        //int num      = mesh->GetN(i_layer,i,j);
		long nccc = mesh->GetN(i_layer,  i,  j);
	        Cell &ccc = p_CellArray[nccc];
		
		h_Rho[cell_number] = ccc.GetDens0();
		h_Jx[cell_number]  = ccc.GetJx();
		h_Jy[cell_number]  = ccc.GetJy();
		h_Jz[cell_number]  = ccc.GetJz();
	    }
	 }
	 cudaMemcpy(d_partRho,h_Rho,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_partJx,h_Jx,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_partJy,h_Jy,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_partJz,h_Jz,Ny*Nz*sizeof(double),cudaMemcpyHostToDevice);
	 
         return 0;
}



int print_particle_in_cell(double *partSurfIn,
   int ny,int nz,int Ny,int Nz,double *x,double *y,double *z,double *px,double *py,double *pz)
{
    dim3 dimBlock(Ny, Nz,1); 
    dim3 dimGrid(1 , 1);
    double *d_part,part[6];
    
    cudaMalloc((void **)&d_part,6*sizeof(double));
    
    getParticleFromCell<<<dimGrid, dimBlock>>>(
       partSurfIn,
       ny,nz,d_part);
    
    cudaMemcpy(part,d_part,6*sizeof(double),cudaMemcpyDeviceToHost);
    
    *x  = part[0];
    *y  = part[1];
    *z  = part[2];
    *px = part[3];
    *py = part[4];
    *pz = part[5];
        
    return 0;
}

int CUDA_WRAP_check_on_device(
   double *partSurfIn,
   int Ny,int Nz,Mesh *mesh,int i_layer,Cell *p_CellArray)
{
    double x,y,z,px,py,pz;
    int ny = 1,nz = 1;

    print_particle_in_cell(partSurfIn,ny,nz,Ny,Nz,&x,&y,&z,&px,&py,&pz);

    printf("CELL      %3d %3d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n ",ny,nz,x,y,z,px,py,pz);

#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE    
    long num      = mesh->GetN(i_layer,ny,nz);
    Cell & ccc   = p_CellArray[num];
    Particle *p  = ccc.GetParticles();
    if(p != NULL)
    {
       printf("DEVIATION %3d %3d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n ",ny,nz,fabs(x-p->f_X),fabs(y-p->f_Y),fabs(z-p->f_Z),
	                                                                                 fabs(px-p->f_Px),fabs(py-p->f_Py),(pz-p->f_Pz));
    }
#endif    
    
    
    return 0;
}

int CUDA_WRAP_write_particle_attributes_fromDevice(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *p_CellArray)
{
        int part_per_cell_max,cell_number,de_facto_particles = 0,wrong_particles = 0.0;
        int width = Ny*Nz; 
        double *h_data_in;
	
#ifndef CUDA_WRAP_PARTICLE_DEVICE_CONTROL_FILE
	return 1;
#endif
	
	if(f_part_device == NULL)
	{
	   if((f_part_device = fopen("particle_device.dat","wt")) == NULL) return EOF;
	}
	
	part_per_cell_max = 1; //findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	h_data_in   = (double*) malloc(NUMBER_ATTRIBUTES*part_per_cell_max*width*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	cudaMemcpy(h_data_in,d_particleResult,NUMBER_ATTRIBUTES*part_per_cell_max*width*sizeof(double),cudaMemcpyDeviceToHost);
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        cell_number = i*Ny + j;
		
	           for(int k = 0;(k < part_per_cell_max);k++)
	           {
			  double x,y,z,px,py,pz;
			  
			  de_facto_particles++;
			  
		          x     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 0];
		          y     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 1];
		          z     = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 2];
		      
		          px    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 3]; 
		          py    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 4];
		          pz    = h_data_in [cell_number*NUMBER_ATTRIBUTES*part_per_cell_max + k*NUMBER_ATTRIBUTES + 5];
			  
			    
                          fprintf(f_part_device,"%10d %5d %5d %3d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n",i_layer,i,j,k,x,y,z,px,py,pz); 
		   }
	     }  
        }
	
	free(h_data_in);
	
        //fclose(f);

	return 0;
}








// "hidden" means computed directly with particles and not yet written to array for computing of fields
int CUDA_WRAP_check_hidden_currents1(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *P,double *d_array,char *name,char *fname_base)
{
#ifndef CUDA_WRAP_CHECK_HIDDEN_VALUES
        return 1;
#endif	
  
        int part_per_cell_max,cell_number,de_facto_particles = 0,wrong_particles = 0.0;
        int width = Ny*Nz,wpa = 0; 
        double *h_data_in;
	FILE *f;
	char fname[100];
	
	sprintf(fname,"hidden%s.dat",fname_base);
	
	f = fopen(fname,"wt");
	
	

#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS		
	puts("BEGIN CURRENTS CHECK =============================================================================");
#endif	
	
	double *h_copy   = (double*) malloc(Ny*Nz*sizeof(double));
	
	cudaMemcpy(h_copy,d_array,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        double curr;
	        cell_number = j*Ny + i;
		
		long ncc = mesh->GetNyz(i,j);
//	        int num      = mesh->GetN(i_layer,i,j);
	        Cell & ccc   = P[ncc];
		
		if(!strcmp(name,"Jy")) curr = ccc.GetJy();
		else 
		{
		   if(!strcmp(name,"Jz")) curr = ccc.GetJz();
		   else
		   {
		      if(!strcmp(name,"Jx")) curr = ccc.GetJx();
		      else
		      {
			 if(!strcmp(name,"Rho")) curr = ccc.GetDens(); 
			 else
			 {
                            if(!strcmp(name,"Ex")) curr = ccc.GetEx();
		            else
		            {
                               if(!strcmp(name,"Ey")) curr = ccc.GetEy();
			       else
			       {
                                  if(!strcmp(name,"Ez")) curr = ccc.GetEz();
				  else
				  {
                                     if(!strcmp(name,"Bx")) curr = ccc.GetBx();
	    	                     else
		                     {
                                        if(!strcmp(name,"By")) curr = ccc.GetBy();
			                else
			                {
                                           if(!strcmp(name,"Bz")) curr = ccc.GetBz();
					   else
					   {
					      if(!strcmp(name,"JxBeam")) curr = ccc.GetJxBeam();
					      else
					      {
						 if(!strcmp(name,"RhoBeam")) curr = ccc.GetRhoBeam();
					      }
					   }
			                }
				     }
				    
				  }
			       }
			        
			    }
			 }
		      }
		   }
		   
		}
		double df;
		if((df = fabs(h_copy[cell_number]-curr)) > TOLERANCE_IDEAL)
		{
		   //printf("%s %5d %5d host %25.15e device %25.15e diff %15.5e \n",name,i,j,curr,h_copy[cell_number],df); 
		   wpa++;
		}
		fprintf(f,"%s %5d %5d host %25.15e device %25.15e diff %15.5e \n",name,i,j,curr,h_copy[cell_number],df); 
	    }
	}
#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS		
	printf("current %s OK %10.4f wrong %10.4f \n",name,1-(double)wpa/Ny/Nz,(double)wpa/Ny/Nz);
#endif	
	fclose(f);
	
	return 0;
}

// "hidden" means computed directly with particles and not yet written to array for computing of fields
int CUDA_WRAP_check_hidden_currents3D(Mesh *mesh,Cell *p_layer,int i_layer,int Ny,int Nz,double *d_array,char *name)
{
  
#ifndef CUDA_WRAP_HIDDEN_CURRENTS_CHECK
        return 0;
#endif	
        int part_per_cell_max,cell_number,de_facto_particles = 0,wrong_particles = 0.0;
        int width = Ny*Nz,wpa = 0; 
        double *h_data_in;
	FILE *f;
	char fname[100];
	
	sprintf(fname,"hidden%s.dat",name);
	
	f = fopen(fname,"wt");
	
	

#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS		
	puts("BEGIN CURRENTS CHECK =============================================================================");
#endif	
	
	double *h_copy   = (double*) malloc(Ny*Nz*sizeof(double));
	
	cudaMemcpy(h_copy,d_array,Ny*Nz*sizeof(double),cudaMemcpyDeviceToHost);
	
        for (int i = 0;i < Ny;i++)
        {
            for (int j = 0;j < Nz;j++)
	    {
	        double curr;
	        cell_number = j*Ny + i;
		
		long ncc = mesh->GetN(i_layer,i,j);
//	        int num      = mesh->GetN(i_layer,i,j);
	        Cell & ccc   = p_layer[ncc];
		
		if(!strcmp(name,"Jy")) curr = ccc.GetJy();
		else 
		{
		   if(!strcmp(name,"Jz")) curr = ccc.GetJz();
		   else
		   {
		      if(!strcmp(name,"Jx")) curr = ccc.GetJx();
		      else
		      {
			 if(!strcmp(name,"Rho")) curr = ccc.GetDens(); 
			 else
			 {
                            if(!strcmp(name,"Ex")) curr = ccc.GetEx();
		            else
		            {
                               if(!strcmp(name,"Ey")) curr = ccc.GetEy();
			       else
			       {
                                  if(!strcmp(name,"Ez")) curr = ccc.GetEz();
				  else
				  {
                                     if(!strcmp(name,"Bx")) curr = ccc.GetBx();
	    	                     else
		                     {
                                        if(!strcmp(name,"By")) curr = ccc.GetBy();
			                else
			                {
                                           if(!strcmp(name,"Bz")) curr = ccc.GetBz();
					   else
					   {
					      if(!strcmp(name,"JxBeam")) curr = ccc.GetJxBeam();
					      else
					      {
						 if(!strcmp(name,"RhoBeam")) curr = ccc.GetRhoBeam();
					      }
					   }
			                }
				     }
				    
				  }
			       }
			        
			    }
			 }
		      }
		   }
		   
		}
		double df;
		if((df = fabs(h_copy[cell_number]-curr)) > TOLERANCE_IDEAL)
		{
		   //printf("%s %5d %5d host %25.15e device %25.15e diff %15.5e \n",name,i,j,curr,h_copy[cell_number],df); 
		   wpa++;
		}
		fprintf(f,"%s %5d %5d host %25.15e device %25.15e diff %15.5e \n",name,i,j,curr,h_copy[cell_number],df); 
	    }
	}
#ifdef CUDA_WRAP_COMPARE_PARTICLE_TRACE_DETAILS		
	printf("current %s OK %10.4f wrong %10.4f \n",name,1-(double)wpa/Ny/Nz,(double)wpa/Ny/Nz);
#endif	
	
	return 0;
}



int CUDA_WRAP_check_hidden_currents(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *P,double *d_array,char *name)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif  
    return CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,d_array,name,name);
}

int CUDA_WRAP_check_all_hidden_fields(Mesh *mesh,int i_layer,int Ny,int Nz,Cell *C,Cell *P,cudaLayer *h_cl,cudaLayer *h_pl)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED     
     return 0;
#endif  
  
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->Ex,"Ex","ExP");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->Ey,"Ey","EyP");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->Ez,"Ez","EzP");

  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->Bx,"Bx","BxP");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->By,"By","ByP");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,P,h_pl->Bz,"Bz","BzP");

  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->Ex,"Ex","ExC");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->Ey,"Ey","EyC");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->Ez,"Ez","EzC");

  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->Bx,"Bx","BxC");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->By,"By","ByC");
  CUDA_WRAP_check_hidden_currents1(mesh,i_layer,Ny,Nz,C,h_cl->Bz,"Bz","BzC");
  
  return 0;
}

int CUDA_WRAP_copyBeamToArray(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray,double **d_rho_beam,double **d_jx_beam)
{
    double *h_rho_beam,*h_jx_beam;
    FILE *f;
    
    h_jx_beam  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
    h_rho_beam = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
    
    if((f = fopen("beam3D.dat","wt")) == NULL)
    {
        puts("beam write failed");
	exit(0);
    }

    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          for(int k = 0;k < Nz;k++)
          {
              long nccc = mesh->GetN(i,j,k);
              Cell &ccc = p_CellArray[nccc];
              
              h_rho_beam[i*Ny*Nz+k*Ny+j] = ccc.GetRhoBeam();
              h_jx_beam[i*Ny*Nz +k*Ny+j] = ccc.GetJxBeam();
	      if(i == 191)
	      {
	         fprintf(f," %d %d %d %e %e \n",i,j,k,ccc.GetRhoBeam(),ccc.GetJxBeam());
	      }
          }
       }
   //    puts("=======================================================");
    }
    fclose(f);
    
    cudaMalloc((void **)d_rho_beam,sizeof(double)*Nx*Ny*Nz);
    cudaMalloc((void **)d_jx_beam, sizeof(double)*Nx*Ny*Nz);
    
    cudaMemcpy(*d_rho_beam,h_rho_beam,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
    cudaMemcpy(*d_jx_beam, h_jx_beam, sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
    
    free(h_jx_beam);
    free(h_rho_beam);
  
    return 0;
}

int CUDA_WRAP_PrintBoundaryValuesFromHost(int Ny,int Nz,Mesh *mesh,Cell *p_CellArray,int i_layer,char *where)
{
         //double *h_Rho,*h_Jx,*h_Jy,*h_Jz;
         int cell_number;
	 double frac_ideal,frac_rude;

	 //printf("%s :: BEGIN CHECK DENSITIES  ============================================================\n",where);
 
         int i = Ny - 1;
         
         for (int j = 0;j < Nz;j++)
	 {
	        cell_number = j*Ny + i;
		
	        //int num      = mesh->GetN(i_layer,i,j);
		long ncpc = mesh->GetN(i_layer,  i+1,  j);
	        Cell &cpc = p_CellArray[ncpc];
		
		long nzero = mesh->GetN(i_layer,  0,  j);
		Cell &zero =p_CellArray[nzero];
		printf("%20s cpc %15.15e zero %25.15e \n",where,cpc.GetEx(),zero.GetEx());
	 }
	 
	 
         return 0;
}


