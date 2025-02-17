#include "cuda_grid.h"

#include <math.h>
#include <stdio.h>
#include "cuda_wrap_control.h"

//#include "cuPrintf.cu"

//double d_fft_of_Rho[10];

__global__ void linearized_kernel(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,    //1
double *d_fft_of_RhoP,   // 2
double *d_fft_of_JxP,    // 3
double *d_fft_of_JyP,    // 4
double *d_fft_of_JzP,    // 5
double *d_fft_of_Ex,     // 6
double *d_fft_of_Ey,     // 7
double *d_fft_of_Ez,     // 8
double *d_fft_of_ExP,     // 9
double *d_fft_of_EyP,     // 10
double *d_fft_of_EzP,     // 11
double *d_fft_of_Jx,     // 12
double *d_fft_of_Jy,     // 13
double *d_fft_of_Jz,     // 14
double *d_fft_of_Bx,     // 15
double *d_fft_of_By,     // 16
double *d_fft_of_Bz,     // 17
double *d_fft_of_JxBeam, // 18
double *d_fft_of_RhoBeam,  // 19
double *d_fft_of_JxBeamP, // 18
double *d_fft_of_RhoBeamP  // 19

)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
	
    //    cuPrintf("nx %d ny %d \n",k,j);
	//cuPrintf("nx %d ny %d dens %e hx %e \n",k,j,dens,hx);
        
         double akz = M_PI/Zlength*(k+0.5);
         double aky = M_PI/Ylength*(j+0.5);
         double ak2 = aky*aky + akz*akz;
         double damp = 1.;

         unsigned int n1 = j + ny*k;

         //         VComplex rp  = (1.-maxRho)*vcRhoP[n1] + maxRho*diff_rp*damp;
         double rp  = d_fft_of_RhoP[n1] = d_fft_of_Rho[n1]; // + diff_rp*damp;
//         rp = dens;
         double jxp = d_fft_of_JxP[n1] = d_fft_of_Jx[n1]; // + diff_jx*damp;
         double jyp = d_fft_of_JyP[n1] = d_fft_of_Jy[n1]; // + diff_jy*damp;
         double jzp = d_fft_of_JzP[n1] = d_fft_of_Jz[n1]; // + diff_jz*damp;  
         double exp = d_fft_of_ExP[n1] = d_fft_of_Ex[n1]; // + diff_jx*damp;
         double eyp = d_fft_of_EyP[n1] = d_fft_of_Ey[n1]; // + diff_jy*damp;
         double ezp = d_fft_of_EzP[n1] = d_fft_of_Ez[n1]; // + diff_jz*damp;  


         double rb  = d_fft_of_RhoBeam[n1];
         double jxb = d_fft_of_JxBeam[n1];
         double rbp  = d_fft_of_RhoBeamP[n1];
         double jxbp = d_fft_of_JxBeamP[n1];
         double h = hx;

         double propagator = (4.-dens*hx*hx)/(4.+dens*hx*hx);
         double denominator = 4.+dens*hx*hx;

         d_fft_of_Rho[n1] = propagator*rp - (rb+rbp)*dens*hx*hx/denominator + 4.*exp*hx*(ak2+dens)/denominator;
         d_fft_of_Ex[n1] = propagator*exp - 2.*(2.*rp+rb+rbp)*dens*hx/((ak2+dens)*denominator);

//         fft_of_Ey[n1] = -(dens*eyp + eyp*ak2 - aky*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);
//         fft_of_Ez[n1] = -(dens*ezp + ezp*ak2 - akz*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);

         d_fft_of_Ey[n1] = -eyp + aky*(rb+rbp+rp+d_fft_of_Rho[n1])/(dens+ak2);
         d_fft_of_Ez[n1] = -ezp + akz*(rb+rbp+rp+d_fft_of_Rho[n1])/(dens+ak2);

         d_fft_of_Jy[n1] = jyp + hx*dens*(d_fft_of_Ey[n1] + eyp)/2.;
         d_fft_of_Jz[n1] = jzp + hx*dens*(d_fft_of_Ez[n1] + ezp)/2.;

         d_fft_of_Jx[n1] = jxp + hx*dens*(d_fft_of_Ex[n1] + exp)/2.;

         double newEy = 0.;
         double newEz = 0.;

         newEy = -eyp + (aky*(rb+rbp+d_fft_of_Rho[n1]+rp) + 2.*(jyp-d_fft_of_Jy[n1])/hx)/ak2;
         newEz = -ezp + (akz*(rb+rbp+d_fft_of_Rho[n1]+rp) + 2.*(jzp-d_fft_of_Jz[n1])/hx)/ak2;

         d_fft_of_Bx[n1] = -aky/ak2*d_fft_of_Jz[n1] + akz/ak2*d_fft_of_Jy[n1];
         d_fft_of_By[n1] = (-akz*(d_fft_of_Jx[n1] + d_fft_of_JxBeam[n1]) + dens*d_fft_of_Ez[n1])/ak2;
	 //printf("device %5d %e %e %e \n",n1,d_fft_of_Jx[n1],d_fft_of_JxBeam[n1],d_fft_of_Ez[n1]);
         d_fft_of_Bz[n1] =  (aky*(d_fft_of_Jx[n1] + d_fft_of_JxBeam[n1]) - dens*d_fft_of_Ey[n1])/ak2;
        
/*        
         if (fabs(fft_of_Ex[n1]) > maxfEx) maxfEx = fabs(fft_of_Ex[n1]);
         if (fabs(fft_of_Ey[n1]) > maxfEy) maxfEy = fabs(fft_of_Ey[n1]);
         if (fabs(fft_of_Ez[n1]) > maxfEz) maxfEz = fabs(fft_of_Ez[n1]);

         if (fabs(fft_of_Bx[n1]) > maxfBx) maxfBx = fabs(fft_of_Bx[n1]);
         if (fabs(fft_of_By[n1]) > maxfBy) maxfBy = fabs(fft_of_By[n1]);
         if (fabs(fft_of_Bz[n1]) > maxfBz) maxfBz = fabs(fft_of_Bz[n1]);
*/         
}

__global__ void linearizedIteratekernel(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,    //1
double *d_fft_of_RhoP,   // 2
double *d_fft_of_JxP,    // 3
double *d_fft_of_JyP,    // 4
double *d_fft_of_JzP,    // 5
double *d_fft_of_Ex,     // 6
double *d_fft_of_Ey,     // 7
double *d_fft_of_Ez,     // 8
double *d_fft_of_EyP,     // 7
double *d_fft_of_EzP,     // 8
double *d_fft_of_Jx,     // 9
double *d_fft_of_Jy,     // 10
double *d_fft_of_Jz,     // 11
double *d_fft_of_Bx,     // 12
double *d_fft_of_By,     // 13
double *d_fft_of_Bz,     // 14
double *d_fft_of_JxBeam, // 15
double *d_fft_of_RhoBeam,  // 16
double *d_fft_of_JxBeamP, // 15
double *d_fft_of_RhoBeamP  // 16
)
{
        unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
	
         double akz = M_PI/Zlength*(k+0.5);
         double aky = M_PI/Ylength*(j+0.5);
         double ak2 = aky*aky + akz*akz;
         double damp = 1.;

         long n1 = j + ny*k;

         double diff_jx = (d_fft_of_JxP[n1] - d_fft_of_Jx[n1])/hx;
         double diff_jy = (d_fft_of_JyP[n1] - d_fft_of_Jy[n1])/hx;
         double diff_jz = (d_fft_of_JzP[n1] - d_fft_of_Jz[n1])/hx;

        


/*
         if (fabs(fft_of_ExP[n1] + fft_of_Ex[n1]) > 1e-5) {
            dens = -2.*diff_jx/(fft_of_ExP[n1] + fft_of_Ex[n1]);
         } else {
            dens = dens;
         };
*/

         //         VComplex rp  = (1.-maxRho)*vcRhoP[n1] + maxRho*diff_rp*damp;
         double rp  = d_fft_of_RhoP[n1]; // 
//         rp = dens;
         double jx = d_fft_of_Jx[n1]; //
         double jy = d_fft_of_Jy[n1]; //
         double jz = d_fft_of_Jz[n1]; //  
         double jxp = d_fft_of_JxP[n1]; //
         double jyp = d_fft_of_JyP[n1]; //
         double jzp = d_fft_of_JzP[n1]; //  

         double rho  = rp + d_fft_of_Jx[n1] - d_fft_of_JxP[n1] 
            - hx*(aky*(jy+d_fft_of_JyP[n1]) + akz*(jz+d_fft_of_JzP[n1]))/2.;
/*
            rho  = fft_of_Rho[n1];

         jx = fft_of_Jx[n1] = jxp + fft_of_Rho[n1] - fft_of_RhoP[n1] 
            + hx*(aky*(jy+fft_of_JyP[n1]) + akz*(jz+fft_of_JzP[n1]))/2.;;
            */
         double diffRho = d_fft_of_Rho[n1] - rho;


         double rb  = d_fft_of_RhoBeam[n1];
         double rbp = d_fft_of_RhoBeamP[n1];
         double jxb  = d_fft_of_JxBeam[n1];
         double jxbp = d_fft_of_JxBeamP[n1];
         double eyp = d_fft_of_EyP[n1];
         double ezp = d_fft_of_EzP[n1];
         double ey = d_fft_of_Ey[n1];
         double ez = d_fft_of_Ez[n1];

         double newEx = -(aky*jy + akz*jz)/ak2;
         double newEy = aky*(rb+rho)/(dens+ak2);
         double newEz = akz*(rb+rho)/(dens+ak2);

         newEy = -eyp + aky*(rb+rbp+rp+d_fft_of_Rho[n1])/(dens+ak2);
         newEz = -ezp + akz*(rb+rbp+rp+d_fft_of_Rho[n1])/(dens+ak2);

//         double newEy = -eyp + (aky*(rb+rbp+rho+rp) + 2.*(jyp-jy)/hx)/ak2;
//         double newEz = -ezp + (akz*(rb+rbp+rho+rp) + 2.*(jzp-jz)/hx)/ak2;

         
         d_fft_of_Ey[n1] = newEy;
         d_fft_of_Ez[n1] = newEz;

         d_fft_of_Rho[n1] = rho;

//         d_fft_of_Ey[n1] = -(dens*eyp + eyp*ak2 - aky*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);
//         fft_of_Ez[n1] = -(dens*ezp + ezp*ak2 - akz*(rp+rb+rbp+fft_of_Rho[n1]))/(dens+ak2);

         //errorEx += (newEx - d_fft_of_Ex[n1])*(newEx - d_fft_of_Ex[n1]);
         d_fft_of_Ex[n1] = newEx;
//         d_fft_of_Ex[n1] = 0.5*(newEx + d_fft_of_Ex[n1]);

         d_fft_of_Bx[n1] = -aky/ak2*d_fft_of_Jz[n1] + akz/ak2*d_fft_of_Jy[n1];
         d_fft_of_By[n1] = (-akz*(d_fft_of_Jx[n1] + d_fft_of_JxBeam[n1]) + dens*newEz)/ak2;
         d_fft_of_Bz[n1] =  (aky*(d_fft_of_Jx[n1] + d_fft_of_JxBeam[n1]) - dens*newEy)/ak2;
}


int CUDA_WRAP_linearized_loop(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,
double *d_fft_of_RhoP,
double *d_fft_of_JxP,
double *d_fft_of_JyP,
double *d_fft_of_JzP,
double *d_fft_of_Ex,
double *d_fft_of_Ey,
double *d_fft_of_Ez,
double *d_fft_of_ExP,
double *d_fft_of_EyP,
double *d_fft_of_EzP,
double *d_fft_of_Jx,
double *d_fft_of_Jy,
double *d_fft_of_Jz,
double *d_fft_of_Bx,
double *d_fft_of_By,
double *d_fft_of_Bz,
double *d_fft_of_JxBeam,
double *d_fft_of_RhoBeam,
double *d_fft_of_JxBeamP,
double *d_fft_of_RhoBeamP
)
{
    dim3 dimBlock(16,16,1); 
    dim3 dimGrid(ny/16,nz/16); 
    
    getCudaGrid(ny,nz,&dimBlock,&dimGrid);
  //  printf("ny %d nz %d block %d %d %d grid %d %d %d \n",ny,nz,dimBlock.x,dimBlock.y,dimBlock.z,dimGrid.x,dimGrid.y,dimGrid.z);
  //  exit(0);
    
    //cudaPrintfInit();
    timeBegin(2);
    linearized_kernel<<<dimGrid, dimBlock>>>(ny,nz,hx,dens,Zlength,Ylength,
	d_fft_of_Rho,
        d_fft_of_RhoP,
        d_fft_of_JxP,
        d_fft_of_JyP,
        d_fft_of_JzP,
        d_fft_of_Ex,
        d_fft_of_Ey,
        d_fft_of_Ez,
        d_fft_of_ExP,
        d_fft_of_EyP,
        d_fft_of_EzP,
        d_fft_of_Jx,
        d_fft_of_Jy,
        d_fft_of_Jz,
        d_fft_of_Bx,
        d_fft_of_By,
        d_fft_of_Bz,
        d_fft_of_JxBeam,
        d_fft_of_RhoBeam,
        d_fft_of_JxBeamP,
        d_fft_of_RhoBeamP
    );
    timeEnd(2);
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();    
    //exit(0);
    
    int err = cudaGetLastError();
    
    return 0;
}

int CUDA_WRAP_linearizedIterateloop(int ny,int nz,double hx,double dens,double Zlength,double Ylength,
double *d_fft_of_Rho,
double *d_fft_of_RhoP,
double *d_fft_of_JxP,
double *d_fft_of_JyP,
double *d_fft_of_JzP,
double *d_fft_of_Ex,
double *d_fft_of_Ey,
double *d_fft_of_Ez,
double *d_fft_of_EyP,
double *d_fft_of_EzP,
double *d_fft_of_Jx,
double *d_fft_of_Jy,
double *d_fft_of_Jz,
double *d_fft_of_Bx,
double *d_fft_of_By,
double *d_fft_of_Bz,
double *d_fft_of_JxBeam,
double *d_fft_of_RhoBeam,
double *d_fft_of_JxBeamP,
double *d_fft_of_RhoBeamP
)
{
    dim3 dimBlock; 
    dim3 dimGrid; 
    
    getCudaGrid(ny,nz,&dimBlock,&dimGrid);
  //  printf("ny %d nz %d block %d %d %d grid %d %d %d \n",ny,nz,dimBlock.x,dimBlock.y,dimBlock.z,dimGrid.x,dimGrid.y,dimGrid.z);
  //  exit(0);
    
//    cudaPrintfInit();
    timeBegin(2);
    linearizedIteratekernel<<<dimGrid, dimBlock>>>(ny,nz,hx,dens,Zlength,Ylength,
	d_fft_of_Rho,
        d_fft_of_RhoP,
        d_fft_of_JxP,
        d_fft_of_JyP,
        d_fft_of_JzP,
        d_fft_of_Ex,
        d_fft_of_Ey,
        d_fft_of_Ez,
        d_fft_of_EyP,
        d_fft_of_EzP,	
        d_fft_of_Jx,
        d_fft_of_Jy,
        d_fft_of_Jz,
        d_fft_of_Bx,
        d_fft_of_By,
        d_fft_of_Bz,
        d_fft_of_JxBeam,
        d_fft_of_RhoBeam,
        d_fft_of_JxBeamP,
        d_fft_of_RhoBeamP
    );
    timeEnd(2);
//    cudaPrintfDisplay(stdout, true);
  //  cudaPrintfEnd();    
    //exit(0);
    
    return 0;
}

