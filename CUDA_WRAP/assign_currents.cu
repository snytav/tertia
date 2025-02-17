#include "cuCell.h"
#include "../run_control.h"
#include "cuBeamValues.h"

__device__ double cuda_atomicAdd(double *address, double val)
{
    double assumed,old=*address;
    do {
        assumed=old;
        old= __longlong_as_double(atomicCAS((unsigned long long int*)address,
                    __double_as_longlong(assumed),
                    __double_as_longlong(val+assumed)));
    }while (assumed!=old);

    return old;
}

__device__ int get3Dposition(int Ny,int Nz,int i,int k,int l)
{
    return (i*Ny*Nz+l*Ny + k); 
}

//---Mesh::MoveBeamInCell ---------------------------------------------->
__device__ void cuMoveBeamInCell(
                      double weight, int Ny,int Nz,int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho,double *JxBeam,double *JyBeam,double *JzBeam,double *RhoBeam,double *d_beam_values,int np)
{
   //double *RhoBeam,*JyBeam,*JzBeam;
   
/*   RhoBeam = bc->rho;
   JxBeam  = bc->jx;
   JyBeam  = bc->jy;
   JzBeam  = bc->jz;
  */ 
   int nccc = get3Dposition(Ny,Nz,i,j,k);
   int ncpc = get3Dposition(Ny,Nz,i,j+1,k);
   int nccp = get3Dposition(Ny,Nz,i,j,k+1);
   int ncpp = get3Dposition(Ny,Nz,i,j+1,k+1);
   int npcc = get3Dposition(Ny,Nz,i+1,j,k);
   int nppc = get3Dposition(Ny,Nz,i+1,j+1,k);
   int npcp = get3Dposition(Ny,Nz,i+1,j,k+1);
   int nppp = get3Dposition(Ny,Nz,i+1,j+1,k+1);
   
   if(np < 10)
   {
//       cuPrintf("in assign np %d i %d l %d k %d \n",np,i,j,k);
       //cuPrintf("i %d j %d k %d Ny %d Nz %d \n",i,j,k,Ny,Nz);
       //cuPrintf("nccc %d ncpc %d nccp %d ncpp %d npcc %d nppc %d npcp %d nppp %d \n",nccc,ncpc,nccp,ncpp,npcc,nppc,npcp,nppp);
       //cuPrintf("JxBeam %e \n",JxBeam[0]);
   }
   //return;
   double axp = x;
   double axc = 1. - axp;
   double ayp = y;
   double ayc = 1. - ayp;
   double azp = z;
   double azc = 1. - azp;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
   write_beam_value(np,BEAM_VALUES_NUMBER,132,d_beam_values,x);
   write_beam_value(np,BEAM_VALUES_NUMBER,133,d_beam_values,y);
   write_beam_value(np,BEAM_VALUES_NUMBER,134,d_beam_values,z);
   
   //return;
   
   write_beam_value(np,BEAM_VALUES_NUMBER,135,d_beam_values,axp);
   write_beam_value(np,BEAM_VALUES_NUMBER,136,d_beam_values,axc);
   
//   return;
   write_beam_value(np,BEAM_VALUES_NUMBER,137,d_beam_values,ayc);
   write_beam_value(np,BEAM_VALUES_NUMBER,138,d_beam_values,ayp);
   write_beam_value(np,BEAM_VALUES_NUMBER,139,d_beam_values,azc);
   write_beam_value(np,BEAM_VALUES_NUMBER,140,d_beam_values,azp);
    //return; 
   write_beam_value(np,BEAM_VALUES_NUMBER,141,d_beam_values,(double)i);
   write_beam_value(np,BEAM_VALUES_NUMBER,142,d_beam_values,(double)j);
   write_beam_value(np,BEAM_VALUES_NUMBER,143,d_beam_values,(double)k);
#endif   
  //return;
   
   //double weight = p->GetWeight();
  // if((i == 185) && (j == 48) && (k == 34)) //cuPrintf("djx %15.5e axc %15.5e ayc %15.5e azc %15.5e np %d \n",djx,axc,ayc,azc,np );
 
   cuda_atomicAdd(&(JxBeam[nccc]),djx*axc*ayc*azc);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,160,d_beam_values,djx*axc*ayc*azc);
#endif   
//   return;
   //atomicAdd(&JyBeam[nccc],djy*axc*ayc*azc);
   JzBeam[nccc]  +=  djz*axc*ayc*azc;
   cuda_atomicAdd(&(RhoBeam[nccc]),drho*axc*ayc*azc);
   //ccc.f_DensArray[isort] += weight*axc*ayc*azc;
//   if(np < 10) cuPrintf("ccc \n");
//   return;
   
   cuda_atomicAdd(&(JxBeam[nccp]),djx*axc*ayc*azp);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,161,d_beam_values,djx*axc*ayc*azp);
#endif   
   
   JyBeam[nccp]  +=  djy*axc*ayc*azp;
   JzBeam[nccp]  +=  djz*axc*ayc*azp;
   cuda_atomicAdd(&(RhoBeam[nccp]),drho*axc*ayc*azp);
   //ccp.f_DensArray[isort] += weight*axc*ayc*azp;

   cuda_atomicAdd(&(JxBeam[ncpc]),djx*axc*ayp*azc);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,162,d_beam_values,djx*axc*ayp*azc);
#endif   
   
   JyBeam[ncpc]  +=  djy*axc*ayp*azc;
   JzBeam[ncpc]  +=  djz*axc*ayp*azc;
   cuda_atomicAdd(&(RhoBeam[ncpc]),drho*axc*ayp*azc);
   //cpc.f_DensArray[isort] += weight*axc*ayp*azc;

   cuda_atomicAdd(&(JxBeam[ncpp]),djx*axc*ayp*azp);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,163,d_beam_values,djx*axc*ayp*azp);
#endif   
   
   JyBeam[ncpp]  +=  djy*axc*ayp*azp;
   JzBeam[ncpp]  +=  djz*axc*ayp*azp;
   cuda_atomicAdd(&(RhoBeam[ncpp]),drho*axc*ayp*azp);
   //cpp.f_DensArray[isort] += weight*axc*ayp*azp;

   //if(np < 10) cuPrintf("cpp \n");
//   return;

   cuda_atomicAdd(&(JxBeam[npcc]),djx*axp*ayc*azc);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,164,d_beam_values,djx*axp*ayc*azc);
#endif   
   
   JyBeam[npcc]  +=  djy*axp*ayc*azc;
   JzBeam[npcc]  +=  djz*axp*ayc*azc;
   cuda_atomicAdd(&(RhoBeam[npcc]),drho*axp*ayc*azc);
   //pcc.f_DensArray[isort] += weight*axp*ayc*azc;

   cuda_atomicAdd(&(JxBeam[npcp]),djx*axp*ayc*azp);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,165,d_beam_values,djx*axp*ayc*azp);
#endif   
   
   JyBeam[npcp]  +=  djy*axp*ayc*azp;
   JzBeam[npcp]  +=  djz*axp*ayc*azp;
   cuda_atomicAdd(&(RhoBeam[npcp]),drho*axp*ayc*azp);
   //pcp.f_DensArray[isort] += weight*axp*ayc*azp;

   cuda_atomicAdd(&(JxBeam[nppc]),djx*axp*ayp*azc);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,166,d_beam_values,djx*axp*ayp*azc);
#endif   
   
   JyBeam[nppc]  +=  djy*axp*ayp*azc;
   JzBeam[nppc]  +=  djz*axp*ayp*azc;
   cuda_atomicAdd(&(RhoBeam[nppc]),drho*axp*ayp*azc);
   //ppc.f_DensArray[isort] += weight*axp*ayp*azc;

   cuda_atomicAdd(&(JxBeam[nppp]),djx*axp*ayp*azp);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	   
   write_beam_value(np,BEAM_VALUES_NUMBER,167,d_beam_values,djx*axp*ayp*azp);
#endif   
   
   JyBeam[nppp]  +=  djy*axp*ayp*azp;
   JzBeam[nppp]  +=  djz*axp*ayp*azp;
   cuda_atomicAdd(&(RhoBeam[nppp]),drho*axp*ayp*azp); 
   //ppp.f_DensArray[isort] += weight*axp*ayp*azp;

//   p->p_Next = ccc.p_BeamHook;
//   ccc.p_BeamHook = p;
}
