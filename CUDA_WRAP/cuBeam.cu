// #include "cuPrintf.cu"
#include "../particles.h"
#include "../cells.h"
#include "../mesh.h"
#include "cuCell.h"
#include "beam_copy.h"
#include "assign_currents.cu"
#include "cuBeamValues.h"
#include "../run_control.h"
#include "split_layer.h"
#include "cuLayers.h"
#include <math.h>

#include "diagnostic_print.h"


#include "../para.h"

#define NP 10
#define NBA 10

beamParticle *beam_particles;

double *h_beam_values,*d_beam_values;
static int first_h_beam_values = 1;

double *d_RhoBeam3D,*d_JxBeam3D;

int CUDA_MALLOC_TEST(char *wh)
{
	double *x;
	printf(" CUDA_MALLOC_TEST at %s  \n",wh);
	cudaMalloc(&x,8);
	printf(" CUDA_MALLOC_TEST at %s  OK \n",wh);
}

__device__ int get4Dposition(int Ny,int Nz,int Np,int i,int k,int l,int n)
{
    return (i*Ny*Nz*Np+k*Nz*Np + l*Np + n); 
}

int get3DpositionHost(int Ny,int Nz,int i,int k,int l)
{
    return (i*Ny*Nz+k*Nz + l); 
}

int get4DpositionHost(int Ny,int Nz,int Np,int i,int k,int l,int n)
{
    return (i*Ny*Nz*Np+k*Nz*Np + l*Np + n); 
}


// TODO: COPY ADDITIONAL LAYER FROM THE RIGHT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        

double *beam_surface,*field3D;
double*      *beam_array,*field3Darray;

int CUDA_WRAP_create_Beam_particle_surface(double*surf,double *surf_array,int width,int height,double *h_data_in)
{
//        int width = Nx;
  //      int height = (Ny+1)*(Nz+1)*6*sizeof(double);
        int size = width*height*sizeof(double);

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned); 
        
        int err = cudaMalloc(&surf_array, size);

        err = cudaMemcpy(surf_array,h_data_in, size, cudaMemcpyHostToDevice);
        
        // Bind the arrays to the surface references 
//         err = cudaBindSurfaceToArray( surf, surf_array);

        return 0;
}
//       
int CUDA_WRAP_write_beam_value(int i,int num_attr,int n,double *h_p,double t)
{
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	

	//int cell_number = i*Ny + j;
	
	h_p [i*num_attr + n] = t;
	
	//cudaMemcpy((void**)d_p,num_attr*ppc_max*Ny*Nz*sizeof(double));
#endif	
	
	return 0;
}

//writing a value to the control array for a definite particle in a definite cell
__device__ int write_beam_value(int i,int num_attr,int n,double *d_p,double t)
{
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	

	//int cell_number = i*Ny + j;
	
	d_p [i*num_attr +  n] = t;
#endif	
	return 0;
}


__device__ void getFieldCell(int Ny,int Nz,int i,int k,int l,cudaCell *cc,double *Ex,double *Ey,double *Ez,double *Bx,double *By,double *Bz)
{
        int n = get3Dposition(Ny,Nz,i,k,l);
	
	cc->f_Ex = Ex[n];
	cc->f_Ey = Ey[n];
	cc->f_Ez = Ez[n];
	cc->f_Bx = Bx[n];
	cc->f_By = By[n];
	cc->f_Bz = Bz[n];
}


__global__ void moveBeamKernel(beamParticle *beam_particles,double *d_beam_values,int Np,int l_Mx,int l_My,int l_Mz,double hx,double hy,double hz,double ts,
			       double *Ex,double *Ey,double *Ez,double *Bx,double *By,double *Bz,double *jx3D,double *jy3D,double *jz3D,double *rho3D)
{
        unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int sizeY = gridDim.y*blockDim.y;
	unsigned int nz = 0;//threadIdx.z;
	unsigned int np;
	int i1,k1,l1;
        double t,t2,t3,t4,t5,t6;
	//double x,y,z,px,py,pz;
	cudaCell ccc,cpc,ccp,cpp,pcc,ppc,pcp,ppp;
	beamParticle *p,curp;
	
	np = sizeY*nx + ny;
	
	if(np >= Np) return;
	
//	cuPrintf("in beamKernel\n");
	////cuPrintf("grimDim.y %d blockDim.y %d sizeY %d ny %d nx %d np %d \n",gridDim.y,blockDim.y,sizeY,nx,ny,np);
        //return;
        
//	curp = *beam_particles;
	p = beam_particles + np;
	//if(np < 10) cuPrintf("at particle np %d %d \n",np,(int)p);
	//return;
	i1 = p->i_X;
	l1 = p->i_Y;
	k1 = p->i_Z;
//	if(i1 < 0) return; // a particle was deleted because of leaving subdomain

// 	if(np < 10) cuPrintf("ilk %d %d %d \n",i1,l1,k1);
	
	//return;

	
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,0,d_beam_values,(double)i1);
	write_beam_value(np,BEAM_VALUES_NUMBER,1,d_beam_values,(double)l1);
	write_beam_value(np,BEAM_VALUES_NUMBER,2,d_beam_values,(double)k1);
#endif
	
	
//	return;
        getFieldCell(l_My,l_Mz,i1,  l1,  k1,  &ccc,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	
	write_beam_value(np,BEAM_VALUES_NUMBER,3,d_beam_values,ccc.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,4,d_beam_values,ccc.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,5,d_beam_values,ccc.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,6,d_beam_values,ccc.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,7,d_beam_values,ccc.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,8,d_beam_values,ccc.f_Bz);
#endif	
/////////////////////	
        getFieldCell(l_My,l_Mz,i1,  l1+1,k1,  &cpc,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,118,d_beam_values,cpc.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,119,d_beam_values,cpc.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,120,d_beam_values,cpc.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,121,d_beam_values,cpc.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,122,d_beam_values,cpc.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,123,d_beam_values,cpc.f_Bz);
#endif
	/////////////////////	
	getFieldCell(l_My,l_Mz,i1,  l1,  k1+1,&ccp,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,124,d_beam_values,ccp.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,125,d_beam_values,ccp.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,126,d_beam_values,ccp.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,127,d_beam_values,ccp.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,128,d_beam_values,ccp.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,129,d_beam_values,ccp.f_Bz);
#endif
/////////////////////	
        getFieldCell(l_My,l_Mz,i1,  l1+1,k1+1,&cpp,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,9,d_beam_values,cpp.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,10,d_beam_values,cpp.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,11,d_beam_values,cpp.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,12,d_beam_values,cpp.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,13,d_beam_values,cpp.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,14,d_beam_values,cpp.f_Bz);
#endif	
/////////////////////	
        getFieldCell(l_My,l_Mz,i1+1,l1,  k1,  &pcc,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,15,d_beam_values,pcc.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,16,d_beam_values,pcc.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,17,d_beam_values,pcc.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,18,d_beam_values,pcc.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,19,d_beam_values,pcc.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,20,d_beam_values,pcc.f_Bz);
#endif	
/////////////////////	
        getFieldCell(l_My,l_Mz,i1+1,l1+1,k1,  &ppc,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	

	write_beam_value(np,BEAM_VALUES_NUMBER,21,d_beam_values,ppc.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,22,d_beam_values,ppc.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,23,d_beam_values,ppc.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,24,d_beam_values,ppc.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,25,d_beam_values,ppc.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,26,d_beam_values,ppc.f_Bz);
#endif	
/////////////////////	
        getFieldCell(l_My,l_Mz,i1+1,l1,  k1+1,&pcp,Ex,Ey,Ez,Bx,By,Bz);
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	
	write_beam_value(np,BEAM_VALUES_NUMBER,27,d_beam_values,pcp.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,28,d_beam_values,pcp.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,29,d_beam_values,pcp.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,30,d_beam_values,pcp.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,31,d_beam_values,pcp.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,32,d_beam_values,pcp.f_Bz);
#endif
/////////////////////	
        getFieldCell(l_My,l_Mz,i1+1,l1+1,k1+1,&ppp,Ex,Ey,Ez,Bx,By,Bz);
//        if(np < 10) //cuPrintf("getField ppp \n");
        //return;
/////////////////////
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	write_beam_value(np,BEAM_VALUES_NUMBER,33,d_beam_values,ppp.f_Ex);
	write_beam_value(np,BEAM_VALUES_NUMBER,34,d_beam_values,ppp.f_Ey);
	write_beam_value(np,BEAM_VALUES_NUMBER,35,d_beam_values,ppp.f_Ez);
	write_beam_value(np,BEAM_VALUES_NUMBER,36,d_beam_values,ppp.f_Bx);
	write_beam_value(np,BEAM_VALUES_NUMBER,37,d_beam_values,ppp.f_By);
	write_beam_value(np,BEAM_VALUES_NUMBER,38,d_beam_values,ppp.f_Bz);
#endif
/////////////////////	
	
	
      //  //cuPrintf("field3D nx %d ny %d %e %e %e %e %e %e \n",nx,ny,ccc.f_Ex,ccc.f_Ey,ccc.f_Ez,ccc.f_Bx,ccc.f_By,ccc.f_Bz);
	//return;
               double weight = p->f_Weight;
               double x  = p->f_X;
               double y  = p->f_Y;
               double z  = p->f_Z;
               double px = p->f_Px;
               double py = p->f_Py;
               double pz = p->f_Pz;
               double pxn = px;
               double pyn = py;
               double pzn = pz;
               double Vx = px / sqrt(1. + px*px + py*py + pz*pz);
               double q2m = p->f_Q2m;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
        write_beam_value(np,BEAM_VALUES_NUMBER,39,d_beam_values,weight);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,40,d_beam_values,x);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,41,d_beam_values,y);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,42,d_beam_values,z);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,43,d_beam_values,px);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,44,d_beam_values,py);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,45,d_beam_values,pz);	       
#endif	
        // return;
               double axc = 1.-x;
               double axp = x;
               double ayc = 1.-y;
               double ayp = y;
               double azc = 1.-z;
               double azp = z;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
        write_beam_value(np,BEAM_VALUES_NUMBER,130,d_beam_values,axc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,46,d_beam_values,axp);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,47,d_beam_values,ayc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,48,d_beam_values,ayp);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,49,d_beam_values,azc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,50,d_beam_values,azp);	       
#endif	       

               double accc = axc*ayc*azc;
               double acpc = axc*ayp*azc;
               double accp = axc*ayc*azp;
               double acpp = axc*ayp*azp;
               double apcc = axp*ayc*azc;
               double appc = axp*ayp*azc;
               double apcp = axp*ayc*azp;
               double appp = axp*ayp*azp;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
        write_beam_value(np,BEAM_VALUES_NUMBER,51,d_beam_values,accc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,52,d_beam_values,acpc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,53,d_beam_values,accp);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,54,d_beam_values,acpp);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,55,d_beam_values,apcc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,56,d_beam_values,appc);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,57,d_beam_values,apcp);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,58,d_beam_values,appp);	       
	
           write_beam_value(np,BEAM_VALUES_NUMBER,152,d_beam_values,px);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,153,d_beam_values,py);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,154,d_beam_values,pz);		
#endif	   
       // return;  
	       double ex, ey, ez;

               double bx=0.;
               double by=0.;
               double bz=0.;
             //  //cuPrintf("before fields\n");  
               ex =
                  accc*ccc.f_Ex + acpc*cpc.f_Ex +
                  accp*ccp.f_Ex + acpp*cpp.f_Ex +
                  apcc*pcc.f_Ex + appc*ppc.f_Ex +
                  apcp*pcp.f_Ex + appp*ppp.f_Ex;

               ey =
                  accc*ccc.f_Ey + acpc*cpc.f_Ey +
                  accp*ccp.f_Ey + acpp*cpp.f_Ey +
                  apcc*pcc.f_Ey + appc*ppc.f_Ey +
                  apcp*pcp.f_Ey + appp*ppp.f_Ey;

               ez =
                  accc*ccc.f_Ez + acpc*cpc.f_Ez +
                  accp*ccp.f_Ez + acpp*cpp.f_Ez +
                  apcc*pcc.f_Ez + appc*ppc.f_Ez +
                  apcp*pcp.f_Ez + appp*ppp.f_Ez;


               bx =
                  accc*ccc.f_Bx + acpc*cpc.f_Bx +
                  accp*ccp.f_Bx + acpp*cpp.f_Bx +
                  apcc*pcc.f_Bx + appc*ppc.f_Bx +
                  apcp*pcp.f_Bx + appp*ppp.f_Bx;

               by =
                  accc*ccc.f_By + acpc*cpc.f_By +
                  accp*ccp.f_By + acpp*cpp.f_By +
                  apcc*pcc.f_By + appc*ppc.f_By +
                  apcp*pcp.f_By + appp*ppp.f_By;

               bz =
                  accc*ccc.f_Bz + acpc*cpc.f_Bz +
                  accp*ccp.f_Bz + acpp*cpp.f_Bz +
                  apcc*pcc.f_Bz + appc*ppc.f_Bz +
                  apcp*pcp.f_Bz + appp*ppp.f_Bz;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
        write_beam_value(np,BEAM_VALUES_NUMBER,59,d_beam_values,ex);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,61,d_beam_values,ey);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,62,d_beam_values,ez);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,63,d_beam_values,bx);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,64,d_beam_values,by);	       
        write_beam_value(np,BEAM_VALUES_NUMBER,65,d_beam_values,bz);	       
#endif		  

//               ex = ey = ez = bx = by = bz = 0.;

               double ex1 = ex;
               double ey1 = ey;
               double ez1 = ez;
               //if(np < 10) //cuPrintf("after fields\n");  
	     //  return;


        /*       if(isort > 0 && iAtomTypeArray[isort] > 0) {//if and only if ionizable ions
                  int iZ = p->GetZ();

                  if (iZ < iAtomTypeArray[isort]) {
                     double field = sqrt(ex*ex + ey*ey + ez*ez);
                     p->Ionize(&ccc, field);
                     p_next = p->p_Next;
                     //	      if (iZ == 0) continue;
                  };
                  q2m *= iZ;
                  weight *= iZ;
               }

               ex *= q2m*ts/2.;
               ey *= q2m*ts/2.;
               ez *= q2m*ts/2.;*/
               /*
               double bx = axc*ccc.f_Bx + axm*mcc.f_Bx;
               double by= ayc*ccc.f_By + aym*cmc.f_By;
               double bz= azc*ccc.f_Bz + azm*ccm.f_Bz;
               */
               //		  double Bext = 1e-2*(PI*domain()->GetTs());
               //					by += Bext;

              // bx += bXext;
              // by += bYext;
              // bz += bZext;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,149,d_beam_values,px);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,150,d_beam_values,py);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,151,d_beam_values,pz);
	   write_beam_value(np,BEAM_VALUES_NUMBER,155,d_beam_values,ex);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,156,d_beam_values,ey);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,157,d_beam_values,ez);
#endif	   
               px += ex;
               py += ey;
               pz += ez;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	       
           write_beam_value(np,BEAM_VALUES_NUMBER,66,d_beam_values,px);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,67,d_beam_values,py);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,68,d_beam_values,pz);	       
#endif	   

	       double f_GammaMax;
               double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,69,d_beam_values,gamma);	       
#endif	   
	       

               if (f_GammaMax < gamma)
                  f_GammaMax = gamma;

               double gamma_r = 1./gamma;																	 //!!!!!!
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,70,d_beam_values,gamma_r);	       
#endif               
               ///////////////////////////////////////////////////////////////////////
               /////////                SCATTERING               /////////////////////
               //////////////////////////////////////////////////////////////////////
               int isort = 0;
	       int ifscatter = 0;
	       int nsorts = 1;
/*	       
               if (isort == 0 && ifscatter) //We scatter only electrons
               {
                  double P = sqrt(px*px+py*py+pz*pz);
                  double IonDensityInCell = 0.;
                  for (int is=1; is<nsorts; is++) {
                     IonDensityInCell += ccc.f_DensArray[is];
                  };

                  double Probability = p->GetScatteringProbability(P, IonDensityInCell);

                  if (Probability > 0.)
                  {
                     double Nx = 2*(double(rand())/RAND_MAX-0.5);
                     double Ny = 2*(double(rand())/RAND_MAX-0.5);
                     double Nz = 2*(double(rand())/RAND_MAX-0.5);

                     double N = sqrt(Nx*Nx + Ny*Ny + Nz*Nz);

                     //arbitary unitvector f =|Probability|
                     double fx = (Probability*Nx)/N;
                     double fy = (Probability*Ny)/N;
                     double fz = (Probability*Nz)/N;

                     float f = sqrt(fx*fx + fy*fy + fz*fz); //TO TEST.

                     cout<<"collision::"<<"prob = "<<Probability<<" f = "<<f<<endl; //TO TEST
                     
                     bx += fx;
                     by += fy;
                     bz += fz;
                  }
               }
*/
               ////////////////////////////////////////////////////////////////////////
               //////////////////////////////////////////////////////////////////////// 
               //double ts;
	       
               double bx1 = bx;
               double by1 = by;
               double bz1 = bz;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,71,d_beam_values,bx1);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,72,d_beam_values,by1);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,73,d_beam_values,bz1);	       
#endif	       

               bx = bx*gamma_r*q2m*ts/2.;
               by = by*gamma_r*q2m*ts/2.;
               bz = bz*gamma_r*q2m*ts/2.;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,74,d_beam_values,bx);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,75,d_beam_values,by);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,76,d_beam_values,bz);	       
#endif	       

               double co = 2./(1. + (bx*bx) + (by*by) + (bz*bz));
           write_beam_value(np,BEAM_VALUES_NUMBER,77,d_beam_values,co);	       
	       

               double p3x = py*bz - pz*by + px;
               double p3y = pz*bx - px*bz + py;
               double p3z = px*by - py*bx + pz;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,78,d_beam_values,p3x);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,79,d_beam_values,p3y);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,80,d_beam_values,p3z);	       
#endif	       

               p3x *= co;
               p3y *= co;
               p3z *= co;

               double px_new = p3y*bz - p3z*by;
               double py_new = p3z*bx - p3x*bz;
               double pz_new = p3x*by - p3y*bx;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,81,d_beam_values,px_new);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,82,d_beam_values,py_new);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,83,d_beam_values,pz_new);	       
#endif	       

               px += (ex + px_new);
               py += (ey + py_new);
               pz += (ez + pz_new);
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,84,d_beam_values,px);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,85,d_beam_values,py);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,86,d_beam_values,pz);	       
#endif	       
               /*
               double damping = 1e-3;

               px -= damping*ts*px*gamma;
               py -= damping*ts*py*gamma;
               pz -= damping*ts*pz*gamma;
               */

               /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
               /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
               double djx,djy,djz;
	       
               gamma = 1./sqrt(1. + px*px + py*py + pz*pz);
               Vx = px*gamma;
               double Vy = py*gamma;
               double Vz = pz*gamma;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,87,d_beam_values,gamma);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,88,d_beam_values,Vx);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,89,d_beam_values,Vy);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,90,d_beam_values,Vz);	       
#endif	       

//               p->SetP(px,py,pz);
//            if(np < 10) cuPrintf("X,P %e %e %e %e %e %e \n",p->f_X,p->f_Y,p->f_Z,px,py,pz);
	   p->f_Px = px;
	   p->f_Py = py;
	   p->f_Pz = pz;
	   
	   //if(np < 10) //cuPrintf("SetP \n");
//	   return;

               double polarity = 1;//p->GetSpecie()->GetPolarity();

               djx = weight*polarity*Vx;
               djy = weight*polarity*Vy;
               djz = weight*polarity*Vz;
               double drho = weight*polarity;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,91,d_beam_values,djx);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,92,d_beam_values,djy);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,93,d_beam_values,djz);	       
           write_beam_value(np,BEAM_VALUES_NUMBER,144,d_beam_values,drho);	       
	       
           write_beam_value(np,BEAM_VALUES_NUMBER,146,d_beam_values,gamma);
           write_beam_value(np,BEAM_VALUES_NUMBER,145,d_beam_values,px);
	   write_beam_value(np,BEAM_VALUES_NUMBER,147,d_beam_values,hx);
	   write_beam_value(np,BEAM_VALUES_NUMBER,131,d_beam_values,ts);
	   write_beam_value(np,BEAM_VALUES_NUMBER,148,d_beam_values,((px*gamma-1.)*ts/hx));
#endif	   
               double dx = (px*gamma-1.)*ts/hx;
               double dy = py*gamma*ts/hy;
               double dz = pz*gamma*ts/hz;
               double xtmp = x + dx;
               double ytmp = y + dy;
               double ztmp = z + dz;
	   
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,94,d_beam_values,dx);
           write_beam_value(np,BEAM_VALUES_NUMBER,95,d_beam_values,dy);
           write_beam_value(np,BEAM_VALUES_NUMBER,96,d_beam_values,dz);
           write_beam_value(np,BEAM_VALUES_NUMBER,97,d_beam_values,xtmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,98,d_beam_values,ytmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,99,d_beam_values,ztmp);
#endif	   
               double full = 1.;

               int i_jump = xtmp;
               int j_jump = ytmp;
               int k_jump = ztmp;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,100,d_beam_values,(double)i_jump);
           write_beam_value(np,BEAM_VALUES_NUMBER,101,d_beam_values,(double)j_jump);
           write_beam_value(np,BEAM_VALUES_NUMBER,102,d_beam_values,(double)k_jump);
#endif	       
               if (xtmp < 0.) {
                  i_jump--;
               };
               if (ytmp < 0.) {
                  j_jump--;
               };
               if (ztmp < 0.) {
                  k_jump--;
               };
	       
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,103,d_beam_values,(double)i_jump);
           write_beam_value(np,BEAM_VALUES_NUMBER,104,d_beam_values,(double)j_jump);
           write_beam_value(np,BEAM_VALUES_NUMBER,105,d_beam_values,(double)k_jump);
#endif	       
	       int i = nx;
	       int j = ny;
	       int k = 0;
	       
               int itmp = i1 + i_jump;
               int jtmp = l1 + j_jump;
               int ktmp = k1 + k_jump;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,106,d_beam_values,(double)itmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,107,d_beam_values,(double)jtmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,108,d_beam_values,(double)ktmp);
#endif	       
               if (itmp < 0) itmp = -1;
               if (itmp > l_Mx-1) itmp = l_Mx;
               if (jtmp < 0) jtmp = -1;
               if (jtmp > l_My-1) jtmp = l_My;
               if (ktmp < 0) ktmp = -1;
               if (ktmp > l_Mz-1) ktmp = l_Mz;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
           write_beam_value(np,BEAM_VALUES_NUMBER,109,d_beam_values,(double)itmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,110,d_beam_values,(double)jtmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,111,d_beam_values,(double)ktmp);
#endif
               xtmp -= i_jump;
               ytmp -= j_jump;
               ztmp -= k_jump;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	   write_beam_value(np,BEAM_VALUES_NUMBER,112,d_beam_values,xtmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,113,d_beam_values,ytmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,114,d_beam_values,ztmp);
#endif
               while (xtmp < 0.) {
		 xtmp += 1.;
	       };
               while (xtmp > 1.) {
		 xtmp -= 1.;
	       };
               while (ytmp < 0.) {
		 ytmp += 1.;
	       };
               while (ytmp > 1.) {
		 ytmp -= 1.;
	       };
               while (ztmp < 0.) {
		 ztmp += 1.;
	       };
               while (ztmp > 1.) {
		 ztmp -= 1.;
	       };
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 	
	   write_beam_value(np,BEAM_VALUES_NUMBER,115,d_beam_values,xtmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,116,d_beam_values,ytmp);
           write_beam_value(np,BEAM_VALUES_NUMBER,117,d_beam_values,ztmp);
#endif
            //   p->SetX(xtmp,ytmp,ztmp);
	   p->f_X = xtmp;
	   p->f_Y = ytmp;
	   p->f_Z = ztmp;
	   
	   p->i_X = itmp;
	   p->i_Y = jtmp;
	   p->i_Z = ktmp;
/*
               int ntmp = GetN(itmp,jtmp,ktmp);
               Cell &ctmp = p_CellArray[ntmp];
               p->l_Cell = ntmp;

*/
           __syncthreads();

	   //if(np < 10) //cuPrintf("befoe BeamInCell \n");
//	   return;
	   
           cuMoveBeamInCell(
                      weight, l_My,l_Mz,isort,
                      itmp, jtmp, ktmp,Vx,Vy,Vz, xtmp, ytmp,ztmp, 
                      djx, djy, djz, drho,jx3D,jy3D,jz3D,rho3D,d_beam_values,np);
	   
	   __syncthreads();

}

int makeFieldArray(int Nx,int Ny,int Nz,int *w,int *h,double **h_data_in,double *h_ex,double *h_ey,double *h_ez,double *h_hx,double *h_hy,double *h_hz)
{
// TODO: COPY ADDITIONAL LAYER FROM THE RIGHT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          
        int width = Nx;
        int height = (Ny+1)*(Nz+1)*sizeof(double)*6;
        int size = width*height*sizeof(double);
        
        *h_data_in = (double *)malloc(size);
        
        for(int i = 0; i < width;i++)
        {
           for(int l = 0; l < Ny;l++)
	   {
	      for(int k = 0;k < Nz;k++)
	      {
		 int j = l*Nz+k; 
//                 (*h_data_in)[i*height + j] = (double)(10*i+j);
                 (*h_data_in)[i*height + j*6+0] = (double)(100*i+10*j);//h_ex[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+1] = (double)(100*i+10*j+1);//h_ey[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+2] = (double)(100*i+10*j+2);//h_ez[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+3] = (double)(100*i+10*j+3);//h_hx[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+4] = (double)(100*i+10*j+4);//h_hy[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+5] = (double)(100*i+10*j+5);//h_hz[i*(Ny*Nz)+j]; 
/*		 
                 (*h_data_in)[i*height + j*6+0] = h_ex[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+1] = h_ey[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+2] = h_ez[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+3] = h_hx[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+4] = h_hy[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+5] = h_hz[i*(Ny*Nz)+j];    
		 */
	      }
                 
           }
        }
// TODO: COPY ADDITIONAL LAYER FROM THE RIGHT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        
        *w = width;
        *h = height;
                  
        return 0;
}



int CUDA_WRAP_beam_move(int Np,int Nx,int Ny,int Nz,double hx,double hy,double hz,double ts,int nstep)
{
    beamCurrents bc;
    
#ifdef CUDA_WRAP_FFTW_ALLOWED
    return 1;
#endif    
  
    dim3 dimBlock(16,16 ,1); 
    
    int gridSize = (int)(ceil(sqrt(Np))/16.0+1.0);
    
   // printf("rank %d in beam move Np %d Nx %d hx %e gridSize %d \n",GetRank(),Np,Nx,hx,gridSize);
    
    
    dim3 dimGrid(gridSize, gridSize); 
        
    struct timeval tv2,tv1;
    
    bc.jx = d_JxBeam3D;
    bc.jy = d_JyBeam3D;
    bc.jz = d_JzBeam3D;
    bc.rho = d_RhoBeam3D;
    
    //cudaPrintfInit(); 
    printf("before beam3D alloc \n");
    exit(0);
    cudaMalloc(&d_JxBeam3D,Nx*Ny*Nz*sizeof(double));
    cudaMalloc(&d_RhoBeam3D,Nx*Ny*Nz*sizeof(double));

    cudaMemset(d_JxBeam3D, 0,Nx*Ny*Nz*sizeof(double));
    cudaMemset(d_RhoBeam3D,0,Nx*Ny*Nz*sizeof(double));
    
  //  printf("rank %d in beam move before kernel Np %d Nx %d hx %e gridSize %d \n",GetRank(),Np,Nx,hx,gridSize);
    
    CUDA_WRAP_printParticles(beam_particles,"before kernel particles ");
    printf("before beam kernel block 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
    moveBeamKernel<<<dimGrid, dimBlock>>>(beam_particles,d_beam_values,Np,Nx,Ny,Nz,hx,hy,hz,ts,d_Ex3D,d_Ey3D,d_Ez3D,d_Bx3D,d_By3D,d_Bz3D,
					  d_JxBeam3D,d_JyBeam3D,d_JzBeam3D,d_RhoBeam3D);
    
    cudaDeviceSynchronize();
    printf("after  beam kernel block 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

    int err = cudaGetLastError();
    
    if(err != cudaSuccess)
    {
       printf("after beam kernel block 1 error %d \n",err);
       exit(0);
    }
    //cudaPrintfDisplay(stdout, true);
    //cudaPrintfEnd();
    puts("qq");
    
    CUDA_DEBUG_print3DmatrixLayer(d_JxBeam3D,Nx*0.8, Ny,Nz,"Jx0.8");
    CUDA_DEBUG_print3DmatrixLayer(d_JxBeam3D,Nx*0.5, Ny,Nz,"Jx0.5");
    
    //exit(0);
    
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED
    if(first_h_beam_values)
    {
       first_h_beam_values = 0;
       h_beam_values = (double*)malloc(Np*BEAM_VALUES_NUMBER*sizeof(double));
    }
    CUDA_WRAP_check_beam_values(Np,BEAM_VALUES_NUMBER,h_beam_values,d_beam_values,dimBlock.x*dimGrid.x,dimBlock.y*dimGrid.y,
                                "beamValues.dat","BEAM",nstep);
#endif
}



int CUDA_WRAP_copy3Dfields(Mesh *mesh,Cell *p_CellArray,int Nx,int Ny,int Nz)
{
   int height = Ny*Nz;
   double *ex,*ey,*ez,*hx,*hy,*hz;
   
   ex = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   ey = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   ez = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   hx = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   hy = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   hz = (double*)malloc(sizeof(double)*Nx*Ny*Nz);
   
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
         for (int i=0; i<Nx; i++)
         {
                long nccc = mesh->GetN(i, j,k);
	        Cell &ccc = p_CellArray[nccc];	
		int n = get3DpositionHost(Ny,Nz,i,j,k);
		ex[n] = ccc.GetEx();
		ey[n] = ccc.GetEy();
		ez[n] = ccc.GetEz();
		hx[n] = ccc.GetBx();
		hy[n] = ccc.GetBy();
		hz[n] = ccc.GetBz();
	 }
      }
   }
   cudaMemcpy(d_Ex3D,ex,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(d_Ey3D,ey,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(d_Ez3D,ez,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(d_Bx3D,hx,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(d_By3D,hy,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(d_Bz3D,hy,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
   
   return 0;
}

int makeParticleArray(int Nx,int Ny,int Nz,int *w,int *h,double **h_data_in,double *x,double *y,double *z,double *px,double *py,double *pz)
{
// TODO: COPY ADDITIONAL LAYER FROM THE RIGHT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!          
        int width = Nx;
        int height = (Ny)*(Nz)*sizeof(double)*NBA;
        int size = width*height*sizeof(double);
        
        *h_data_in = (double *)malloc(size);
        
        for(int i = 0; i < width;i++)
        {
           for(int l = 0; l < Ny;l++)
	   {
	      for(int k = 0;k < Nz;k++)
	      {
		 int j = l*Nz+k; 
//                 (*h_data_in)[i*height + j] = (double)(10*i+j);
                 (*h_data_in)[i*height + j*NBA+0] = x[i*(Ny*Nz)+l*Nz+k];    
                 (*h_data_in)[i*height + j*NBA+1] = y[i*(Ny*Nz)+l*Nz+k];
                 (*h_data_in)[i*height + j*NBA+2] = z[i*(Ny*Nz)+l*Nz+k];
                 (*h_data_in)[i*height + j*NBA+3] = px[i*(Ny*Nz)+l*Nz+k];
                 (*h_data_in)[i*height + j*NBA+4] = py[i*(Ny*Nz)+l*Nz+k];
                 (*h_data_in)[i*height + j*NBA+5] = pz[i*(Ny*Nz)+l*Nz+k];
/*		 
                 (*h_data_in)[i*height + j*6+0] = h_ex[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+1] = h_ey[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+2] = h_ez[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+3] = h_hx[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+4] = h_hy[i*(Ny*Nz)+j];    
                 (*h_data_in)[i*height + j*6+5] = h_hz[i*(Ny*Nz)+j];    
		 */
	      }
                 
           }
        }
// TODO: COPY ADDITIONAL LAYER FROM THE RIGHT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        
        *w = width;
        *h = height;
                  
        return 0;
}

int CUDA_WRAP_getNumberOfBeamParticles(Mesh *mesh,Cell *p_CellArray,int Nx,int Ny,int Nz)
{
   int height = Ny*Nz,total_np = 0,np;
   beamParticle *bp;
   
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
         for (int i=(Nx-1)*GetRank(); i < (Nx- 1)*(GetRank() + 1) ; i++)
         {
              long nccc = mesh->GetN(i, j,k);
              Cell &ccc = p_CellArray[nccc];	    
	      Particle *p  = ccc.GetBeamParticles();
		
	      //printf("ijk %d %d %d \n",i,j,k);
	      for(np = 0;(p != NULL);np++)
	      {
		 p = p->p_Next;
	      }
	      total_np += np;
	      
	 }
      }
   }
   
  // printf("rank %d COUNT Number of beam particles %d average %d \n",GetRank(),total_np,total_np/(Nx*Ny*Nz));
   return total_np;
}

int CUDA_WRAP_copy3Dparticles(Mesh *mesh,Cell *p_CellArray,int Np,int Nx,int Ny,int Nz)
{
   int height = Ny*Nz,total_np = 0,np,cuda_total_np = 0,buf_size;
   beamParticle *bp,bp5,bp3;
   
   //printf("IN COPY 3D rank %d Nx %d Np %d \n", GetRank(),Nx,Np);
   
   bp = (beamParticle*)malloc(sizeof(beamParticle)*Np);
   
   if(bp == NULL) printf("beam particles host alloc failed \n");
   
//   int err_cmalloc = cudaMalloc(&beam_particles,sizeof(beamParticle)*Np);
   if(Np < BEAM_DEFAULT_NUMBER)
   {
      buf_size = BEAM_DEFAULT_NUMBER;
   }
   else
   {
      buf_size = 2*Np;
   }      
   int err_cmalloc = cudaMalloc(&beam_particles,sizeof(beamParticle)*buf_size);
   
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
         for (int i= (Nx-1)*GetRank(); i < (Nx-1)*(GetRank()+1)  ; i++)
         {
              long nccc = mesh->GetN(i, j,k);
              Cell &ccc = p_CellArray[nccc];	    
	      Particle *p  = ccc.GetBeamParticles();
		
	      //printf("ijk %d %d %d \n",i,j,k);
	      for(np = 0;(p != NULL);np++)
	      {
#ifdef CUDA_WRAP_PARALLEL_DEBUG		
   	         printf("QQ ijk %d %d %d %e %e %e \n",i,j,k,p->f_X,p->f_Y,p->f_Z);
#endif		 
                 if(total_np == 307)
		 {
		    int i307 = 0; 
		 }
                 
	         double x,y,z,px,py,pz; 
		 
		// printf("rank %d xmin %e total_np %d x %e min %d x %d max %d \n",GetRank(),GetXmax()*GetRank(),total_np,p->f_X,
		//                          (Nx-1)*GetRank(),i,(Nx-1)*(GetRank() + 1)
		// );
		 
      		    beamParticle *p_cuda = bp + cuda_total_np;

		    p_cuda->f_X      = p->f_X;
		    p_cuda->f_Y      = p->f_Y;
		    p_cuda->f_Z      = p->f_Z;
		    p_cuda->f_Px     = p->f_Px;
		    p_cuda->f_Py     = p->f_Py;
		    p_cuda->f_Pz     = p->f_Pz;
		    p_cuda->f_Weight = p->f_Weight;
		    p_cuda->f_Q2m    = p->f_Q2m;
		 //if(total_np < 16) printf("Pz %d %25.15e \n",total_np,p_cuda->f_Pz);
		    p_cuda->i_X      = i - (Nx - 1)*GetRank();
		    p_cuda->i_Y      = j;
		    p_cuda->i_Z      = k;
		//    curbp = *bp;
   		    cuda_total_np++;
            	//    printf("rank %d cuda_total_np %d x %d \n",GetRank(),cuda_total_np,p_cuda->i_X);
		    
		 
		 
		 p = p->p_Next;
	      }
	      total_np++;
	      //total_np += np;
	      
	 }
      }
   }
   
   bp5 = bp[5];
   bp[3] = bp[3];
   
   int err_copy = cudaMemcpy(beam_particles,bp,sizeof(beamParticle)*cuda_total_np,cudaMemcpyHostToDevice);
   
   printf("rank %d errors copy %d malloc %d cuda_total_np %d Np %d \n",GetRank(),err_copy,err_cmalloc,cuda_total_np,Np);

  // printf("rank %d before print particles %d average %d \n",GetRank(),cuda_total_np,cuda_total_np/(Nx*Ny*Nz));
   
   CUDA_WRAP_printParticles(beam_particles,"end copy3D");

  
 //  printf("rank %d Number of beam particles %d average %d \n",GetRank(),cuda_total_np,cuda_total_np/(Nx*Ny*Nz));
   return cuda_total_np;
}

int CUDA_WRAP_beam_prepare(int Nx,int Ny,int Nz,Mesh *mesh,Cell *p_CellArray)
{
    int height,width,height_p,width_p;
    double *h_data_in,*h_data_in_part;
  //  double *d_ex,*d_ey,*d_ez,*d_hx,*d_hy,*d_hz;
    double *ax,*ay,*az,*apx,*apy,*apz;
    int Np;
    
    printf("IN BEAM PREPARE rank %d \n", GetRank());
    Np = CUDA_WRAP_getNumberOfBeamParticles(mesh,p_CellArray,Nx,Ny,Nz);

    printf("BEFORE COPY 3D PARTICLES rank %d Np %d \n", GetRank(),Np);
    Np = CUDA_WRAP_copy3Dparticles(mesh,p_CellArray,Np,Nx,Ny,Nz);
    printf("rank %d after copy particles Np %d \n",GetRank(),Np);
    int err = CUDA_WRAP_alloc_beam_values(Np,BEAM_VALUES_NUMBER,&h_beam_values,&d_beam_values);
    if(err != cudaSuccess) 
    {
       printf("beam prepare error beam values %d \n",err);
    }
   
    //printf("1083  d_RhoBeam3D %p \n", d_RhoBeam3D );  
err = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_RhoBeam3D);
    if(err != cudaSuccess)
    {
       printf("beam prepare error rho beam %d \n",err);
       exit(0);
    }
    else
    {
       printf("beam-prepare d_RhoBeam3D %p \n ", d_RhoBeam3D  );
       //exit(0);       
    }

    err = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_JxBeam3D);
    if(err != cudaSuccess)
    {
       printf("beam prepare error jx  beam %d \n",err);
       exit(0);
    }

    err = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_JyBeam3D);
    if(err != cudaSuccess)
    {
       printf("beam prepare error jy  beam %d \n",err);
       exit(0);
    }

    err = CUDA_WRAP_alloc3DArray(Nx,Ny,Nz,&d_JzBeam3D);
    if(err != cudaSuccess)
    {
       printf("beam prepare error jz  beam %d \n",err);
       exit(0);
    }
    
    err = CUDA_WRAP_alloc3Dfields(Nx,Ny,Nz);
    if(err != cudaSuccess)
    {
        printf("beam prepare alloc3D error %d \n",err);
        exit(0);
    }

    
// makeFieldArray(Nx,Ny,Nz,&width,&height,&h_data_in,d_ex,d_ey,d_ez,d_hx,d_hy,d_hz);
//    makeParticleArray(Nx,Ny,Nz,&width_p,&height_p,&h_data_in_part,ax,ay,az,apx,apy,apz);
    
    //CUDA_WRAP_create_Beam_particle_surface(beam_surface,beam_array,width_p,height_p,h_data_in_part);
    //CUDA_WRAP_create_Beam_particle_surface(field3D,field3Darray,width,height,h_data_in);
    
    
    
    printf("rank %d return beam_prepare Np %d \n",GetRank(),Np);
    
    setBeamNp(Np);
    
    return Np;
}

int CUDA_WRAP_compareBeamCurrents(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray)
{
#ifndef CUDA_WRAP_COMPARE_BEAM_CURRENTS_ALLOWED 
    return 1;
#endif    
  
    double *h_rho_beam,*h_jx_beam,*h_jy_beam,*h_jz_beam;
    FILE *f;
    
    h_jx_beam  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
//    h_jy_beam  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
//    h_jz_beam  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
    h_rho_beam = (double *)malloc(sizeof(double)*Nx*Ny*Nz);
    cudaMalloc(&d_JxBeam3D,Nx*Ny*Nz*sizeof(double));
    cudaMalloc(&d_RhoBeam3D,Nx*Ny*Nz*sizeof(double));

    cudaMemcpy(h_rho_beam,d_RhoBeam3D,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_jx_beam, d_JxBeam3D, sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_jy_beam, d_JyBeam3D, sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
//    cudaMemcpy(h_jz_beam, d_JzBeam3D, sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
    
    
    if((f = fopen("beam3Dcmp.dat","wt")) == NULL)
    {
        puts("beam write failed");
	exit(0);
    }
    int total_cells = 0,wrong_cells = 0;

    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          for(int k = 0;k < Nz;k++)
          {
              long nccc = mesh->GetN(i,j,k);
              Cell &ccc = p_CellArray[nccc];
              
              double drho = fabs(h_rho_beam[i*Ny*Nz+k*Ny+j] - ccc.GetRhoBeam());
              double djx  = fabs(h_jx_beam[i*Ny*Nz +k*Ny+j] - ccc.GetJxBeam());
//              double djy  = fabs(h_jy_beam[i*Ny*Nz +k*Ny+j] - ccc.GetJy());
//              double djz  = fabs(h_jz_beam[i*Ny*Nz +k*Ny+j] - ccc.GetJxBeam());
	      
	      if((djx > BEAM_TOLERANCE) || (fabs(ccc.GetJxBeam()) > BEAM_TOLERANCE) || (drho > BEAM_TOLERANCE) || (fabs(ccc.GetRhoBeam()) > BEAM_TOLERANCE)) total_cells++; 
	      
	      if((djx > BEAM_TOLERANCE) || (drho > BEAM_TOLERANCE))
	      {
              //   fprintf(f," %d %d %d drho %15.5e %15.5e/%15.5e dx %15.5e %15.5e/%15.5e \n",i,j,k,drho,h_rho_beam[i*Ny*Nz+k*Ny+j],ccc.GetRhoBeam(),
		//                                                                               djx, h_jx_beam[i*Ny*Nz +k*Ny+j],ccc.GetJxBeam());
		
		 wrong_cells++;
	      } 
	       fprintf(f," %d %d %d drho %15.5e %15.5e/%15.5e dx %15.5e %15.5e/%15.5e \n",i,j,k,drho,h_rho_beam[i*Ny*Nz+j*Nz+k],ccc.GetRhoBeam(),
		                                                                               djx, h_jx_beam[i*Ny*Nz+j*Nz+k],ccc.GetJxBeam());
          }
       }
   //    puts("=======================================================");
    }
    double xi  = 0.0;
    
    if(total_cells > 0) xi = (double)wrong_cells/(double)total_cells;
    fprintf(f,"wrong cells %.4f \n",xi);
    fclose(f);
    
    
   // printf("wrong cells %.4f \n",xi);
    
    
    free(h_jx_beam);
    free(h_rho_beam);
  
    return 0;
}


int CUDA_WRAP_compare3DField(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray,double *d_ex,int num)
{
  
#ifndef CUDA_WRAP_CHECK_FIELD
    return 1;
#endif
    
    double *h_ex;
    char fname[100];
    FILE *f;
    
    h_ex  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);

    cudaMemcpy(h_ex,d_ex,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
    
    sprintf(fname,"fields3Dcmp%d.dat",num);
    
    if((f = fopen(fname,"wt")) == NULL)
    {
        puts("field write failed");
	exit(0);
    }
    int total_cells = 0,wrong_cells = 0;

    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          for(int k = 0;k < Nz;k++)
          {
              long nccc = mesh->GetN(i,j,k);
              Cell &ccc = p_CellArray[nccc];
	      double t;
	      
	      switch (num)
	      {
		case 0: t = ccc.GetEx();
		        break;
		case 1: t = ccc.GetEy();
		        break;
		case 2: t = ccc.GetEz();
		        break;
		case 3: t = ccc.GetBx();
		        break;
		case 4: t = ccc.GetBy();
		        break;
		case 5: t = ccc.GetBz();
		        break;
	      }
              
              double drho = fabs(h_ex[i*Ny*Nz+k*Ny+j] - t);
	      
      
	      if(drho > BEAM_TOLERANCE)
	      {
		 fprintf(f," %d %d %d drho %15.5e %15.5e/%15.5e \n",i,j,k,drho,t,h_ex[i*Ny*Nz+k*Ny+j]);
		 wrong_cells++;
	      } 
	      
          }
       }
   //    puts("=======================================================");
    }
    double xi  = 0.0;
    
    xi = (double)wrong_cells/((double)Nx*Ny*Nz);
    fprintf(f,"wrong cells %12.4f for field component %d \n",xi,num);
    fclose(f);
    
    
    //printf("wrong cells %12.4f for field component %d \n",xi,num);
    
    
    free(h_ex);
  
    return 0;
}

int CUDA_WRAP_compare3DFields(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray)
{
    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ex3D,0);
    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ey3D,1);
    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ez3D,2);

    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Bx3D,3);
    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_By3D,4);
    CUDA_WRAP_compare3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Bz3D,5);
    
    return 0;
}

int CUDA_WRAP_copy3DField(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray,double *d_ex,int num)
{
    double *h_ex;
    char fname[100];
    FILE *f;
    
    h_ex  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);

    
    //sprintf(fname,"fields3Dcmp%d.dat",num);
    

    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
          for(int k = 0;k < Nz;k++)
          {
              long nccc = mesh->GetN(i,j,k);
              Cell &ccc = p_CellArray[nccc];
	      double t;
	      
	      switch (num)
	      {
		case 0: t = ccc.GetEx();
		        break;
		case 1: t = ccc.GetEy();
		        break;
		case 2: t = ccc.GetEz();
		        break;
		case 3: t = ccc.GetBx();
		        break;
		case 4: t = ccc.GetBy();
		        break;
		case 5: t = ccc.GetBz();
		        break;
	      }
              
              h_ex[i*Ny*Nz+k*Ny+j] = t;
	      
          }
       }
   //    puts("=======================================================");
    }
    
    cudaMemcpy(d_ex,h_ex,sizeof(double)*Nx*Ny*Nz,cudaMemcpyHostToDevice);
  
    return 0;
}

int CUDA_WRAP_copy3DFields(Mesh *mesh,int Nx,int Ny,int Nz,Cell *p_CellArray)
{
    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ex3D,0);
    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ey3D,1);
    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ez3D,2);

    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Bx3D,3);
    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_By3D,4);
    CUDA_WRAP_copy3DField(mesh,Nx,Ny,Nz,p_CellArray,d_Ex3D,5);
    
    return 0;
}


int CUDA_WRAP_writeXYSection(int Nx,int Ny,int Nz,double hx,double hy,double *d_3d,char *name,int step,Mesh *mesh,Cell *p_CellArray,int num)
{
    double *h_ex;
    char fname[100];
    FILE *f;
    
    h_ex  = (double *)malloc(sizeof(double)*Nx*Ny*Nz);

    cudaMemcpy(h_ex,d_3d,sizeof(double)*Nx*Ny*Nz,cudaMemcpyDeviceToHost);
    
    sprintf(fname,"field_%s_XY_%03d_rank%03d.dat",name,step,GetRank());
    
    f = fopen(fname,"wt");
    
    for(int i = 0;i < Nx;i++)
    {
       for(int j = 0;j < Ny;j++)
       {
	   int k = Nz/2;    
           long nccc = mesh->GetN(i,j,k);
           Cell &ccc = p_CellArray[nccc];
	   double t;
	   
#ifdef CUDA_WRAP_FFTW_ALLOWED	   
	      switch (num)
	      {
		case 0: t = ccc.GetEx();
		        break;
		case 1: t = ccc.GetEy();
		        break;
		case 2: t = ccc.GetEz();
		        break;
		case 3: t = ccc.GetBx();
		        break;
		case 4: t = ccc.GetBy();
		        break;
		case 5: t = ccc.GetBz();
		        break;
		case 6: t = ccc.GetRhoBeam();
		        break;
	      }
#else
	   t = h_ex[i*Ny*Nz+k*Ny+j];
#endif	   
	   
    	   fprintf(f," %e %e %15.5e \n",i*hx,j*hy,t);
          
       }
   //    puts("=======================================================");
    }
    double xi  = 0.0;
    
    fclose(f);
    
    free(h_ex);
  
    return 0;
}

int CUDA_WRAP_diagnose(int l_Xsize,int l_Ysize,int l_Zsize,double hx, double hy,int step,Mesh *mesh,Cell *p_CellArray)
{
  CUDA_WRAP_writeXYSection(l_Xsize,l_Ysize,l_Zsize,hx,hy,d_Ex3D,"EX",step,mesh,p_CellArray,0);
  CUDA_WRAP_writeXYSection(l_Xsize,l_Ysize,l_Zsize,hx,hy,d_Ey3D,"EY",step,mesh,p_CellArray,1);
  CUDA_WRAP_writeXYSection(l_Xsize,l_Ysize,l_Zsize,hx,hy,d_Ez3D,"EZ",step,mesh,p_CellArray,2);
  
  CUDA_WRAP_writeXYSection(l_Xsize,l_Ysize,l_Zsize,hx,hy,d_RhoBeam3D,"RhoBeam",step,mesh,p_CellArray,6);
  
  return 0;
}


int CUDA_WRAP_allocLayer(cudaLayer **dl,int Ny,int Nz,int Np)
{
   double *d_Ex,*d_Ey,*d_Ez,*d_Bx,*d_By,*d_Bz,*d_Jx,*d_Jy,*d_Jz,*d_Rho;
   cudaLayer *l,*h_l = (cudaLayer*)malloc(sizeof(cudaLayer));
   beamParticle *p;
   cudaMalloc((void**)&l,sizeof(cudaLayer));

   int err = cudaMalloc(&d_Ex,sizeof(double)*Ny*Nz);
   printf("in allo");
   cudaMalloc(&d_Ey,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Ez,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Bx,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_By,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Bz,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Jx,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Jy,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Jz,sizeof(double)*Ny*Nz);
   cudaMalloc(&d_Rho,sizeof(double)*Ny*Nz);
   cudaMalloc(&p,Np*sizeof(beamParticle));

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
   h_l->particles = p;

   cudaMemcpy(l,h_l,sizeof(cudaLayer),cudaMemcpyHostToDevice);
   
   *dl = l;
   return 0;
   
}


int CUDA_WRAP_printParticleListFromHost(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,char *where,int nstep)
{ 
#ifndef CUDA_WRAP_PRINT_10_PARTICLES
   return 1;
#endif
   
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   int np = 0;
   char name[100];
   FILE *f;

   sprintf(name, "plasmaParticleListFromHost_at_%s_Layer_%3d_nstep_%05d.dat",where,iLayer,nstep);

   if((f = fopen(name,"wt")) == NULL) return 1;
   
   int err = cudaGetLastError();
      
   for (int j=0; j<Ny; j++)
   {
      for (int k=0; k<Nz; k++)
      {
          np = 0;
              //long ncc = mesh->GetN(iLayer, j,k);
	      long ncc = mesh->GetNyz(j,  k);
              Cell &ccc = p_CellArray[ncc];
	      
	      Particle *p  = ccc.GetParticles();
		
	      for(;p;np++)
	      {


		  fprintf(f,"%5d %5d %3d %2d %10.3e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",
                  j,k,
                  np,p->GetSort(),p->f_Q2m,
                  p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Py,p->f_Pz);
		  p = p->p_Next;
		  
	      }
	      
      }
   }
   fclose(f);

   return 0;
}

int CUDA_WRAP_copyLayerToDevice(Mesh *mesh,Cell *p_CellArray,int iLayer,int Ny,int Nz,cudaLayer **dl)
{ 
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   int np = 0;
   

   
   int err = cudaGetLastError();
   printf("in copyLayerToDevice begin err %d \n",err);
      
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
              long ncc = mesh->GetN(iLayer, j,k);
	      //long ncc = mesh->GetNyz(j,  k);
              Cell &ccc = p_CellArray[ncc];
	      
	      Particle *p  = ccc.GetParticles();
		
	      for(;p;np++)
	      {
		  p = p->p_Next;
	      }
	      
      }
   }
   int err1 = cudaGetLastError();
   printf("in copyLayerToDevice particle count err %d \n",err1);
   
   Ex = (double *)malloc(sizeof(double)*Ny*Nz);
   Ey = (double *)malloc(sizeof(double)*Ny*Nz);
   Ez = (double *)malloc(sizeof(double)*Ny*Nz);
   Bx = (double *)malloc(sizeof(double)*Ny*Nz);
   By = (double *)malloc(sizeof(double)*Ny*Nz);
   Bz = (double *)malloc(sizeof(double)*Ny*Nz);
   Jx = (double *)malloc(sizeof(double)*Ny*Nz);
   Jy = (double *)malloc(sizeof(double)*Ny*Nz);
   Jz = (double *)malloc(sizeof(double)*Ny*Nz);
   Rho = (double *)malloc(sizeof(double)*Ny*Nz);
   
   bp = (beamParticle *)malloc(np*sizeof(beamParticle));
   if(bp == NULL) puts("bp NULL");
   int err2 = cudaGetLastError();
  // printf("in copyLayerToDevice alloc err %d np before list composition %d \n",err2,np);
   
//   CUDA_WRAP_allocLayer(dl,Ny,Nz,np);
   
   np = 0;
   
#ifdef  CUDA_WRAP_PARALLEL_DEBUG   
   FILE *f = fopen("layer_being_formed.dat","wt");
#endif   
	      
   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
              long nccc = mesh->GetN(iLayer,j,  k);
              Cell &ccc = p_CellArray[nccc];	    
	      Particle *p  = ccc.GetParticles();
		
	      for(;p;np++)
	      {
		 beamParticle *pc = bp + np;
		 pc->f_X      = p->f_X;
		 pc->f_Y      = p->f_Y;
		 pc->f_Z      = p->f_Z;
		 pc->f_Px     = p->f_Px;
		 pc->f_Py     = p->f_Py;
		 pc->f_Pz     = p->f_Pz;
		 pc->f_Weight = p->f_Weight;
		 pc->f_Q2m    = p->f_Q2m;
		 //if(total_np < 16) printf("Pz %d %25.15e \n",total_np,p_cuda->f_Pz);
		 pc->i_X      = iLayer;
		 pc->i_Y      = j;
		 pc->i_Z      = k;
		 pc->isort    = p->GetSort(); 
#ifdef  CUDA_WRAP_PARALLEL_DEBUG		 
		 fprintf(f,"%10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",np,pc->i_X,pc->i_Y,pc->i_Z,pc->f_X,pc->f_Y,pc->f_Z,pc->f_Px,pc->f_Py,pc->f_Pz);
#endif		 
		 
		 p = p->p_Next;
	      }  
	      long n = j + Ny*k;

              Rho[n] = ccc.GetDens(); // - dens;

              Ex[n]  = ccc.GetEx();
              Ey[n]  = ccc.GetEy();
              Ez[n]  = ccc.GetEz();
	      
              Bx[n]  = ccc.GetEx();
              By[n]  = ccc.GetEy();
              Bz[n]  = ccc.GetEz();

              Jx[n] = ccc.GetJx();
              Jy[n] = ccc.GetJy();
              Jz[n] = ccc.GetJz();
	 
      }
   }
   int err3 = cudaGetLastError();
   printf("in copyLayerToDevice particle list err %d \n",err3);
   
   
#ifdef  CUDA_WRAP_PARALLEL_DEBUG   
   fclose(f);
#endif   
   //double *d_Ex,*d_Ey,*d_Ez,*d_Bx,*d_By,*d_Bz,*d_Jx,*d_Jy,*d_Jz,*d_Rho;
   cudaLayer *h_dl;// = (cudaLayer*)malloc(sizeof(cudaLayer));
  
   printf("cuBeam 1593 \n"); 
   CUDA_WRAP_allocLayerOnHost(&h_dl,Ny,Nz,np);
   
//   cudaMemcpy(h_dl,*dl,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
   
   cudaMemcpy(h_dl->Ex,Ex,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->Ey,Ey,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->Ez,Ez,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);

   cudaMemcpy(h_dl->Bx,Bx,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->By,By,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->Bz,Bz,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   
   cudaMemcpy(h_dl->Jx,Jx,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->Jy,Jy,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   cudaMemcpy(h_dl->Jz,Jz,sizeof(double)*Ny*Nz,cudaMemcpyHostToDevice);
   h_dl->Np = np;
   h_dl->Ny = Ny;
   h_dl->Nz = Nz;   
   int err4 = cudaGetLastError();
   printf("before  CUDA_WRAP_printLayerParticles np %d  (h_dl->particles %p  \n",np,h_dl->particles );

   CUDA_WRAP_printLayerParticles(h_dl,"NOT-YET-FORMED");
   printf("after   CUDA_WRAP_printLayerParticles \n  ");
   
   printf("in copyLayerToDevice particle copy err %d np %d\n",err4,np);
   
   for(int i = 0;i < 10;i++) printf("%d %e %e \n",i,bp[i].f_Y,bp[i].f_Z);

   int errbc = cudaMemcpy(h_dl->particles,bp,np*sizeof(beamParticle),cudaMemcpyHostToDevice);
   
   printf("error particle copy %d \n",errbc);

  // exit(0);
   
   CUDA_WRAP_printLayerParticles(h_dl,"FORMED");

   
   *dl = h_dl;
   //cudaMemcpy(*dl,h_dl,sizeof(cudaLayer),cudaMemcpyHostToDevice);
   int err5 = cudaGetLastError();
   printf("in copyLayerToDevice end err %d \n",err5);
   
//   exit(0);
   return np;
}


int CUDA_WRAP_writeMatrixFromDevice(int Ny,int Nz,double hy,double hz,double *d_m,int iLayer,char *name)
{
    double *h_m = (double *)malloc(Ny*Nz*sizeof(double));
    char s[100];
    FILE *f;
    
    cudaMemcpy(h_m,d_m,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToHost);
    
    sprintf(s,"layer_%s_%03d.dat",name,iLayer);
    f = fopen(s,"wt");
    
    for(int j = 0;j < Ny;j++)
    {
        for(int k = 0;k < Nz;k++)
	{
	    fprintf(f,"%15.5e %15.5e %25.15e \n",j*hy,k*hz,h_m[k*Ny+j]);
	}
    }
    fclose(f);
    
}

__device__ void assignParticles(beamParticle *dst,beamParticle *src)
{
       dst->f_X = src->f_X;
       dst->f_Y = src->f_Y;
       dst->f_Z = src->f_Z;
       dst->f_Px = src->f_Px;
       dst->f_Py = src->f_Py;
       dst->f_Pz = src->f_Pz;
       dst->f_Weight = src->f_Weight;
       dst->f_Q2m    = src->f_Q2m;
       dst->i_X = src->i_X;
       dst->i_Y = src->i_Y;
       dst->i_Z = src->i_Z;
       dst->isort = src->isort;
}


__global__ void getBeamFlyList(beamParticle *beam_particles,int Np_out,double x_min,double x_max,beamParticle *fly_list_min,
			                                                              beamParticle *fly_list_max,int *size_fly)
{
        //unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
        //unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;
	//unsigned int sizeY = gridDim.y*blockDim.y;
	//unsigned int nz = 0;//threadIdx.z;
	unsigned int np;
	int i1,k1,l1;
        double t,t2,t3,t4,t5,t6;
	//double x,y,z,px,py,pz;
	cudaCell ccc,cpc,ccp,cpp,pcc,ppc,pcp,ppp;
	beamParticle *p,*curp;
	int size_fly_list_min,size_fly_list_max,Np = Np_out;
	
//	np = sizeY*nx + ny;
	
//	if(np >= *Np) return;
	
// 	cuPrintf("in beamKernel\n");
	//cuPrintf("grimDim.y %d blockDim.y %d sizeY %d ny %d nx %d np %d \n",gridDim.y,blockDim.y,sizeY,nx,ny,np);
        //return;
	
// 	cuPrintf("in beamflyList\n");
	//return;
	
	//curp = beam_particles;
	size_fly_list_max = 0;
	size_fly_list_min = 0;
	
// 	cuPrintf("xmin %e xmax %e  %d %d \n",x_min,x_max,size_fly_list_min,size_fly_list_max);
	//return;
	
	for(np = 0;np < Np;np++)
	{
	    p = beam_particles + np;
            double x  = p->f_X;
            double y  = p->f_Y;
            double z  = p->f_Z;

	    if((x < x_min) || (x > x_max))
	    {
// 	       cuPrintf("%d flies !!! xmin %e x %e xmax %e \n",np,x_min,x,x_max);
	       
               if(x < x_min)
	       {
	          curp = fly_list_min + (size_fly_list_min);
	          assignParticles(curp,p);
	          (size_fly_list_min)++;
	       }
	       else
	       {
	          if(x > x_max)
	          {
 	             curp = fly_list_max + (size_fly_list_max);
	             assignParticles(curp,p);
	             (size_fly_list_max)++;
		  }
	       }
	       curp = p;
	       p =  beam_particles + Np -1;
	       assignParticles(curp,p);
	       Np--;
	    }
	}
// 	cuPrintf("almost end \n");
	
	
	size_fly[0] = Np;
	size_fly[1] = size_fly_list_min;
	size_fly[2] = size_fly_list_max;
}

int CUDA_WRAP_getFlyList(int *Np,double x_min,double x_max,beamParticle *fly_list_min,int *size_fly_list_min,
			                                                              beamParticle *fly_list_max,int *size_fly_list_max)
{
    beamCurrents bc;
    static int *d_size,h_size[3];
    static int first_call = 1;
  
    dim3 dimBlock(1,1 ,1); 
    dim3 dimGrid(1, 1); 

    if(first_call == 1)
    {
       cudaMalloc(&d_size,3*sizeof(int));
       first_call = 0; 
    }
    

//     cudaPrintfInit();
    
    getBeamFlyList<<<dimGrid, dimBlock>>>(beam_particles,*Np,x_min,x_max,fly_list_min,fly_list_max,d_size);
    
    cudaDeviceSynchronize();
    int err = cudaGetLastError();
    if(err != cudaSuccess)
    {
       printf("fly list error %d\n",err);
       exit(0);
    }
    exit(0);   
    
//     cudaPrintfDisplay(stdout, true);
//     cudaPrintfEnd();
    
    cudaMemcpy(h_size,d_size,3*sizeof(int),cudaMemcpyDeviceToHost);
    
    *Np = h_size[0];
    *size_fly_list_min = h_size[1];
    *size_fly_list_max = h_size[2];
    
    return 0;
}



int addBeamParticles(int *Np,beamParticle *d_fly_list,int size_fly_list)
{
    printf("in addBeamPartices %d \n",size_fly_list);
    cudaMemcpy(beam_particles + (*Np),d_fly_list,size_fly_list*sizeof(beamParticle),cudaMemcpyDeviceToDevice);
    
    *Np += size_fly_list;
    
    return 0;
}

int copyParticlesToDevice(beamParticle *d_p,beamParticle *h_p,int Np)
{
    return cudaMemcpy(d_p,h_p,Np*sizeof(beamParticle),cudaMemcpyHostToDevice);
}

int copyParticlesToHost(beamParticle *h_p,beamParticle *d_p,int Np)
{
    return cudaMemcpy(h_p,d_p,Np*sizeof(beamParticle),cudaMemcpyDeviceToHost);
}

int CUDA_WRAP_allocFly(int Np,beamParticle **d_send_down,beamParticle **d_send_up,beamParticle **d_recv_down,beamParticle **d_recv_up)
{
       int np = Np*FLYING_PARTICLES_RATIO;
       
       cudaMalloc(d_send_down,sizeof(beamParticle)*np);
       cudaMalloc(d_send_up,  sizeof(beamParticle)*np);
       cudaMalloc(d_recv_down,sizeof(beamParticle)*np);
       cudaMalloc(d_recv_up,  sizeof(beamParticle)*np);  
       
       return 0;
}
