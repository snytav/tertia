#include "cuCell.h"

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


__global__ void getBeamFlyList(beamParticle *beam_particles,int *Np,double x_min,double x_max,beamParticle *fly_list_min,int *size_fly_list_min,
			                                                              beamParticle *fly_list_max,int *size_fly_list_max)
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
	
//	np = sizeY*nx + ny;
	
//	if(np >= *Np) return;
	
	//cuPrintf("in beamKernel\n");
	//cuPrintf("grimDim.y %d blockDim.y %d sizeY %d ny %d nx %d np %d \n",gridDim.y,blockDim.y,sizeY,nx,ny,np);
        //return;
	
	//curp = beam_particles;
	for(np = 0;np < *Np;np++)
	{
	    p = beam_particles + np;
            double x  = p->f_X;
            double y  = p->f_Y;
            double z  = p->f_Z;

	    if((x < x_min) || (x > x_max))
	    {
               if(x < x_min)
	       {
	          curp = fly_list_min + (*size_fly_list_min);
	          assignParticles(curp,p);
	          (*size_fly_list_min)++;
	       }
	       else
	       {
	          if(x > x_max)
	          {
 	             curp = fly_list_max + (*size_fly_list_max);
	             assignParticles(curp,p);
	             (*size_fly_list_max)++;
		  }
	       }
	       curp = p;
	       p =  beam_particles + *Np -1;
	       assignParticles(curp,p);
	       (*Np)--;
	    }
	}
}

int CUDA_WRAP_getFlyList(int *Np,double x_min,double x_max,beamParticle *fly_list_min,int *size_fly_list_min,
			                                                              beamParticle *fly_list_max,int *size_fly_list_max)
{
    beamCurrents bc;
  
#ifndef CUDA_WRAP_FFTW_ALLOWED    
    dim3 dimBlock(1,1 ,1); 
    
    dim3 dimGrid(1, 1); 
        
    getBeamFlyList<<<dimGrid, dimBlock>>>(beam_particles,Np,x_min,x_max,fly_list_min,size_fly_list_min,fly_list_max,size_fly_list_max);
#else
    
#endif    
    
    return 0;
}



int addBeamParticles(int *Np,beamParticle *d_fly_list,int size_fly_list)
{
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