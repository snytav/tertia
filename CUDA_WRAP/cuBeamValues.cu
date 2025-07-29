#include "../run_control.h"
#include <stdio.h>
#include "labels.h"
#include "cuParticles.h"

#include <string>
using namespace std;

int CUDA_WRAP_alloc_beam_values(int Np,int num_attr,double **h_p,double **d_p)
{
        int err;
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 

        puts("in alloc values");
	
	*h_p   = (double*) malloc(num_attr*Np*sizeof(double));
	
	err = cudaMalloc((void**)d_p,num_attr*Np*sizeof(double));
	printf("alloc values error %d \n",err);
	
	cudaMemset(*d_p,0,num_attr*Np*sizeof(double));
	
	memset(*h_p,0,num_attr*Np*sizeof(double));

	int err1 = cudaGetLastError();

	printf("end alloc values error %d\n",err1);
#endif
	
	return err;
}

double printAttributesTable(double *h_copy,double *h_p,int Np,int num_attr,char *beam_or_plasma,int nstep)
{
	FILE *f,*f_out,*f_dump;
	char fname_attr[1000];
	char name_out[100];
	char name_dump[100];
	int wrong_particles;
	double frac_rude = 0.0,frac_medium = 0.0,frac_light = 0.0,frac;

	sprintf(fname_attr,"attr_%s_nstep_%010d.dat",beam_or_plasma,nstep);
	if((f = fopen(fname_attr,"wt")) == NULL) return -1.0;
	
//	double wrong_array = (double *)malloc(num_attr*sizeof(double));
//	double delta_array = (double *)malloc(num_attr*sizeof(double));
//	
//	wrong_array = (double *)malloc(num_attr*sizeof(double));
//	delta_array = (double *)malloc(num_attr*sizeof(double));
//        int width = Ny*Nz; 
//        double *h_data_in;
	

	for(int n = 0;n < num_attr; n++)
    {
	   for (int i = 0;i < Np;i++)
	   {
           int wpa = 0,wrong_particles = 0;;
     	   double fr_attr,x,cu_x;
	
           cu_x = h_copy[n*num_attr + i];
     	   x    = h_p   [n*num_attr + i];
     	   
     	   if(fabs(cu_x - x) > PARTICLE_TOLERANCE)
     	   {
			   wrong_particles++;
		   }
		   
        }
        frac = (((double)wrong_particles)/Np)*100;
        fprintf(f,"attribute %3d wrong particles %10d of %10d, %2f  ",n,wrong_particles,Np,frac);
        if(frac < 1.0)
        {
			frac_light += 1.0;
		}
		else
		{
			if (frac < 30)
			{
				frac_medium += 1.0;
			}
			else
			{
			    frac_rude += 1.0;
			}
		}
	}
	frac_light  /= num_attr;
	frac_medium /= num_attr;
	frac_rude   /= num_attr;
	fprintf(f,"attributes error light %10.3e medium %10.3e rude %10.3e \n ",frac_light,frac_medium,frac_rude);
	fclose(f);
	
	return frac_rude;
}

double CUDA_WRAP_check_beam_values(int Np,int num_attr,double *h_p,double *d_p,int blocksize_x,int blocksize_y,char *fname,char *beam_or_plasma,int nstep)
{
        int cell_number,wrong_particles = 0;
	double    *h_copy,frac_err,delta = 0.0,*wrong_array,*delta_array,res;
	int wrong_flag = 0;
	printf("in ceck beam values %s \n ",beam_or_plasma);
	
	FILE *f,*f_out,*f_dump;
	char fname_attr[1000];
	char name_out[100];
	char name_dump[100];

	

	sprintf(name_out,"%s_nstep_%010d.dat",beam_or_plasma,nstep);
	sprintf(name_dump,"VLPL_CPU_values_%s_nstep_%010d.dat",beam_or_plasma,nstep);
	printf("fname_attr %s out %s dump %s \n",fname_attr, fname,name_out,name_dump);
	
	if((f = fopen(fname_attr,"wt")) == NULL) return -1.0;
	f_out = fopen(name_out,"wt");
	if((f_dump = fopen(name_dump,"wt")) == NULL) return -1.0;
	printf("files out %s dump %s  opened  \n", name_out,name_dump);
	
	wrong_array = (double *)malloc(num_attr*sizeof(double));
	delta_array = (double *)malloc(num_attr*sizeof(double));
//        int width = Ny*Nz; 
//        double *h_data_in;
	
	printf("BEGIN  %s-RELATED VALUES sCHECK =============================================================================",beam_or_plasma);
	
	//part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	h_copy   = (double*) malloc(num_attr*Np*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	int err = cudaMemcpy(h_copy,d_p,num_attr*Np*sizeof(double),cudaMemcpyDeviceToHost);
	if((res = printAttributesTable(h_copy,h_p,Np,num_attr,beam_or_plasma,nstep)) < 0.0) return -1.0;
        int Np1 = Np;
	string s = "";
	fprintf(f_out,"%15s ",s.c_str());
	for (int i = 0;i < Np1;i++)
	{
	    fprintf(f_out,"%15d ",i);
	}
	fprintf(f_out,"\n");
        for(int n = 0;n < num_attr; n++)
        {
           int wpa = 0,wrong_particles = 0;;
	   double fr_attr,x,cu_x;
	
	   for (int i = 0;i < Np1;i++)
           {
	
            cu_x = h_copy[n*num_attr + i];
     	    x    = h_p   [n*num_attr + i];
			  
            fprintf(f_dump,"attr %10d np  %10d cpu %25.15e gpu %25.15e delta %15.5e \n",n,i,x,cu_x,fabs(cu_x - x));
           }
       }
       fclose(f_dump);	
       return res;
   }
   
/*			  
      	       
#ifdef CUDA_WRAP_PARTICLE_VALUES_DETAILS	     
#endif	     
        }
        fprintf(f_out,"\n");
		fprintf(f_dump,"\n");
        fr_attr = (double)wpa/(Np);
        fprintf(f,"value %3d OK %7.2f wrong %7.2f delta %15.5e wpa %10d Np %10d CORRECT %10d \n",n,1.0 - fr_attr,fr_attr,delta,wpa,Np,Np - wpa);
	//printf("\n value %3d OK %7.2f wrong %7.2f delta %15.5e wpa %10d Np %10d CORRECT %10d \n",n,1.0 - fr_attr,fr_attr,delta,wpa,Np,Np - wpa);
	if(Np - wpa < blocksize_x*blocksize_y) wrong_flag = 1;
	
	wrong_array[n] = fr_attr;
	delta_array[n] = delta;
	
	//puts("___________________________________________________________________________________________________________");
  *///  }

/*	free(h_copy);
	
	frac_err = (double)wrong_particles/(Np*num_attr);
	fclose(f_out);
	
	
/*	FILE *wf,*df;
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
	}*/
       /*
	double max_delta = 0.0;
	for(int i = 0;i < num_attr;i++)
	{
//	    fprintf(wf,"%15.5e ",wrong_array[i]);
//	    fprintf(df,"%15.5e ",delta_array[i]);
	    
	    if(max_delta < delta_array[i]) 
	    {
	       max_delta = delta_array[i];
//	       last_max_delta_value = i;
	    }
	    
	      
	}
//	fprintf(wf,"\n");
//	fprintf(df,"\n");
	
//	fclose(wf);
//	fclose(df);
  
          
//	free(wrong_array);
//	free(delta_array);
	
/*	last_wrong = frac_err;
	last_delta = max_delta;
	
        if(wrong_flag == 1) printf("\nONE OR MORE VALUES ARE WRONG !!!!!!!!!!!!!!!!!!!!!!!!!\n");
	printf("%s-RELATED CHECK OK %.4f wrong %.4f delta %15.5e =================================================\n",beam_or_plasma,
	       1.0-frac_err,frac_err,max_delta);
	fclose(f);
	
        return frac_err;
*/	
//}
