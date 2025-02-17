#include "../run_control.h"
#include <stdio.h>

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

	printf("end alloc values error\n");
#endif
	
	return err;
}

double CUDA_WRAP_check_beam_values(int Np,int num_attr,double *h_p,double *d_p,int blocksize_x,int blocksize_y,char *fname)
{
        int cell_number,wrong_particles = 0;
	double    *h_copy,frac_err,delta = 0.0,*wrong_array,*delta_array;
	int wrong_flag = 0;
	
	FILE *f;
	
	f = fopen(fname,"wt");
	
	wrong_array = (double *)malloc(num_attr*sizeof(double));
	delta_array = (double *)malloc(num_attr*sizeof(double));
//        int width = Ny*Nz; 
//        double *h_data_in;
	
	puts("BEGIN  BEAM-RELATED VALUES sCHECK =============================================================================");
	
	//part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	h_copy   = (double*) malloc(num_attr*Np*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	int err = cudaMemcpy(h_copy,d_p,num_attr*Np*sizeof(double),cudaMemcpyDeviceToHost);

    for(int n = 0;n < num_attr;n++)
    {
        int wpa = 0,wrong_particles = 0;;
	double fr_attr,x,cu_x;
	
	delta = 0.0;
	
        for (int i = 0;i < Np;i++)
        {
	
            cu_x = h_copy[i*num_attr + n];
	    x    = h_p   [i*num_attr + n];
			  
//#ifdef CUDA_WRAP_BEAM_VALUES_DETAILS
			  if((fabs(x-cu_x) > BEAM_TOLERANCE)  
                            &&  (i < blocksize_x*blocksize_y) 
			    )
			//  {
			//   if(i < 50) 
			   {
			       fprintf(f,"%5d %5d %25.15e/%25.15e delta %15.5e \n",n,i,x,cu_x,fabs(cu_x - x));
			   }
			 // }
//#endif			
                          if(delta < fabs(cu_x - x)) delta = fabs(cu_x - x); 
			  
			  if(  fabs(x-cu_x) > BEAM_TOLERANCE)
			  {
			      wrong_particles++;
			      wpa++;
			      //printf(" %d  %d particle %d wrong: x %.2e/%.2e %15.5e\n",i,j,k,x,cu_x,fabs(x-cu_x));
			  }
			  
      	       
#ifdef CUDA_WRAP_PARTICLE_VALUES_DETAILS	     
#endif	     
        }
        fr_attr = (double)wpa/(Np);
        fprintf(f,"value %3d OK %7.2f wrong %7.2f delta %15.5e wpa %10d Np %10d CORRECT %10d \n",n,1.0 - fr_attr,fr_attr,delta,wpa,Np,Np - wpa);
	printf("\n value %3d OK %7.2f wrong %7.2f delta %15.5e wpa %10d Np %10d CORRECT %10d \n",n,1.0 - fr_attr,fr_attr,delta,wpa,Np,Np - wpa);
	if(Np - wpa < blocksize_x*blocksize_y) wrong_flag = 1;
	
	wrong_array[n] = fr_attr;
	delta_array[n] = delta;
	
	//puts("___________________________________________________________________________________________________________");
    }
	
	free(h_copy);
	
	frac_err = (double)wrong_particles/(Np*num_attr);
	
	
	
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
*/	
        if(wrong_flag == 1) printf("\nONE OR MORE VALUES ARE WRONG !!!!!!!!!!!!!!!!!!!!!!!!!\n");
	printf("BEAM-RELATED CHECK OK %.4f wrong %.4f delta %15.5e =================================================\n",
	       1.0-frac_err,frac_err,max_delta);
	fclose(f);
	
        return frac_err;
}
