#include "write_value.h"

#include <stdlib.h>
#include<stdio.h>



double *h_plasma_values;

int CUDA_WRAP_write_plasma_value(int i,int n,double t)
{
	static int first = 1;
//     int cell_number = i*Ny + j;

    if(first == 1)
    {
        h_plasma_values = (double*)malloc(PLASMA_VALUES_NUMBER*MAX_PLASMA_PARTICLES*sizeof(double));
        first = 0;
    }

	h_plasma_values [i*PLASMA_VALUES_NUMBER + n] = t;


	return 0;
}


int CUDA_WRAP_save_all_plasma_values(Mesh *m,int iSplit,const char *where)
{
    FILE *f,*f_txt;
    char fname[100];
    int nstep  = m->GetControlDomain()->GetCntrl()->GetNstep();


    sprintf(fname,"plasma_values_%s_%010d_%05d.bin",where,nstep,iSplit);


    if((f = fopen(fname,"wb")) == NULL) return 1;

    sprintf(fname,"plasma_values_%s_%010d_%05d.dat",where,nstep,iSplit);


    if((f_txt = fopen(fname,"wt")) == NULL) return 1;


    if((f = fopen(fname,"wb")) == NULL) return 1;

    fwrite(h_plasma_values,sizeof(double),PLASMA_VALUES_NUMBER*MAX_PLASMA_PARTICLES,f);

    fclose(f);

    for(int i = 0;i < MAX_PLASMA_PARTICLES;i++)
    {
        for(int n = 0;n < PLASMA_VALUES_NUMBER;n++)
        {
            fprintf(f_txt,"%10d %10d %25.15e \n",i,n,h_plasma_values[i*PLASMA_VALUES_NUMBER + n]);
        }
    }
    fclose(f_txt);

    return 0;
}
