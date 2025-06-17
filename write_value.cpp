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


int CUDA_WRAP_save_all_plasma_values(Mesh *m,const char *where)
{
    FILE *f;
    char fname[100];
    int nstep  = m->GetControlDomain()->GetCntrl()->GetNstep();


    sprintf(fname,"particles_%s_%010d.dat",where,nstep);


    if((f = fopen(fname,"wb")) == NULL) return 1;

    fwrite(h_plasma_values,sizeof(double),PLASMA_VALUES_NUMBER*MAX_PLASMA_PARTICLES,f);

    return 0;
}
