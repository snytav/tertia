#include "write_value.h"

#include <stdlib.h>

double *h_plasma_values;

int CUDA_WRAP_write_plasma_value(int i,int num_attr,int n,double t)
{
	static int first = 1;
//     int cell_number = i*Ny + j;

    if(first == 0)
    {
        h_plasma_values = (double*)malloc(PLASMA_VALUES_NUMBER*MAX_PLASMA_PARTICLES*sizeof(double));
        first = 0;
    }

	h_plasma_values [i*num_attr + n] = t;


	return 0;
}
