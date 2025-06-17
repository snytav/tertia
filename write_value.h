#ifndef WRITE_VALUE_H

#include "cell3d.h"
#include "mesh.h"

#define PLASMA_VALUES_NUMBER 200
#define MAX_PLASMA_PARTICLES 50000


int CUDA_WRAP_write_plasma_value(int i,int n,double t);

int CUDA_WRAP_save_all_plasma_values(Mesh *m,const char *where);

#endif
