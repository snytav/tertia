#include "../mesh.h"
#include "../cell3d.h"
#include "cuLayers.h"

#ifndef COPY_LAYER_H

int CUDA_WRAP_get_particles_number(Mesh *mesh,Cell *p_CellArray);

int CUDA_WRAP_copy_from_CellArray2Layer(Mesh *mesh,Cell *p_CellArray,
		cudaLayer **cl,int iLayer);


#endif
