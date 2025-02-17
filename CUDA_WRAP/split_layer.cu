#include "split_layer.h"

#include "../cells.h"
#include "../mesh.h"


int copyLayerToLayer(beamLayer *to,beamLayer *from)

   cudaMemcpy((*to)->Ex,(*from)->Ex,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->Ey,(*from)->Ey,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->Ez,(*from)->Ez,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);

   cudaMemcpy((*to)->Bx,(*from)->Bx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->By,(*from)->By,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->Bz,(*from)->Bz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   
   cudaMemcpy((*to)->Jx,(*from)->Jx,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->Jy,(*from)->Jy,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
   cudaMemcpy((*to)->Jz,(*from)->Jz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);

   cudaMemcpy((*to)->particles,(*from)->particles,np*sizeof(beamParticle),cudaMemcpyDeviceToDevice);
   
   return 0;
}
