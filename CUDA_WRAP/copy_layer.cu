

int CUDA_WRAP_get_particles_number(Mesh *mesh,Cell *p_CellArray)
{
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   cudaLayer *h_dl;
   int np = 0;

   int err = cudaGetLastError();
   printf("in getLayerParticles begin err %d \n",err);

   for (int k=0; k<Nz; k++)
   {
      for (int j=0; j<Ny; j++)
      {
          long ncc = mesh->GetNyz(j,  k);
          Cell &ccc = p_CellArray[ncc];

	      Particle *p  = ccc.GetParticles();

	      for(;p;np++)
	      {
		  p = p->p_Next;
	      }

      }
   }

   return np;

}
