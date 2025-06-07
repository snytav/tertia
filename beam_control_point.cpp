#include "cell3d.h"
#include "mesh.h"



int printBeamDensity3D(Mesh *p_M,char *where)
{
// #ifdef CUDA_WRAP_BEAM_3D_DENSITY_PRINT
    char fname[100];
    FILE *f;
    long Mx,My,Mz,dMx,dMy,dMz;

    p_M->GetSizes(Mx,My,Mz,dMx,dMy,dMz);

    int nstep = p_M->GetControlDomain()->GetCntrl()->GetNstep();

    sprintf(fname,"beam_%s_%010d.dat",where,nstep);

    f = fopen(fname,"wt");

    for (int i = -dMx; i < Mx + dMx - 1; i++) {
      for (int k = -dMz; k < Mz + dMz - 1; k++) {
         for (int j = -dMy; j < My + dMy - 1; j++) {
            Cell& ccc = p_M->GetCell(i,j,k);
	    fprintf(f,"%3d %3d %3d %25.15e %25.15e\n",i,j,k,ccc.GetJxBeam(),ccc.GetRhoBeam());
/*	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();

	    ccc_c.SetFields(fds);
*/
         }
      }
    }
    fclose(f);
// #endif

   return 0;
}

int printBeamParticles(Mesh *p_M,char *where)
{
// #ifdef CUDA_WRAP_BEAM_PARTICLES_PRINT
    char fname[100];
    FILE *f;
    long Mx,My,Mz,dMx,dMy,dMz;

    int nstep = p_M->GetControlDomain()->GetCntrl()->GetNstep();

    p_M->GetSizes(Mx,My,Mz,dMx,dMy,dMz);

    sprintf(fname,"beamParticles_%s_%010d.dat",where,nstep);

    f = fopen(fname,"wt");

    for (int i = -dMx; i < Mx + dMx - 1; i++) {
      for (int k = -dMz; k < Mz + dMz - 1; k++) {
         for (int j = -dMy; j < My + dMy - 1; j++) {
            Cell& ccc = p_M->GetCell(i,j,k);
	    Particle *p = ccc.GetBeamParticles();
	    int num = 0;
	    while(p)
	    {
	       fprintf(f,"%3d %3d %3d %d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",i,j,k,num,p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Pz,p->f_Pz,p->f_Weight,p->f_Q2m);

	       p = p->p_Next;
	       num++;
	    }

/*	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();

	    ccc_c.SetFields(fds);
*/
         }
      }
    }
    fclose(f);
// #endif


   return 0;
}
