#include "mesh.h"


int ControlPoint(Mesh *M,char *where)
{
    int j,k,nstep  = M->GetControlDomain()->GetCntrl()->GetNstep();
    char fname[100];
    FILE *f;
    int l_Mz = M->GetMz(),
        l_My = M->GetMy(),
        l_Mx = M->GetMx();

    Cell *p_CelLayerP = M->get_p_CellLayerP();



    sprintf(fname,"particles_%s_%010d.dat",where,nstep);

    if ((f = fopen(fname,"wt")) == NULL) return 1;


    for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {

//          i=iLayer;
//          int ip = i+1;
         long ncc = M->GetNyz(j,  k);

         long npc = ncc + 1;
         long ncp = ncc + l_sizeY;
         long npp = ncp + 1;
         long nmc = ncc - 1;
         long ncm = ncc - l_sizeY;
         long nmm = ncm - 1;
         long nmp = ncp - 1;
         long npm = npc - l_sizeY;

         Particle *p = NULL;
         Cell &pcc = p_CellLayerP[ncc];
         Cell &ppc = p_CellLayerP[npc];
         Cell &pcp = p_CellLayerP[ncp];
         Cell &ppp = p_CellLayerP[npp];
         Cell &pmc = p_CellLayerP[nmc];
         Cell &pcm = p_CellLayerP[ncm];
         Cell &pmm = p_CellLayerP[nmm];
         Cell &pmp = p_CellLayerP[nmp];
         Cell &ppm = p_CellLayerP[npm];
         double djx = 0., djy = 0., djz = 0.;
         int n = 0;

         p = pcc.p_Particles;

         if (p==NULL)
            continue;

         p_PrevPart = NULL;
         while(p)
         {
            Particle *p_next = p->p_Next;

            int isort = p->GetSort();
            if (isort > 0) {
               int ttest = 0;
            }
//             if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
//                double check1=0;
//             };
//            create_h_plasma_particles(this->getLayerParticles(iLayer));
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,0,(double)j);
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,1,(double)k);
            l_Processed++;
            double weight = p->f_Weight;
            double xp  = p->f_X;
            double yp  = p->f_Y;
            double zp  = p->f_Z;
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,2,weight);
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,3,xp);
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,4,yp);
//            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,5,zp);

            double x = xp;
            double y = yp;
            double z = zp;

            if (xp<0||xp>1 || yp<0||yp>1 || zp<0||zp>1)
            {
               domain()->out_Flog << "Wrong MoveParticles: x="
                  << xp << " y=" << yp << " z=" << zp << "\n";
               domain()->out_Flog.flush();
               exit(-212);
            }

            double px = p->f_Px;
            double py = p->f_Py;
            double pz = p->f_Pz;
            fprintf(f, "%10d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e",
                    n++,
                    x,
                    y,
                    z,
                    px,
                    py,
                    pz);
         }
      }
   }
   fclose(f);

    return 0;
}
