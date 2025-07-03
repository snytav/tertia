#include "cell3d.h"
#include "mesh.h"
#include "beam_control_point.h"



int plasmaControlPoint(Mesh *M,Cell *p_CellLayerP,const char *where)
{
	
    int j,k,nstep  = M->GetControlDomain()->GetCntrl()->GetNstep();
     int n = 0;
    char fname[100],field_name[100];
    FILE *f,*f_field;
    int l_Mz = M->GetMz(),
        l_My = M->GetMy(),
        l_Mx = M->GetMx();
   int l_Processed;


    sprintf(fname,"particles_%s_%010d.dat",where,nstep);
    sprintf(field_name,"fields_%s_%010d.dat",where,nstep);

    if ((f = fopen(fname,"wt")) == NULL) return 1;
    if ((f_field = fopen(field_name,"wt")) == NULL) return 1;


    for (k=0; k< l_Mz; k++)
   {
      for (j=0; j< l_My; j++)
      {

//          i=iLayer;
//          int ip = i+1;
         long ncc = M->GetNyz(j,  k);

         long npc = ncc + 1;
         long ncp = ncc + l_My;
         long npp = ncp + 1;
         long nmc = ncc - 1;
         long ncm = ncc - l_My;
         long nmm = ncm - 1;
         long nmp = ncp - 1;
         long npm = npc - l_My;

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

         fprintf(f_field,"j %10d k %10d Ex %25.15e Ey %25.15e Ez %25.15e Bx %25.15e By %25.15e Bz %25.15e  Jx %25.15e Jy %25.15e Jz %25.15e \n",
                          j,k,
                          pcc.GetEx(),
                          pcc.GetEy(),
                          pcc.GetEz(),
                          pcc.GetBx(),
                          pcc.GetBy(),
                          pcc.GetBz(),
                          pcc.GetJx(),
                          pcc.GetJy(),
                          pcc.GetEz()
                 );




         p = pcc.GetParticles();   //p_Particles;

         if (p==NULL)
            continue;

         Particle *p_PrevPart = NULL;
         int n_loc = 0;
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
               M->GetControlDomain()->out_Flog << "Wrong MoveParticles: x="
                  << xp << " y=" << yp << " z=" << zp << "\n";
               M->GetControlDomain()->out_Flog.flush();
               exit(-212);
            }

            double px = p->f_Px;
            double py = p->f_Py;
            double pz = p->f_Pz;
            fprintf(f, "j %10d k %10d n_loc %05d n %10d %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e\n",
                    j,k,
                    n_loc++,
                    n++,
                    x,
                    y,
                    z,
                    px,
                    py,
                    pz);
            p = p_next;
         }

         int q1 = 0;

      }
      int q2 = 0;
   }
   fclose(f_field); ///
   fclose(f);

    return 0;
}


int ControlPoint(Mesh *M,const char *where)
{
   printBeamDensity3D(M,where);
   printBeamParticles(M,where);
   string name = where;
   name = name + "_Player";

   Cell * p_CellLayerP = M->get_p_CellLayerP();

   plasmaControlPoint(M,p_CellLayerP,name.c_str());

   name = where;
   name = name + "_Сlayer";

   Cell * p_CellLayerС = M->get_p_CellLayerC();

   plasmaControlPoint(M,p_CellLayerС,name.c_str());


}
