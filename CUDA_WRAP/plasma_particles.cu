
#include "cuCell.h"

#include "../particles.h"
#include "../cells.h"
#include "../mesh.h"
#include "cuBeam.h"
#include "cuBeamValues.h"
#include "../run_control.h"
#include "plasma_particles.h"
#include <stdio.h>
//#include "cuPrintf.cu"
#include "cuLayers.h"
#include <sys/time.h>


double *h_plasma_values,*d_plasma_values;

cudaLayer *cl,*pl;

void __device__ cuDepositCurrentsInCellSplit(
                                 beamParticle *p, int isort, int Ny,
                                 int i, int j, int k, 
                                 double Vx, double Vy, double Vz, 
                                 double x, double y, double z, 
                                 double djx, double djy, double djz, double drho,cudaLayer *cl,double *d_p,int np);
                                 
__device__ double cuda_atomicAddP(double *address, double val)
{
    double assumed,old=*address;
    do {
        assumed=old;
        old= __longlong_as_double(atomicCAS((unsigned long long int*)address,
                    __double_as_longlong(assumed),
                    __double_as_longlong(val+assumed)));
    }while (assumed!=old);

    return old;
}                                 

int particlesPrepareAtLayer(Mesh *mesh,Cell *p_CellArrayP,Cell *p_CellArrayC,int iLayer,int Ny,int Nz,int Np)
{
    static int prepareFirstCall = 1;
      
    if(prepareFirstCall)
    {
        CUDA_WRAP_alloc_beam_values(Np,PLASMA_VALUES_NUMBER,&h_plasma_values,&d_plasma_values);
        prepareFirstCall = 0;
    }
    
    
    return 0;
}

__device__ void copyParticle(beamParticle *dst,beamParticle *src)
{
    dst->f_Px     = src->f_Px;
    dst->f_Py     = src->f_Py;
    dst->f_Pz     = src->f_Pz;
    dst->f_X      = src->f_X;
    dst->f_Y      = src->f_Y;
    dst->f_Z      = src->f_Z;
    dst->i_X      = src->i_X;
    dst->i_Y      = src->i_Y;
    dst->i_Z      = src->i_Z;
    dst->isort    = src->isort;
    dst->f_Weight = src->f_Weight;
    dst->f_Q2m    = src->f_Q2m;
}


__global__ void cuMoveSplitParticlesKernel(int iLayer,int iSplit,int Np,cudaLayer *cl,cudaLayer *pl,int Ny,int Nz,double hx,double hy,double hz,
                                     double *djx0,double *djy0,double *djz0,double *drho0,int iFullStep,double *d_p)
{
         unsigned int nx = blockIdx.x * blockDim.x + threadIdx.x; 
         unsigned int ny = blockIdx.y * blockDim.y + threadIdx.y;
         unsigned int sizeY = gridDim.y*blockDim.y;
         beamParticle *p;
         int np;
         unsigned int j = nx,k = ny;
         
         
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("moveSplit \n");
#endif         
//         if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;

	 np = sizeY*nx + ny;
	 
#ifdef PLASMA_MOVE_CUPRINTF	 
	 cuPrintf("np %d Np %d \n",np,Np);
#endif	 
	
	 if(np >= pl->Np) return;
#ifdef PLASMA_MOVE_CUPRINTF	 
	 cuPrintf("after Np check Np %d \n",pl->Np);
#endif	 
	 //return;
	 p = pl->particles + np;
         j = p->i_Y;
         //cuPrintf("jread %d \n",j);
         //if(iLayer == 120 && iSplit == 1 && iFullStep == 0) return;
         //j = 0;
#ifdef PLASMA_MOVE_CUPRINTF	 
         cuPrintf("first-j \n");
#endif    
         
         //return;
         k = p->i_Z;  
//#ifdef PLASMA_VALUES_CUPRINTF         
//         if(np < 50) cuPrintf("jjj np %d j %d k %d i %d %e %e %e\n",np,p->i_Y,p->i_Z,p->i_X,p->f_Px,p->f_Py,p->f_Pz);
//#endif         
         
         write_plasma_value(np,PLASMA_VALUES_NUMBER,0,d_p,(double)j);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,1,d_p,(double)k);
         //return; 
         int i=iLayer;
         int ip = i+1;
         long ncc = k*Ny + j;
         long l_sizeY = Ny;

         long npc = ncc + 1;
         long ncp = ncc + l_sizeY;
         long npp = ncp + 1;
         long nmc = ncc - 1;
         long ncm = ncc - l_sizeY;
         long nmm = ncm - 1;
         long nmp = ncp - 1;
         long npm = npc - l_sizeY;

         double djx = 0., djy = 0., djz = 0.;
          
          
        //    isort = p->GetSort();
         double weight = p->f_Weight;
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("weight %e \n",weight);
#endif         
         //return;
         
         double xp  = p->f_X;
         double yp  = p->f_Y;
         double zp  = p->f_Z;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,2,d_p,weight);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,3,d_p,xp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,4,d_p,yp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,5,d_p,zp);
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("read coords \n");
#endif         
         //return;

         double x = xp;
         double y = yp;
         double z = zp;

         double px = p->f_Px;
         double py = p->f_Py;
         double pz = p->f_Pz;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,6,d_p,px);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,7,d_p,py);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,8,d_p,pz);
         
         double pxp = px;
         double pyp = py;
         double pzp = pz;
         double gammap = sqrt(1. + px*px + py*py + pz*pz);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,11,d_p,gammap);
         
         double Vx = px / gammap;
//         maxVx = max(maxVx,fabs(Vx));
         double q2m = p->f_Q2m;

         double Vxp = Vx;
         double Vyp = py/gammap;
         double Vzp = pz/gammap;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,9,d_p,Vxp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,10,d_p,Vyp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,11,d_p,Vzp);
         

         double y_est = j + yp + Vyp/(1.-Vxp)*hx/hy;
         double z_est = k + zp + Vzp/(1.-Vxp)*hx/hz;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,12,d_p,y_est);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,13,d_p,z_est);

         while (y_est > Ny) y_est -= Ny;
         while (y_est < 0)  y_est += Ny;
         while (z_est > Nz) z_est -= Nz;
         while (z_est < 0)  z_est += Nz;

         write_plasma_value(np,PLASMA_VALUES_NUMBER,14,d_p,y_est);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,15,d_p,z_est);

         int j_est = y_est;
         int k_est = z_est;

         double ym = y_est - j_est;
         double zm = y_est - z_est;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,16,d_p,ym);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,17,d_p,zm);

         if (ym + yp != 0.) 
         {
               double dummy = 0.;
         }
         //cuPrintf("begin interplolate \n");
//         if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;


         double ayc = 1.-yp;
         double ayp = yp;
         double azc = 1.-zp;
         double azp = zp;

         double myc = 1.-ym;
         double myp = ym;
         double mzc = 1.-zm;
         double mzp = zm;

         double apcc = ayc*azc;
         double appc = ayp*azc;
         double apcp = ayc*azp;
         double appp = ayp*azp;

         double accc = myc*mzc;
         double acpc = myp*mzc;
         double accp = myc*mzp;
         double acpp = myp*mzp;
         double ex, ey, ez;
         double exp, eyp, ezp;
         double bxp, byp, bzp;
         double exm, eym, ezm;
         double bxm, bym, bzm;

         double bx=0.;
         double by=0.;
         double bz=0.;
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("fields preparation \n");
#endif         
         //cuPrintf("after 29 \n");
        // if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;
         //return;

         exp = pl->Ex[ncc];
         //if(np == 36)
         //{
         //   cuPrintf("EXP36 np %d j %d k %d exp %25.15e\n",np,j,k,exp);
          //  return;
         //}
         
        // cuPrintf("after pl->Ex npc %d k %d j %d Ny %d \n",npc,k,j,Ny);
        // if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;         
         write_plasma_value(np,PLASMA_VALUES_NUMBER,18,d_p,exp);
         
//            exp = apcc*pcc.f_Ex + appc*ppc.f_Ex + apcp*pcp.f_Ex + appp*ppp.f_Ex;

         //cuPrintf("after 18 \n");
         //if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;

         eyp = pl->Ey[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,19,d_p,eyp);
//            eyp = apcc*pcc.f_Ey + appc*ppc.f_Ey + apcp*pcp.f_Ey + appp*ppp.f_Ey;

         ezp = pl->Ez[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,20,d_p,ezp);
         
//            ezp = apcc*pcc.f_Ez + appc*ppc.f_Ez + apcp*pcp.f_Ez + appp*ppp.f_Ez;

         //cuPrintf("after 20 \n");
         //if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;

       bxp = pl->Bx[ncc];
//            bxp = apcc*pl->Bx[npc] + appc*pl->Bx[npp] + apcp*pl->Bx[ncp] + appp*pl->Bx[npp];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,21,d_p,bxp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,22,d_p,accc);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,23,d_p,appc);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,24,d_p,apcp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,25,d_p,appp);
         //write_plasma_value(np,PLASMA_VALUES_NUMBER,26,d_p,pl->Bx[npc]);
         //write_plasma_value(np,PLASMA_VALUES_NUMBER,27,d_p,pl->Bx[npp]);
         //write_plasma_value(np,PLASMA_VALUES_NUMBER,28,d_p,pl->Bx[ncp]);
         //write_plasma_value(np,PLASMA_VALUES_NUMBER,29,d_p,pl->Bx[npp]);
         
         //cuPrintf("after 29 \n");
         //if(iLayer == 120 && iSplit == 1 && iFullStep == 0)return;
         
         byp = pl->By[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,30,d_p,byp);
         
//            byp = apcc*pcc.f_By + appc*ppc.f_By + apcp*pcp.f_By + appp*ppp.f_By;

         bzp = pl->Bz[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,31,d_p,bzp);
//            bzp = apcc*pcc.f_Bz + appc*ppc.f_Bz + apcp*pcp.f_Bz + appp*ppp.f_Bz;

         exm = cl->Ex[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,32,d_p,exm);
         
//         exm = accc*ccc.f_Ex + acpc*cpc.f_Ex + accp*ccp.f_Ex + acpp*cpp.f_Ex;

         eym = cl->Ey[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,33,d_p,eym);
         
//         eym = accc*ccc.f_Ey + acpc*cpc.f_Ey + accp*ccp.f_Ey + acpp*cpp.f_Ey;

         ezm = cl->Ez[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,34,d_p,ezm);
         
//         ezm = accc*ccc.f_Ez + acpc*cpc.f_Ez + accp*ccp.f_Ez + acpp*cpp.f_Ez;

         bxm = cl->Bx[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,35,d_p,bxm);
         
//         bxm = accc*ccc.f_Bx + acpc*cpc.f_Bx + accp*ccp.f_Bx + acpp*cpp.f_Bx;

         bym = cl->By[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,36,d_p,bym);
         
//         bym = accc*ccc.f_By + acpc*cpc.f_By + accp*ccp.f_By + acpp*cpp.f_By;

         bzm = cl->Bz[ncc];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,37,d_p,bzm);
         
//         bzm = accc*ccc.f_Bz + acpc*cpc.f_Bz + accp*ccp.f_Bz + acpp*cpp.f_Bz;
#ifdef PLASMA_MOVE_CUPRINTF
         cuPrintf("fields prepared \n");
#endif         
         //return;

         ex = 0.5*(exp+exm);
         ey = 0.5*(eyp+eym);
         ez = 0.5*(ezp+ezm);
         
         bx = 0.5*(bxp+bxm);
         by = 0.5*(byp+bym);
         bz = 0.5*(bzp+bzm);
         
         write_plasma_value(np,PLASMA_VALUES_NUMBER,38,d_p,ex);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,39,d_p,ey);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,40,d_p,ez);

         write_plasma_value(np,PLASMA_VALUES_NUMBER,22,d_p,bx);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,23,d_p,by);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,24,d_p,bz);
         

         double ex1 = ex;
         double ey1 = ey;
         double ez1 = ez;

/*
         if(isort > 0 && iAtomTypeArray[isort] > 0 && iFullStep) {//if and only if ionizable ions
            int iZ = p->GetZ();

            if (iZ < iAtomTypeArray[isort]) {
               double field = sqrt(ex*ex + ey*ey + ez*ez);
               p->Ionize(&ccc, field);
               p_next = p->p_Next;
               //	      if (iZ == 0) continue;
            };
            q2m *= iZ;
            weight *= iZ;
         }
*/
         ex *= q2m*hx/2.;
         ey *= q2m*hx/2.;
         ez *= q2m*hx/2.;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,41,d_p,ex);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,42,d_p,ey);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,43,d_p,ez);

         px += ex/(1.-Vx);
         py += ey/(1.-Vx);
         pz += ez/(1.-Vx);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,44,d_p,px);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,45,d_p,py);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,46,d_p,pz);
         

         double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!

     //    if (f_GammaMax < gamma)
       //      f_GammaMax = gamma;

         double gamma_r = 1./gamma;																	 //!!!!!!

         double bXext = 0.0,bYext = 0.0,bZext = 0.0;
         bx += bXext;
         by += bYext;
         bz += bZext;

         double bx1 = bx;
         double by1 = by;
         double bz1 = bz;

            bx = bx*gamma_r*q2m/(1.-Vx)*hx/2.;
            by = by*gamma_r*q2m/(1.-Vx)*hx/2.;
            bz = bz*gamma_r*q2m/(1.-Vx)*hx/2.;

            double co = 2./(1. + (bx*bx) + (by*by) + (bz*bz));

            double p3x = py*bz - pz*by + px;
            double p3y = pz*bx - px*bz + py;
            double p3z = px*by - py*bx + pz;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,47,d_p,p3x);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,48,d_p,p3y);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,49,d_p,p3z);            

            p3x *= co;
            p3y *= co;
            p3z *= co;

            double px_new = p3y*bz - p3z*by;
            double py_new = p3z*bx - p3x*bz;
            double pz_new = p3x*by - p3y*bx;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,50,d_p,px_new);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,51,d_p,py_new);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,52,d_p,pz_new);            

            px += ex/(1.-Vx) + px_new;
            py += ey/(1.-Vx) + py_new;
            pz += ez/(1.-Vx) + pz_new;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,53,d_p,px);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,54,d_p,py);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,55,d_p,pz);            
            gamma = sqrt(1. + px*px + py*py + pz*pz);
            
            double Vxm = px/gamma;
            double Vym = py/gamma;
            double Vzm = pz/gamma;

            Vx = 0.5*(Vxm+Vxp);
            //maxVx = max(maxVx,fabs(Vx));
            double Vy = 0.5*(Vym+Vyp);
            double Vz = 0.5*(Vzm+Vzp);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,56,d_p,Vx);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,57,d_p,Vy);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,58,d_p,Vz);                      

            int isort = p->isort;

            djx = weight*djx0[isort]*Vxm;
            djy = weight*djy0[isort]*Vym;
            djz = weight*djz0[isort]*Vzm;
            double drho = weight*drho0[isort];
         write_plasma_value(np,PLASMA_VALUES_NUMBER,59,d_p,djx);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,60,d_p,djy);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,61,d_p,djz);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,62,d_p,drho);                      
        
            

            double xtmp = 0.;
            double ytmp = yp;
            double ztmp = zp;

            double full = 1.;
            double part_step = 1.;

            int itmp = iLayer;
            int jtmp = j;
            int ktmp = k;

            djx *= 1./(1.-Vx);
            djy *= 1./(1.-Vx);
            djz *= 1./(1.-Vx);
            drho *= 1./(1.-Vx);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,63,d_p,djx);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,64,d_p,djy);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,65,d_p,djz);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,66,d_p,drho);                      
            

            double dy = Vy*hx/hy;
            double dz = Vz*hx/hz;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,67,d_p,dy);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,68,d_p,dz);                      

            dy = dy/(1.-Vx);
            dz = dz/(1.-Vx);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,69,d_p,dy);                      
         write_plasma_value(np,PLASMA_VALUES_NUMBER,70,d_p,dz); 
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("computing end \n");
#endif         
         //return;
         
            double partdx = 0.;
            double step = 1.;
            // --- first half-step


            // Particle pusher cel per cell//////////////////////////////////////


            int j_jump = j;
            int k_jump = k;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,71,d_p,(int)j_jump);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,72,d_p,(int)k_jump);               
            xtmp = 0;
            ytmp = yp;
            ztmp = zp;
         
         write_plasma_value(np,PLASMA_VALUES_NUMBER,73,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,74,d_p,ztmp);               
         
            if (fabs(dy)>1. || fabs(dz)>1.) {
               if (fabs(dy) > fabs(dz)) {
                  step = partdx = fabs(dy);
               } else {
                  step = partdx = fabs(dz);
               };
            }
         write_plasma_value(np,PLASMA_VALUES_NUMBER,75,d_p,step);               

            if (partdx < 1.) {
               partdx = step = 1.;
            }
         write_plasma_value(np,PLASMA_VALUES_NUMBER,76,d_p,partdx);               
#ifdef PLASMA_MOVE_CUPRINTF         
         cuPrintf("before 1st while ytmp,ztmp %e %e\n",ytmp,ztmp);
#endif         
         //return;

            while (partdx>0.) {
               if (partdx > 1.) {
                  partdx -= 1.;
                  part_step = 1./step;
               } else {
                  part_step = partdx/step;
                  partdx = 0.;
               }
               xtmp = 0.;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,77,d_p,partdx);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,78,d_p,part_step);               
               

               ytmp += dy*part_step + j_jump;
               ztmp += dz*part_step + k_jump;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,79,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,80,d_p,ztmp);               
               

               while (ytmp > Ny) ytmp -= Ny;
               while (ytmp < 0) ytmp += Ny;
               while (ztmp > Nz) ztmp -= Nz;
               while (ztmp < 0) ztmp += Nz;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,81,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,82,d_p,ztmp);               
               

               int j_jump = ytmp;
               int k_jump = ztmp;
               ytmp -= j_jump;
               ztmp -= k_jump;
               if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) {
                  double checkpoint21 = 0.;
               };
         write_plasma_value(np,PLASMA_VALUES_NUMBER,83,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,84,d_p,ztmp);               

               xtmp = 0;

               int itmp = iLayer;
               int jtmp = j_jump;
               int ktmp = k_jump;
         write_plasma_value(np,PLASMA_VALUES_NUMBER,85,d_p,(double)j_jump);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,86,d_p,(double)k_jump);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,94,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,95,d_p,ztmp);                   

//               int ntmp = GetN(itmp,jtmp,ktmp);

               if (fabs(djy) > 0.) {
                  int check = 0;
               };
#ifdef PLASMA_MOVE_CUPRINTF               
               //cuPrintf("partdx %e j %d \n",partdx,j);
#endif               
               //return;
               cuDepositCurrentsInCellSplit(p, isort,Ny, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
                  djx*part_step, djy*part_step, djz*part_step, drho*part_step,cl,d_p,np);
         write_plasma_value(np,PLASMA_VALUES_NUMBER,96,d_p,ytmp);               
         write_plasma_value(np,PLASMA_VALUES_NUMBER,97,d_p,ztmp);                      
#ifdef PLASMA_MOVE_CUPRINTF                  
               //cuPrintf("partdx after %e \n",partdx);
               //cuPrintf("jk-tmp1 %d %d \n",jtmp,ktmp);
#endif               
            //   return;                  
            }
            /*
/////////////////////////// particle pusher one cell ///////////////////////
            xtmp = 0;
            ytmp = yp + dy + j;
            ztmp = zp + dz + k;

            while (ytmp > l_My) ytmp -= l_My;
            while (ytmp < 0) ytmp += l_My;
            while (ztmp > l_Mz) ztmp -= l_Mz;
            while (ztmp < 0) ztmp += l_Mz;

            jtmp = ytmp;
            ktmp = ztmp;
           

            ytmp -= jtmp;
            ztmp -= ktmp;
            //if(np < 50) cuPrintf("jk-tmp1 %d %d \n",jtmp,ktmp);
            DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
               djx, djy, djz, drho);
/////////////////////// end of one cell pusher ///////////////////
*/
            if (iFullStep) {
               xtmp = 0.;
               //p->SetP(px,py,pz);
               p->f_Px = px;
               p->f_Py = py;
               p->f_Pz = pz;
            write_plasma_value(np,PLASMA_VALUES_NUMBER,92,d_p,ytmp); 
            write_plasma_value(np,PLASMA_VALUES_NUMBER,93,d_p,ztmp); 
               
               //p->SetX(xtmp,ytmp,ztmp);
               p->f_X = xtmp;
               p->f_Y = ytmp;
               p->f_Z = ztmp;
               //if(np < 50) cuPrintf("new coords %e %25.15e %25.15e \n",p->f_X,p->f_Y,p->f_Z);
               p->i_X = iLayer;
               p->i_Y = jtmp;
               p->i_Z = ktmp;
               //cuPrintf("jk-tmp %d %d np %d \n",jtmp,ktmp,np);
               
               /*
               long nnew = GetNyz(jtmp,ktmp);
               Cell &cnew = p_CellLayerC[nnew];
               p->p_Next = cnew.p_Particles;
               cnew.p_Particles = p;
               pcc.p_Particles = p_next; */
               int num_per_cell = Np/(Ny*Nz);
               int cell_num    = np/num_per_cell;
               int num_in_cell = np - cell_num*num_per_cell;
               int new_num_in_cell = num_per_cell - num_in_cell;
               int new_np = cell_num*num_per_cell + new_num_in_cell- 1;
               copyParticle(cl->particles + new_np,p);
               
               if(np < 10) 
               {
                 // cuPrintf("np %d percell %d cellnum %d incell %d new %d,newnp %d\n",np,num_per_cell,cell_num,num_in_cell,new_num_in_cell,new_np);
                  //beamParticle *cp = cl->particles + new_np;
                 // cuPrintf("new percell %d %e %e \n",new_np,cp->f_Y,cp->f_Z);
               }
            }
//            p = p_next;
              write_plasma_value(np,PLASMA_VALUES_NUMBER,99,d_p,(double)np);

}


//---Mesh::DepositCurrentsInCellSplit ---------------------------------------------->
void __device__ cuDepositCurrentsInCellSplit(
                                 beamParticle *p, int isort, int Ny,
                                 int i, int j, int k, 
                                 double Vx, double Vy, double Vz, 
                                 double x, double y, double z, 
                                 double djx, double djy, double djz, double drho,cudaLayer *cl,double *d_p,int np)
{
#ifdef PLASMA_MOVE_CUPRINTF
 //  cuPrintf("in cuDeposit \n");
#endif   
   //return;
   long ncc = j +  k*Ny;
   
   int l_sizeY = Ny;

   long npc = ncc + 1;
   long ncp = ncc + l_sizeY;
   long npp = ncp + 1;
   long nmc = ncc - 1;
   long ncm = ncc - l_sizeY;
   long nmm = ncm - 1;
   long nmp = ncp - 1;
   long npm = npc - l_sizeY;


   x = 0.;
   double ayc = 1.-y;
   double ayp = y;
   double azc = 1.-z;
   double azp = z;

   double accc = ayc*azc;
   double acpc = ayp*azc;
   double accp = ayc*azp;
   double acpp = ayp*azp;

   double weight = fabs(drho);

#ifdef PLASMA_MOVE_CUPRINTF   
 //   cuPrintf("b add %d \n",ncc);
#endif

   //return;
   write_plasma_value(np,PLASMA_VALUES_NUMBER,87,d_p,djx);  
   cuda_atomicAddP(&(cl->Jx[ncc]),djx);
   write_plasma_value(np,PLASMA_VALUES_NUMBER,88,d_p,djy);  
   write_plasma_value(np,PLASMA_VALUES_NUMBER,89,d_p,djz);  
   write_plasma_value(np,PLASMA_VALUES_NUMBER,90,d_p,drho);
   write_plasma_value(np,PLASMA_VALUES_NUMBER,91,d_p,(double)ncc);  


   cuda_atomicAddP(&(cl->Jy[ncc]),djy);
   cuda_atomicAddP(&(cl->Jz[ncc]),djz);
//   cuPrintf("rho before %e \n",cl->Rho[ncc]);
   cuda_atomicAddP(&(cl->Rho[ncc]), drho);
//    += drho;
//   cuPrintf("rho after %e \n",cl->Rho[ncc]);
}

int CUDA_WRAP_write_plasma_value(int i,int num_attr,int n,double t)
{

#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 

	//int cell_number = i*Ny + j;
	
	h_plasma_values [i*num_attr + n] = t;
	
	//cudaMemcpy((void**)d_p,num_attr*ppc_max*Ny*Nz*sizeof(double));
	
#endif	
	
	return 0;
}

//writing a value to the control array for a definite particle in a definite cell
__device__ int write_plasma_value(int i,int num_attr,int n,double *d_p,double t)
{
#ifdef CUDA_WRAP_CHECK_BEAM_VALUES_ALLOWED 

	//int cell_number = i*Ny + j;
	if(n == 0) 
	{
//	   cuPrintf("before write %e  \n",d_p [i*num_attr +  n]);
	}
	d_p [i*num_attr +  n] = t;
	if(n == 0) 
	{
//	   cuPrintf("after write  %e  \n",d_p [i*num_attr +  n]);
	}
#endif	
	return 0;
}

double CUDA_WRAP_check_plasma_values(int Np,int num_attr,int blocksize_x,int blocksize_y)
{
     return CUDA_WRAP_check_beam_values(Np,num_attr,h_plasma_values,d_plasma_values,blocksize_x,blocksize_y,"plasmaCheck.dat");
}


void cuMoveSplitParticles(int iLayer,int iSplit,cudaLayer *h_cl,cudaLayer *h_pl,int Ny,int Nz,double hx,double hy,double hz,
                                     double *djx0,double *djy0,double *djz0,double *drho0,int nsorts,int iFullStep)
{

#ifdef CUDA_WRAP_FFTW_ALLOWED
     return;
#endif

     int Np = h_pl->Np;
     static cudaLayer *d_cl,*d_pl;
     struct timeval tv1,tv2,tf1,tf2;

#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
         int err = cudaGetLastError();
         printf("begin particles %d \n",err);
#endif

#ifdef CUDA_WRAP_FFTW_ALLOWED
     return;
#endif     

     gettimeofday(&tf1,NULL);
 
     dim3 dimBlock(16,16 ,1); 
    
     int gridSize = (int)(ceil(sqrt(Np))/16.0+1.0);
    
     dim3 dimGrid(gridSize, gridSize); 
    
     static double *d_djx0,*d_djy0,*d_djz0,*d_drho0;
     static int first = 1;
     
     if(first == 1)
     {
        cudaMalloc(&d_djx0,sizeof(double)*nsorts);
        cudaMalloc(&d_djy0,sizeof(double)*nsorts);
        cudaMalloc(&d_djz0,sizeof(double)*nsorts);
        cudaMalloc(&d_drho0,sizeof(double)*nsorts);
     
        cudaMemcpy(d_djx0,djx0,sizeof(double)*nsorts,cudaMemcpyHostToDevice);
        cudaMemcpy(d_djy0,djy0,sizeof(double)*nsorts,cudaMemcpyHostToDevice);
        cudaMemcpy(d_djz0,djz0,sizeof(double)*nsorts,cudaMemcpyHostToDevice);
        cudaMemcpy(d_drho0,drho0,sizeof(double)*nsorts,cudaMemcpyHostToDevice);
     
        cudaMalloc(&d_pl,sizeof(cudaLayer));
        cudaMemcpy(d_pl,h_pl,sizeof(cudaLayer),cudaMemcpyHostToDevice);
        cudaMalloc(&d_cl,sizeof(cudaLayer));
        cudaMemcpy(d_cl,h_cl,sizeof(cudaLayer),cudaMemcpyHostToDevice);
        
        first = 0;
     }
         err = cudaGetLastError();
         printf("particles copy init %d \n",err);
     
     if(iLayer <= 119 && iSplit >= 1) 
     {
        int i120 = 0;
     }
  //   CUDA_WRAP_print_plasma_values(Np,PLASMA_VALUES_NUMBER,"before");
     
 //    cudaPrintfInit();
     gettimeofday(&tv1,NULL);
     err = cudaGetLastError();
     printf("before particles kernel %d \n",err);
     
     cuMoveSplitParticlesKernel<<<dimGrid, dimBlock>>>(iLayer,iSplit,Np,d_cl,d_pl,Ny,Nz,hx,hy,hz,
                                     d_djx0,d_djy0,d_djz0,d_drho0,iFullStep,d_plasma_values);

     cudaDeviceSynchronize();
     err = cudaGetLastError();
     printf("after particles kernel %d \n",err);
                                           
     gettimeofday(&tv2,NULL);
    // printf("particle kernel %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec));
                                     
 //    cudaPrintfDisplay(stdout, true);
 //    cudaPrintfEnd();   
 
      
     
     if(iFullStep)
     {
        h_cl->Np = Np;
     }
//     CUDA_WRAP_printLayerParticles(h_cl,"A");
//     CUDA_WRAP_print_plasma_values(Np,PLASMA_VALUES_NUMBER,"after");
     
    // cudaMemcpy(h_cl,d_cl,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
//     CUDA_WRAP_printLayerParticles(h_cl,"B");
     
     if(iLayer <= 119)
     {
        int z = 0;
     }
                                     
#ifdef CUDA_WRAP_CHECK_PLASMA_VALUES_ALLOWED
     CUDA_WRAP_check_plasma_values(Np,PLASMA_VALUES_NUMBER,dimBlock.x*dimGrid.x,dimBlock.y*dimGrid.y);
#endif
//     CUDA_WRAP_print_plasma_values(Np,PLASMA_VALUES_NUMBER,"end");
     gettimeofday(&tf2,NULL);

//     printf("particle kernel %e all %e\n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec),(tf2.tv_sec-tf1.tv_sec)+1e-6*(tf2.tv_usec-tf1.tv_usec));
}                                     


double CUDA_WRAP_print_plasma_values(int Np,int num_attr,char *where)
{
#ifndef CUDA_WRAP_PRINT_PLASMA_VALUES
        return 0.0;
#endif        

        int cell_number,wrong_particles = 0;
	double    *h_copy,frac_err,delta = 0.0,*wrong_array,*delta_array;
	int wrong_flag = 0;
	char s[100];
	FILE *f;
	
	sprintf(s,"plasmaValues_%s.dat",where);
	f = fopen(s,"wt");
	
	wrong_array = (double *)malloc(num_attr*sizeof(double));
	delta_array = (double *)malloc(num_attr*sizeof(double));
//        int width = Ny*Nz; 
//        double *h_data_in;
	
	puts("BEGIN  BEAM-RELATED VALUES sCHECK =============================================================================");
	
	//part_per_cell_max = findMaxNumberOfParticlesPerCell(mesh,i_layer,Ny,Nz,p_CellArray);
	h_copy   = (double*) malloc(num_attr*Np*sizeof(double));
	
	//GET PARTICLE DATA FROM SURFACE
	//CUDA_WRAP_get_particle_surface(partSurfOut,cuOutputArrayX,NUMBER_ATTRIBUTES*part_per_cell_max,width,h_data_in);
	cudaMemcpy(h_copy,d_plasma_values,num_attr*Np*sizeof(double),cudaMemcpyDeviceToHost);

    for(int n = 0;n < num_attr;n++)
    {
        int wpa = 0,wrong_particles = 0;;
	double fr_attr,x,cu_x;
	
	delta = 0.0;
	
        for (int i = 0;i < 50;i++)
        {
	
            cu_x = h_copy[i*num_attr + n];
			  
            fprintf(f,"%5d %5d %25.15e \n",n,i,cu_x);
	}
    }
	
	free(h_copy);
	fclose(f);
	
        return 0.0;
}

int CUDA_WRAP_print_layer(cudaLayer *d_l,int np,char *name,int Ny,int Nz)
{
   double *Ex,*Ey,*Ez,*Bx,*By,*Bz,*Jx,*Jy,*Jz,*Rho;
   beamParticle *bp;
   cudaLayer *h_l = (cudaLayer *)malloc(sizeof(cudaLayer));
//   int np = 0;
   char fname[100];
   beamParticle *p;
      
   Ex = (double *)malloc(sizeof(double)*Ny*Nz);
   Ey = (double *)malloc(sizeof(double)*Ny*Nz);
   Ez = (double *)malloc(sizeof(double)*Ny*Nz);
   Bx = (double *)malloc(sizeof(double)*Ny*Nz);
   By = (double *)malloc(sizeof(double)*Ny*Nz);
   Bz = (double *)malloc(sizeof(double)*Ny*Nz);
   Jx = (double *)malloc(sizeof(double)*Ny*Nz);
   Jy = (double *)malloc(sizeof(double)*Ny*Nz);
   Jz = (double *)malloc(sizeof(double)*Ny*Nz);
   Rho = (double *)malloc(sizeof(double)*Ny*Nz);
   
   bp = (beamParticle *)malloc(np*sizeof(beamParticle));
   
   cudaMemcpy(h_l,d_l,sizeof(cudaLayer),cudaMemcpyDeviceToHost);
   
   cudaMemcpy(bp,h_l->particles,sizeof(beamParticle)*np,cudaMemcpyDeviceToHost);
   
   sprintf(fname,"%s.dat",name);
   
   FILE *f = fopen(fname,"wt");
   
   for(int i = 0; i < h_l->Np;i++)
   {
      p = bp + i;
      fprintf(f,"%10d %5d %5d %5d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",i,p->i_X,p->i_Y,p->i_Z,p->f_X,p->f_Y,p->f_Z,p->f_Px,p->f_Py,p->f_Pz);
   }
   
   return 0;
}



