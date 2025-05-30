#include <stdlib.h>

#include <math.h>

#include "vlpl3d.h"

#include "CUDA_WRAP/cuBeamValues.h"

#include "run_control.h"

#include "para.h"

//---Mesh:: ---------------------------------------------->
void Mesh::MoveBeamParticles() 
{
   l_Processed = 0;
   f_GammaMax  = 1.;
   double ElaserPhoton = 1.2398e-4/domain()->GetWavelength();
   int isort;
   long i, j, k, n;
   double ts = Ts();
   double ts2hx = Ts()/Hx();
   double ts2hy = Ts()/Hy();
   double ts2hz = Ts()/Hz();
   double hx = Hx();
   double hy = Hy();
   double hz = Hz();

   ClearBeamCurrents();

   int nsorts = domain()->GetNsorts();
   double* djx0= new double[nsorts];
   double* djy0= new double[nsorts];
   double* djz0= new double[nsorts];
   double* drho0= new double[nsorts];

   Cell ctmp;

   double bXext = domain()->GetBxExternal()/ctmp.sf_DimFields;
   double bYext = domain()->GetByExternal()/ctmp.sf_DimFields;
   double bZext = domain()->GetBzExternal()/ctmp.sf_DimFields;
   int ifscatter = domain()->GetSpecie(0)->GetScatterFlag();
   int* iAtomTypeArray = new int[nsorts]; 

   for (isort=0; isort<nsorts; isort++)
   {
      Specie* spec = domain()->GetSpecie(isort);
      spec->GetdJ( djx0[isort], djy0[isort], djz0[isort], drho0[isort] );
      iAtomTypeArray[isort] = spec->GetAtomType(); 
   }

   FILE *beamf = fopen("hostBeamFile.dat","wt");
   int   total_np = 0;
   long xUpperLocalParallelLimit;

   if(GetRank() < GetSize()-1)
   {
      xUpperLocalParallelLimit = l_Mx + 1;
   }
   else
   {
      xUpperLocalParallelLimit = l_Mx;
   }   
   
   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
	 for (i=0; i < xUpperLocalParallelLimit; i++)
         //for (i = GetRank()*(l_Mx-1); i <= xUpperParallelLimit; i++)
         {
            long nccc = GetN(i,  j,  k);
            long ncpc = GetN(i,  j+1,k);
            long nccp = GetN(i,  j,  k+1);
            long ncpp = GetN(i,  j+1,k+1);
            long npcc = GetN(i+1,j,  k);
            long nppc = GetN(i+1,j+1,k);
            long npcp = GetN(i+1,j,  k+1);
            long nppp = GetN(i+1,j+1,k+1);

            Particle *p = NULL;
            Cell &ccc = p_CellArray[nccc];
            Cell &cpc = p_CellArray[ncpc];
            Cell &ccp = p_CellArray[nccp];
            Cell &cpp = p_CellArray[ncpp];
            Cell &pcc = p_CellArray[npcc];
            Cell &ppc = p_CellArray[nppc];
            Cell &pcp = p_CellArray[npcp];
            Cell &ppp = p_CellArray[nppp];

            if (j == l_My/2 && k == l_Mz/2 && i == l_Mx*2/3.) {
               double fdummy = 0.;
            }

            double djx = 0., djy = 0., djz = 0.;

            p = ccc.p_BeamParticles;

            if (p==NULL)
               continue;

            int NP_guess = 1000000; //guess of beam particles number;
            h_beam_values = (double *)malloc(NP_guess*BEAM_VALUES_NUMBER*sizeof(double));
	    
	    int nump = 0;
	    
            p_PrevPart = NULL;
            while(p)
            {
//	       nump = 0; 
#ifdef CUDA_WRAP_BEAM_PARTICLES_PRINT
	       fprintf(beamf," %d %d %d %d\n",i,j,k,nump++);
#endif	       
	       
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,0,h_beam_values,(double)i);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,1,h_beam_values,(double)j);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,2,h_beam_values,(double)k);
	       
               Particle *p_next = p->p_Next;
               isort = p->GetSort();
               if (isort > 0) {
                  int ttest = 0;
               }

               l_Processed++;
               double weight = p->f_Weight;
               double x  = p->f_X;
               double y  = p->f_Y;
               double z  = p->f_Z;
               double px = p->f_Px;
               double py = p->f_Py;
               double pz = p->f_Pz;
               double pxn = px;
               double pyn = py;
               double pzn = pz;
               double Vx = px / sqrt(1. + px*px + py*py + pz*pz);
               double q2m = p->f_Q2m;
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,39,h_beam_values,weight);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,40,h_beam_values,x);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,41,h_beam_values,y);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,42,h_beam_values,z);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,43,h_beam_values,px);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,44,h_beam_values,py);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,45,h_beam_values,pz);	       
	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,3,h_beam_values,ccc.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,4,h_beam_values,ccc.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,5,h_beam_values,ccc.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,6,h_beam_values,ccc.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,7,h_beam_values,ccc.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,8,h_beam_values,ccc.f_Bz);	       

               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,118,h_beam_values,cpc.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,119,h_beam_values,cpc.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,120,h_beam_values,cpc.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,121,h_beam_values,cpc.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,122,h_beam_values,cpc.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,123,h_beam_values,cpc.f_Bz);
	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,124,h_beam_values,ccp.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,125,h_beam_values,ccp.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,126,h_beam_values,ccp.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,127,h_beam_values,ccp.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,128,h_beam_values,ccp.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,129,h_beam_values,ccp.f_Bz);	       
	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,9,h_beam_values,cpp.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,10,h_beam_values,cpp.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,11,h_beam_values,cpp.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,12,h_beam_values,cpp.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,13,h_beam_values,cpp.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,14,h_beam_values,cpp.f_Bz);	       

               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,15,h_beam_values,pcc.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,16,h_beam_values,pcc.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,17,h_beam_values,pcc.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,18,h_beam_values,pcc.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,19,h_beam_values,pcc.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,20,h_beam_values,pcc.f_Bz);	       
	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,21,h_beam_values,ppc.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,22,h_beam_values,ppc.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,23,h_beam_values,ppc.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,24,h_beam_values,ppc.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,25,h_beam_values,ppc.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,26,h_beam_values,ppc.f_Bz);	       
	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,27,h_beam_values,pcp.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,28,h_beam_values,pcp.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,29,h_beam_values,pcp.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,30,h_beam_values,pcp.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,31,h_beam_values,pcp.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,32,h_beam_values,pcp.f_Bz);	       

               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,33,h_beam_values,ppp.f_Ex);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,34,h_beam_values,ppp.f_Ey);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,35,h_beam_values,ppp.f_Ez);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,36,h_beam_values,ppp.f_Bx);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,37,h_beam_values,ppp.f_By);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,38,h_beam_values,ppp.f_Bz);	       
	       

               if (x<0||x>1 || y<0||y>1 || z<0||z>1)
               {
                  domain()->out_Flog << "Wrong MoveParticles: x="
                     << x << " y=" << y << " z=" << z << "\n";
                  domain()->out_Flog.flush();
                  exit(-12);
               }

               double axc = 1.-x;
               double axp = x;
               double ayc = 1.-y;
               double ayp = y;
               double azc = 1.-z;
               double azp = z;
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,130,h_beam_values,axc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,46,h_beam_values,axp);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,47,h_beam_values,ayc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,48,h_beam_values,ayp);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,49,h_beam_values,azc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,50,h_beam_values,azp);	       
	       
               double accc = axc*ayc*azc;
               double acpc = axc*ayp*azc;
               double accp = axc*ayc*azp;
               double acpp = axc*ayp*azp;
               double apcc = axp*ayc*azc;
               double appc = axp*ayp*azc;
               double apcp = axp*ayc*azp;
               double appp = axp*ayp*azp;
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,51,h_beam_values,accc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,52,h_beam_values,acpc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,53,h_beam_values,accp);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,54,h_beam_values,acpp);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,55,h_beam_values,apcc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,56,h_beam_values,appc);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,57,h_beam_values,apcp);	       
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,58,h_beam_values,appp);	       
	
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,152,h_beam_values,px);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,153,h_beam_values,py);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,154,h_beam_values,pz);		       
	       
               double ex, ey, ez;

               double bx=0.;
               double by=0.;
               double bz=0.;

               ex =
                  accc*ccc.f_Ex + acpc*cpc.f_Ex +
                  accp*ccp.f_Ex + acpp*cpp.f_Ex +
                  apcc*pcc.f_Ex + appc*ppc.f_Ex +
                  apcp*pcp.f_Ex + appp*ppp.f_Ex;

               ey =
                  accc*ccc.f_Ey + acpc*cpc.f_Ey +
                  accp*ccp.f_Ey + acpp*cpp.f_Ey +
                  apcc*pcc.f_Ey + appc*ppc.f_Ey +
                  apcp*pcp.f_Ey + appp*ppp.f_Ey;

               ez =
                  accc*ccc.f_Ez + acpc*cpc.f_Ez +
                  accp*ccp.f_Ez + acpp*cpp.f_Ez +
                  apcc*pcc.f_Ez + appc*ppc.f_Ez +
                  apcp*pcp.f_Ez + appp*ppp.f_Ez;


               bx =
                  accc*ccc.f_Bx + acpc*cpc.f_Bx +
                  accp*ccp.f_Bx + acpp*cpp.f_Bx +
                  apcc*pcc.f_Bx + appc*ppc.f_Bx +
                  apcp*pcp.f_Bx + appp*ppp.f_Bx;

               by =
                  accc*ccc.f_By + acpc*cpc.f_By +
                  accp*ccp.f_By + acpp*cpp.f_By +
                  apcc*pcc.f_By + appc*ppc.f_By +
                  apcp*pcp.f_By + appp*ppp.f_By;

               bz =
                  accc*ccc.f_Bz + acpc*cpc.f_Bz +
                  accp*ccp.f_Bz + acpp*cpp.f_Bz +
                  apcc*pcc.f_Bz + appc*ppc.f_Bz +
                  apcp*pcp.f_Bz + appp*ppp.f_Bz;

    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,59,h_beam_values,ex);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,61,h_beam_values,ey);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,62,h_beam_values,ez);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,63,h_beam_values,bx);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,64,h_beam_values,by);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,65,h_beam_values,bz);
               //printf("%15.5e %15.5e %15.5e %15.5e %15.5e %15.5e\n",x,y,z,ex,ey,ez);
		  

//               ex = ey = ez = bx = by = bz = 0.;

               double ex1 = ex;
               double ey1 = ey;
               double ez1 = ez;


              /* if(isort > 0 && iAtomTypeArray[isort] > 0) {//if and only if ionizable ions
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

               ex *= q2m*ts/2.;
               ey *= q2m*ts/2.;
               ez *= q2m*ts/2.;*/
               /*
               double bx = axc*ccc.f_Bx + axm*mcc.f_Bx;
               double by= ayc*ccc.f_By + aym*cmc.f_By;
               double bz= azc*ccc.f_Bz + azm*ccm.f_Bz;
               */
               //		  double Bext = 1e-2*(PI*domain()->GetTs());
               //					by += Bext;

               bx += bXext;
               by += bYext;
               bz += bZext;
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,149,h_beam_values,px);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,150,h_beam_values,py);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,151,h_beam_values,pz);	
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,155,h_beam_values,ex);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,156,h_beam_values,ey);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,157,h_beam_values,ez);	
               px += ex;
               py += ey;
               pz += ez;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,66,h_beam_values,px);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,67,h_beam_values,py);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,68,h_beam_values,pz);	       
	       

               double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,69,h_beam_values,gamma);	       

               if (f_GammaMax < gamma)
                  f_GammaMax = gamma;

               double gamma_r = 1./gamma;																	 //!!!!!!
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,70,h_beam_values,gamma_r);	       

               ///////////////////////////////////////////////////////////////////////
               /////////                SCATTERING               /////////////////////
               //////////////////////////////////////////////////////////////////////

               if (isort == 0 && ifscatter) //We scatter only electrons
               {
                  double P = sqrt(px*px+py*py+pz*pz);
                  double IonDensityInCell = 0.;
                  for (int is=1; is<nsorts; is++) {
                     IonDensityInCell += ccc.f_DensArray[is];
                  };

                  double Probability = p->GetScatteringProbability(P, IonDensityInCell);

                  if (Probability > 0.)
                  {
                     double Nx = 2*(double(rand())/RAND_MAX-0.5);
                     double Ny = 2*(double(rand())/RAND_MAX-0.5);
                     double Nz = 2*(double(rand())/RAND_MAX-0.5);

                     double N = sqrt(Nx*Nx + Ny*Ny + Nz*Nz);

                     //arbitary unitvector f =|Probability|
                     double fx = (Probability*Nx)/N;
                     double fy = (Probability*Ny)/N;
                     double fz = (Probability*Nz)/N;

                     /*float f = sqrt(fx*fx + fy*fy + fz*fz); //TO TEST.

                     cout<<"collision::"<<"prob = "<<Probability<<" f = "<<f<<endl; //TO TEST
                     */
                     bx += fx;
                     by += fy;
                     bz += fz;
                  }
               }

               ////////////////////////////////////////////////////////////////////////
               //////////////////////////////////////////////////////////////////////// 
               double bx1 = bx;
               double by1 = by;
               double bz1 = bz;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,71,h_beam_values,bx1);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,72,h_beam_values,by1);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,73,h_beam_values,bz1);	       
	       

               bx = bx*gamma_r*q2m*ts/2.;
               by = by*gamma_r*q2m*ts/2.;
               bz = bz*gamma_r*q2m*ts/2.;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,74,h_beam_values,bx);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,75,h_beam_values,by);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,76,h_beam_values,bz);	       
	       

               double co = 2./(1. + (bx*bx) + (by*by) + (bz*bz));
       	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,77,h_beam_values,co);	       


               double p3x = py*bz - pz*by + px;
               double p3y = pz*bx - px*bz + py;
               double p3z = px*by - py*bx + pz;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,78,h_beam_values,p3x);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,79,h_beam_values,p3y);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,80,h_beam_values,p3z);	       
	       

               p3x *= co;
               p3y *= co;
               p3z *= co;
	       

               double px_new = p3y*bz - p3z*by;
               double py_new = p3z*bx - p3x*bz;
               double pz_new = p3x*by - p3y*bx;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,81,h_beam_values,px_new);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,82,h_beam_values,py_new);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,83,h_beam_values,pz_new);	       

               px += (ex + px_new);
               py += (ey + py_new);
               pz += (ez + pz_new);
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,84,h_beam_values,px);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,85,h_beam_values,py);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,86,h_beam_values,pz);	       
	       
               /*
               double damping = 1e-3;

               px -= damping*ts*px*gamma;
               py -= damping*ts*py*gamma;
               pz -= damping*ts*pz*gamma;
               */

               /////////////////////////////////////////////////////////////////////////////////////////////////
               ///////////////////// Synchrotron Damping ///////////////////////////////////////////////////////
               /////////////////////////////////////////////////////////////////////////////////////////////////

               double p2 = px*px + py*py + pz*pz;
               Synchrotron *pSynchrotron = domain()->GetSynchrotron();
               double eSynMin = pSynchrotron->GetSynMin();
               if ( 1.+p2 > eSynMin*eSynMin ) {

                  double dpx = px - pxn;
                  double dpy = py - pyn;
                  double dpz = pz - pzn;
                  double pdp = px*dpx + py*dpy + pz*dpz;

                  dpx -= px*(pdp/p2); // This is now the transverse momentum change
                  dpy -= py*(pdp/p2);
                  dpz -= pz*(pdp/p2);

                  double p2absrev = 1.f/sqrt(p2);
                  double nx = px*p2absrev;
                  double ny = py*p2absrev;
                  double nz = pz*p2absrev;

                  double dp2 = dpx*dpx + dpy*dpy + dpz*dpz;

                  double dpabs = sqrt(dpx*dpx + dpy*dpy + dpz*dpz);
                  double Gamma2 = 1. + p2;
                  double Gamma = sqrt(Gamma2);

                  double omega_c_omega_laser = 3.* Gamma2 * dpabs / (4.*PI*ts);

                  double Ephoton   = ElaserPhoton*omega_c_omega_laser;
                  double alfa = 1./137.;
                  double nph = 4./9. * alfa * dpabs;

                  /////////////////////////////
                  double pt2    = py*py + pz*pz;
                  double pt2_p2 = sqrt(pt2)/sqrt(p2);
                  double theta  = asin(pt2_p2);
                  double phi  = atan2(pz, py);

                  pSynchrotron->AddParticle(p, nph, theta, phi, Ephoton);

                  px -= (nph * Ephoton * nx / 5.11e5f);
                  py -= (nph * Ephoton * ny / 5.11e5f);
                  pz -= (nph * Ephoton * nz / 5.11e5f);
               };

               /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
               /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

               gamma = 1./sqrt(1. + px*px + py*py + pz*pz);
               Vx = px*gamma;
               double Vy = py*gamma;
               double Vz = pz*gamma;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,87,h_beam_values,gamma);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,88,h_beam_values,Vx);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,89,h_beam_values,Vy);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,90,h_beam_values,Vz);	       

               p->SetP(px,py,pz);

               double polarity = p->GetSpecie()->GetPolarity();

               djx = weight*polarity*Vx;
               djy = weight*polarity*Vy;
               djz = weight*polarity*Vz;
               double drho = weight*polarity;
               CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,91,h_beam_values,djx);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,92,h_beam_values,djy);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,93,h_beam_values,djz);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,144,h_beam_values,drho);	       

	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,146,h_beam_values,gamma);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,145,h_beam_values,px);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,147,h_beam_values,hx);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,131,h_beam_values,ts);
	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,148,h_beam_values,((px*gamma-1.)*ts/hx));
	       
               double dx = (px*gamma-1.)*ts/hx;
               double dy = py*gamma*ts/hy;
               double dz = pz*gamma*ts/hz;
               double xtmp = x + dx;
               double ytmp = y + dy;
               double ztmp = z + dz;
	       
	      
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,94,h_beam_values,dx);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,95,h_beam_values,dy);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,96,h_beam_values,dz);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,97,h_beam_values,xtmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,98,h_beam_values,ytmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,99,h_beam_values,ztmp);	       
	       

               double full = 1.;

               int i_jump = xtmp;
               int j_jump = ytmp;
               int k_jump = ztmp;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,100,h_beam_values,(double)i_jump);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,101,h_beam_values,(double)j_jump);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,102,h_beam_values,(double)k_jump);	       
	       
	       
               if (xtmp < 0.) {
                  i_jump--;
               };
               if (ytmp < 0.) {
                  j_jump--;
               };
               if (ztmp < 0.) {
                  k_jump--;
               };
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,103,h_beam_values,(double)i_jump);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,104,h_beam_values,(double)j_jump);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,105,h_beam_values,(double)k_jump);	       
	       
               int itmp = i + i_jump;
               int jtmp = j + j_jump;
               int ktmp = k + k_jump;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,106,h_beam_values,(double)itmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,107,h_beam_values,(double)jtmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,108,h_beam_values,(double)ktmp);	       
	       
               if (itmp < 0) itmp = -1;
               if (itmp > l_Mx-1) itmp = l_Mx;
               if (jtmp < 0) jtmp = -1;
               if (jtmp > l_My-1) jtmp = l_My;
               if (ktmp < 0) ktmp = -1;
               if (ktmp > l_Mz-1) ktmp = l_Mz;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,109,h_beam_values,(double)itmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,110,h_beam_values,(double)jtmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,111,h_beam_values,(double)ktmp);	       

               xtmp -= i_jump;
               ytmp -= j_jump;
               ztmp -= k_jump;
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,112,h_beam_values,xtmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,113,h_beam_values,ytmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,114,h_beam_values,ztmp);	       
	       

               while (xtmp < 0.) {
		 xtmp += 1.;
	       };
               while (xtmp > 1.) {
		 xtmp -= 1.;
	       };
               while (ytmp < 0.) {
		 ytmp += 1.;
	       };
               while (ytmp > 1.) {
		 ytmp -= 1.;
	       };
               while (ztmp < 0.) {
		 ztmp += 1.;
	       };
               while (ztmp > 1.) {
		 ztmp -= 1.;
	       };
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,115,h_beam_values,xtmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,116,h_beam_values,ytmp);	       
    	       CUDA_WRAP_write_beam_value(total_np,BEAM_VALUES_NUMBER,117,h_beam_values,ztmp);	       

               p->SetX(xtmp,ytmp,ztmp);

               int ntmp = GetN(itmp,jtmp,ktmp);
               Cell &ctmp = p_CellArray[ntmp];
               p->l_Cell = ntmp;

               MoveBeamInCell(p, isort, 
                  itmp, jtmp, ktmp, 
                  Vx, Vy, Vz, 
                  xtmp, ytmp, ztmp, 
                  djx, djy, djz, drho,total_np);

               ccc.p_BeamParticles = p = p_next;
	       
	       total_np++;
            }
         }
      }
   }
   fclose(beamf);
   printf("Total number of particles on host %d \n",total_np);
   long totalNe = domain()->GetSpecie(0)->GetNp();
   //   cout << "We have " << totalNe << " electrons \n";


   for (k=-1; k<l_Mz+1; k++)
   {
      for (j=-1; j<l_My+1; j++)
      {
         for (i=-1; i<l_Mx+1; i++)
         {
            Cell &c = GetCell(i,  j,  k);
            Particle *p = c.p_BeamHook;
            while (p) {
               Particle *pnext = p->p_Next;
               p->p_Next = c.p_BeamParticles;
               c.p_BeamParticles = p;
               p->l_Cell = c.l_N;
               c.p_BeamHook = p = pnext;
            };
         }
      }
   }

   delete[] djx0;
   delete[] djy0;
   delete[] djz0;
   delete[] drho0;
   delete[] iAtomTypeArray;
}

//---Mesh::MoveBeamInCell ---------------------------------------------->
void Mesh::MoveBeamInCell(
                      Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho,int np)
{

   Cell &ccc = GetCell(i,j,k);
   Cell &cpc = GetCell(i,j+1,k);
   Cell &ccp = GetCell(i,j,k+1);
   Cell &cpp = GetCell(i,j+1,k+1);
   Cell &pcc = GetCell(i+1,j,k);
   Cell &ppc = GetCell(i+1,j+1,k);
   Cell &pcp = GetCell(i+1,j,k+1);
   Cell &ppp = GetCell(i+1,j+1,k+1);

   double axp = x;
   double axc = 1. - axp;
   double ayp = y;
   double ayc = 1. - ayp;
   double azp = z;
   double azc = 1. - azp;
#ifdef CUDA_WRAP_PARALLEL_DEBUG
   printf("%3d %3d %3d rank %2d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e \n",i,j,k,GetRank(),x,y,z,axc,ayc,azc);
#endif   
   
   
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,132,h_beam_values,x);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,133,h_beam_values,y);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,134,h_beam_values,z);
   
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,135,h_beam_values,axp);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,136,h_beam_values,axc);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,137,h_beam_values,ayc);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,138,h_beam_values,ayp);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,139,h_beam_values,azc);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,140,h_beam_values,azp);
   
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,141,h_beam_values,(double)i);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,142,h_beam_values,(double)j);
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,143,h_beam_values,(double)k);   

   double weight = p->GetWeight();

   ccc.f_JxBeam  +=  djx*axc*ayc*azc;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,160,h_beam_values,djx*axc*ayc*azc);
   ccc.f_JyBeam  +=  djy*axc*ayc*azc;
   ccc.f_JzBeam  +=  djz*axc*ayc*azc;
   ccc.f_RhoBeam += drho*axc*ayc*azc;
   ccc.f_DensArray[isort] += weight*axc*ayc*azc;
#ifdef CUDA_WRAP_PARALLEL_DEBUG   
   printf("%3d %3d %3d rank %2d ccc %25.15e %25.15e \n",i,j,k,GetRank(),ccc.f_JxBeam,ccc.f_RhoBeam);
#endif   

   ccp.f_JxBeam  +=  djx*axc*ayc*azp;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,161,h_beam_values,djx*axc*ayc*azp);
   ccp.f_JyBeam  +=  djy*axc*ayc*azp;
   ccp.f_JzBeam  +=  djz*axc*ayc*azp;
   ccp.f_RhoBeam += drho*axc*ayc*azp;
   ccp.f_DensArray[isort] += weight*axc*ayc*azp;
#ifdef CUDA_WRAP_PARALLEL_DEBUG   
   printf("%3d %3d %3d rank %2d ccp %25.15e %25.15e \n",i,j,k,GetRank(),ccp.f_JxBeam,ccp.f_RhoBeam);
#endif   

   cpc.f_JxBeam  +=  djx*axc*ayp*azc;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,162,h_beam_values,djx*axc*ayp*azc);
   cpc.f_JyBeam  +=  djy*axc*ayp*azc;
   cpc.f_JzBeam  +=  djz*axc*ayp*azc;
   cpc.f_RhoBeam += drho*axc*ayp*azc;
   cpc.f_DensArray[isort] += weight*axc*ayp*azc;

   cpp.f_JxBeam  +=  djx*axc*ayp*azp;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,163,h_beam_values,djx*axc*ayp*azp);
   cpp.f_JyBeam  +=  djy*axc*ayp*azp;
   cpp.f_JzBeam  +=  djz*axc*ayp*azp;
   cpp.f_RhoBeam += drho*axc*ayp*azp;
   cpp.f_DensArray[isort] += weight*axc*ayp*azp;

   pcc.f_JxBeam  +=  djx*axp*ayc*azc;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,164,h_beam_values,djx*axp*ayc*azc);
   pcc.f_JyBeam  +=  djy*axp*ayc*azc;
   pcc.f_JzBeam  +=  djz*axp*ayc*azc;
   pcc.f_RhoBeam += drho*axp*ayc*azc;
   pcc.f_DensArray[isort] += weight*axp*ayc*azc;

   pcp.f_JxBeam  +=  djx*axp*ayc*azp;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,165,h_beam_values,djx*axp*ayc*azp);
   pcp.f_JyBeam  +=  djy*axp*ayc*azp;
   pcp.f_JzBeam  +=  djz*axp*ayc*azp;
   pcp.f_RhoBeam += drho*axp*ayc*azp;
   pcp.f_DensArray[isort] += weight*axp*ayc*azp;

   ppc.f_JxBeam  +=  djx*axp*ayp*azc;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,166,h_beam_values,djx*axp*ayp*azc);
   ppc.f_JyBeam  +=  djy*axp*ayp*azc;
   ppc.f_JzBeam  +=  djz*axp*ayp*azc;
   ppc.f_RhoBeam += drho*axp*ayp*azc;
   ppc.f_DensArray[isort] += weight*axp*ayp*azc;

   ppp.f_JxBeam  +=  djx*axp*ayp*azp;
   CUDA_WRAP_write_beam_value(np,BEAM_VALUES_NUMBER,167,h_beam_values,djx*axp*ayp*azp);
   ppp.f_JyBeam  +=  djy*axp*ayp*azp;
   ppp.f_JzBeam  +=  djz*axp*ayp*azp;
   ppp.f_RhoBeam += drho*axp*ayp*azp;
   ppp.f_DensArray[isort] += weight*axp*ayp*azp;
#ifdef CUDA_WRAP_PARALLEL_DEBUG	   
   printf("beam dens comp %5d %5d %5d %e sort %d rank %d \n",i,j,k,ccc.f_RhoBeam,isort,GetRank());
#endif   

   p->p_Next = ccc.p_BeamHook;
   ccc.p_BeamHook = p;
}
