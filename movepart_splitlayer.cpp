#include "CUDA_WRAP/beam_copy.h"
#include "CUDA_WRAP/copy_layer.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include <complex>
#include <math.h>
#include "vlpl3d.h"
#include <sys/time.h>
#include "run_control.h"

#include "CUDA_WRAP/cuBeam.h"
#include "CUDA_WRAP/cuLayers.h"
#include "CUDA_WRAP/paraLayers.h"
#include "CUDA_WRAP/paraCPUlayers.h"

#include "CUDA_WRAP/plasma_particles.h"
#include "CUDA_WRAP/diagnostic_print.h"
#include "CUDA_WRAP/cuParticles.h"
#include "CUDA_WRAP/cuDiagnose.h"

#include "para.h"
#include "control_point.h"

static double maxVx;

//---Mesh:: ---------------------------------------------->
void Mesh::MoveAllSplitLayers() 
{
    cudaLayer *h_P,*h_C,*h_left,*h_right;
    struct timeval tv1,tv2,tvl1,tvl2,tvs1,tvs2,tvs15,tvs12,tv15;
    cudaLayer *host_send_layer,*h_cl,*h_pl;

    printf("in MoveAllSplitLayers \n ");

//#ifndef PARALLEL_ONLY
    gettimeofday(&tv1,NULL);
    //printf("begin moveAll %d \n",GetRank());
//    cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-A");
//    CUDA_WRAP_printBeamDensity3D(this,GetControlDomain()->p_Cntrl->GetNstep(),"RECEIVED-A");
   
   int NxSplit = GetNxSplit();
   for (long n=0; n<l_sizeXYZ; n++) {
      for (int i=0; i<CURR_DIM; i++) {
         p_CellArray[n].f_Currents[i] = 0.;
      }
   }
   printf("before set %d \n",l_Mx);
   //Set_l_Mx(&l_Mx);
   printf("after set %d \n",l_Mx);
   CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"I");
   
   
//         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 6");

//    cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-B");
      printf("rank %d before recv layer -2 \n",GetRank());
   printf("MoveAll rank %d l_Mx %d \n",GetRank(),l_Mx);
  int err0 = cudaGetLastError();
   cudaLayer *h_basic_layer,**h_layers = (cudaLayer**)malloc((l_Mx+1)*sizeof(cudaLayer*));
//   int err01 = cudaGetLastError();   
   CUDA_WRAP_alloc3Dfields(l_Mx,l_My,l_Mz);
  int err02 = cudaGetLastError();
   CUDA_WRAP_alloc3Dcurrents(l_Mx,l_My,l_Mz);
  int err03 = cudaGetLastError();
   CUDA_WRAP_copy3Dfields(this,p_CellArray,l_Mx,l_My,l_Mz);
  int err1 = cudaGetLastError();
  SeedFrontParticles();
  printf("rank %d before recv layer -1 \n",GetRank());
  //exit(0);

   CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"II");
   
  int err1_5 = cudaGetLastError();
   int Np = CUDA_WRAP_copyLayerToDevice(this,p_CellArray,l_Mx,l_My,l_Mz,&h_basic_layer);
   printf("rank %d Np %d \n",GetRank(),Np);
   CUDA_WRAP_printLayerParticles(h_basic_layer,"INIT");
//    cuLayerPrintCentre(h_basic_layer,-1313,this,p_CellArray,"INIT");

//    cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-C");
   
   CUDA_WRAP_allocLayerOnHost(&h_C,l_My,l_Mz,Np);
   CUDA_WRAP_printLayerParticles(h_C,"INIT-C ");
   CUDA_WRAP_allocLayerOnHost(&h_P,l_My,l_Mz,Np);
       cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 5"); 

   CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"III");
   cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-D");     
//   int err2 = cudaGetLastError();
   maxVx = 0.;
   printf("before copy loop %d \n",l_Mx);
   //exit(0);
   for (int iLayer=l_Mx; iLayer>-1; iLayer--) 
   {
#ifdef  CUDA_WRAP_PARALLEL_DEBUG   
      printf("Layer begin copy %d *********************************************************************************************************** \n",iLayer); 
#endif
      
      int seq_iLayer = (l_Mx - 1)*GetRank() + iLayer;
       CUDA_WRAP_allocLayerOnHost(&h_layers[iLayer],l_My,l_Mz,Np);
       printf("h_layers[iLayer]->Rho %p \n ",(h_layers[iLayer])->Rho);
      CUDA_WRAP_copyLayerFrom3D(iLayer,l_My,l_Mz,Np,&h_layers[iLayer]); 
      cudaLayer *t = h_layers[iLayer];
      printf("h_layers[l_Mx]-B %d %d %d \n",h_layers[l_Mx]->Ny,h_layers[l_Mx]->Nz,h_layers[l_Mx]->Np);
      //exit(0);
      
      if(GetRank() == 1)
      {
//         //printf("Layer %d seq %d ===========================================================================================================",iLayer,seq_iLayer); 
//         CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,t->JxBeam,"JxBeam");
      }
   } 
   //exit(0);

   double hx = GetControlDomain()->GetHx(),
          hy = GetControlDomain()->GetHy(),
          hz = GetControlDomain()->GetHx();


   int nstep = this->domain()->p_Cntrl->GetNstep();
  // CUDA_WRAP_diagnose_host_layers(h_layers);

//    CUDA_WRAP_diagnose_host_layers(l_Mx,l_My,l_Mz,
//                                    hx,hy,hz,h_layers,nstep);
//    CUDA_WRAP_diagnose_host_layers(l_Mx,l_My,l_Mz,
  //                                 hx,hy,hz,h_layers,nstep);
   
   //CUDA_WRAP_copyLayerParticles(h_layers[l_Mx],h_basic_layer);
  // CUDA_WR        AP_printLayerParticles(h_layers[l_Mx],"LAST");
//#endif   
   printf("rank %d before recv layer l_Mx %d  Ny %d  Nz %d Np %d \n ",GetRank(),l_Mx,(h_layers[l_Mx])->Ny,(h_layers[l_Mx])->Nz,(h_layers[l_Mx])->Np);
   //exit(0);
   cudaLayer *recv_layer;
   cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 4");
   
   cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-0"); 
   
   CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"IV");
   printf("h_layers[l_Mx]-A %d %d %d \n",h_layers[l_Mx]->Ny,h_layers[l_Mx]->Nz,h_layers[l_Mx]->Np);
//#ifdef CPU_COMPUTING
   CUDA_WRAP_createNewLayer(&recv_layer,h_layers[l_Mx]); 
//#else
//   recv_layer = (cudaLayer *)malloc(sizeof(cudaLayer));
//#endif
   CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"before recevied rank");
  // printf("before received rank %d \n",GetRank());
   cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-1"); 
   
   ReceiveLayer(recv_layer,l_My,l_Mz,Np);
     cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 3"); 
   
   cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED-2"); 
   printf("Layer RECEIVED from above rank %d  ===================================================== \n",GetRank());
   //exit(0);
   cuLayerPrintCentre(h_layers[l_Mx],l_Mx,this,p_CellArray,"before received");
   //CUDA_WRAP_printLayerParticles(h_layers[l_Mx],"RECEIVED");
   //printf("END RECEIVED from above rank %d =======================================================\n",GetRank());
   
   //printf("rank %d after recv layer l_Mx %d  Ny %d  Nz %d Np %d \n ",GetRank(),l_Mx,(h_layers[l_Mx])->Ny,(h_layers[l_Mx])->Nz,(h_layers[l_Mx])->Np);
   cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"RECEIVED"); 
   if(GetRank() < GetSize() - 1)
   {
#ifdef CUDA_WRAP_FFTW_ALLOWED	    
       CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"before recevied");

       CUDA_WRAP_setLayerToMesh(this,p_CellArray,l_Mx,l_My,l_Mz,recv_layer,1);
       printf("in getLayerParticles before err \n");
       CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,l_Mx,l_My,l_Mz,"after recevied");
       printf("in getLayerParticles after err \n");
      cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"INIT host-P4");
       
       
#else       
//      CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_layers[l_Mx],h_basic_layer);
      CUDA_WRAP_copyLayerParticles(h_basic_layer,h_layers[l_Mx]);
      CUDA_WRAP_printLayerParticles(h_basic_layer,"copied to basic");
#endif      
   }
   printf("finished recv Layer %d \n",GetRank());

   
//   if(GetRank() == 0) exit(0);
     cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 2");
      cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"INIT host-P3");
     
        cuLayerPrintCentre(h_C,l_Mx-2,this,p_CellArray,"host-P3"); 
#ifndef PARALLEL_ONLY
 //  printf("before main loop %d rank %d\n",l_Mx,GetRank());
   
   for (int iLayer=l_Mx-1; iLayer>-1; iLayer--) {
       gettimeofday(&tvl1,NULL); 
     
//      CUDA_WRAP_copyLayerFrom3D(iLayer,l_My,l_Mz,Np,&h_layers[iLayer]); 
#ifdef CUDA_WRAP_FFTW_ALLOWED	      
      for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
         for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
            long np = GetN(iLayer+1,j,k);
            long nc = np-1;
            long nYZ = GetNyz(j,k);
            p_CellLayerP[nYZ] = p_CellArray[np];
	    
            Cell& ccc = p_CellArray[np];
	    Cell &ccc_c =  p_CellLayerC[nYZ];
	    Cell &ccc_p =  p_CellLayerC[nYZ];
	    double *fds = ccc_p.GetFields();
	    
	    ccc_c.SetFields(fds);
	    

	    
         }
      }
#endif
     cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"assign"); 

     CUDA_WRAP_printBeamDensity3D(this,GetControlDomain()->p_Cntrl->GetNstep(),"Assign");
     
     cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"before 1"); 
     cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before 1"); 
     cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"INIT host-P2");
     printf("rank %d init loop\n",GetRank());
     //exit(0);
      
      CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"INIT host",this->GetControlDomain()->p_Cntrl->GetNstep());
//      //printf("rank %d after printlist\n",GetRank());
//      if(GetRank() == 0) exit(0);

      cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"INIT host-P1");

      CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_P,h_layers[iLayer+1]);
      cuLayerPrintCentre(h_layers[iLayer+1],iLayer+1,this,p_CellArray,"INIT host");
      cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"INIT host");
      cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-INIT host");
      CUDA_WRAP_printLayerParticles(h_P,"afgter copy to P");
     cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"afgter copy to P"); 

      
      getLayersPC(&h_cl,&h_pl);
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
      
      gettimeofday(&tv15,NULL);
      printf("rank %d before split\n",GetRank());
//      exit(0);
      
      cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-before loop");
      cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"before loop"); 
     
      for (int iSplit=0; iSplit<NxSplit; iSplit++) {
	 gettimeofday(&tvs1,NULL); 
#ifdef CUDA_WRAP_FFTW_ALLOWED	 
         for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
            for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
               long np = GetN(iLayer+1,j,k);
               long nc = np-1;
               long nYZ = GetNyz(j,k);
               double fRhoBeamC = p_CellArray[nc].f_RhoBeam;
               double fRhoBeamP = p_CellArray[np].f_RhoBeam;
               double fJxBeamC = p_CellArray[nc].f_JxBeam;
               double fJxBeamP = p_CellArray[np].f_JxBeam;
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	       
	        printf("begin loop %5d %5d %15.5e %15.5e %15.5e %15.5e\n",j,k,fJxBeamC,fJxBeamP,(fJxBeamC*(NxSplit-iSplit) + fJxBeamP*iSplit)/NxSplit,
		                                                                               (fJxBeamC*(NxSplit-iSplit-1.) + fJxBeamP*(iSplit+1.))/NxSplit);
#endif		
            }
         }
#endif  	 
	
//	 int err1 = cudaGetLastError();
	 
	// CUDA_WRAP_allocLayerOnHost(&h_left,l_My,l_Mz,Np);
	// CUDA_WRAP_allocLayerOnHost(&h_right,l_My,l_Mz,Np);
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 1");
//#ifndef CUDA_WRAP_FFTW_ALLOWED
	 if(iLayer<= 118 )
	 {
	    CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_P->Ex,"R");
	 }
	 //CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_left,h_layers[iLayer]);
	 h_left  = h_layers[iLayer];
	 h_right = h_layers[iLayer+1];
//#endif	 
	 //CUDA_WRAP_copyLayerParticles(h_right,h_layers[iLayer+1]);
         //CUDA_WRAP_printLayerParticles(h_layers[l_Mx],"right");

         //printf("rank %d before basic \n",GetRank());
//         if(GetRank() == 0) exit(0);
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 2");
	 
	 cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"loop 2"); 
	 printf("Loop 2\n");
	 //exit(0);
	 
	 if((iLayer == l_Mx-1) && (iSplit == 0))
	 {  
#ifndef CUDA_WRAP_FFTW_ALLOWED		   
	    CUDA_WRAP_printLayerParticles(h_basic_layer,"basic");
            CUDA_WRAP_copyLayerParticles(h_P,h_basic_layer);
            CUDA_WRAP_printLayerParticles(h_P,"P");	 
#endif	    
	 }
         CUDA_WRAP_printLayerParticles(h_P,"after copy from basic ");

#ifdef CUDA_WRAP_FFTW_ALLOWED	 
         for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
            for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
               long np = GetN(iLayer+1,j,k);
               long nc = np-1;
               long nYZ = GetNyz(j,k);
               double fRhoBeamC = p_CellArray[nc].f_RhoBeam;
               double fRhoBeamP = p_CellArray[np].f_RhoBeam;
               double fJxBeamC = p_CellArray[nc].f_JxBeam;
               double fJxBeamP = p_CellArray[np].f_JxBeam;
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	       
	       printf("афтер цопы %5d %5d %15.5e %15.5e %15.5e %15.5e\n",j,k,fJxBeamC,fJxBeamP,(fJxBeamC*(NxSplit-iSplit) + fJxBeamP*iSplit)/NxSplit,
		                                                                               (fJxBeamC*(NxSplit-iSplit-1.) + fJxBeamP*(iSplit+1.))/NxSplit);
#endif	       
            }
         }
#endif  	 
	 cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"афтер цопы"); 
	 //cuLayerPrintCentre(h_layers[iLayer],iLayer);
         //printf("rank %d after basic \n",GetRank());
//         if(GetRank() == 0) exit(0);

	 CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"basic",this->GetControlDomain()->p_Cntrl->GetNstep());
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 3");
	 cuLayerPrintCentre(h_P,-1,this,p_CellLayerP,"P-loop 3");
	 
	 //CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_right,h_layers[iLayer+1]);
//#ifndef CUDA_WRAP_FFTW_ALLOWED	 
	 if(iLayer<= 118) CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_left->JxBeam,"R");
	 
	 
	 if(iLayer == 118 && iSplit >= 0)
	 {
	     CUDA_WRAP_check_hidden_currents3D(this,p_CellArray,iLayer,l_My,l_Mz,h_left->JxBeam,"JxBeam");
	     CUDA_WRAP_check_hidden_currents3D(this,p_CellArray,iLayer+1,l_My,l_Mz,h_right->JxBeam,"JxBeam");
	 }
	
	 
	 CUDA_WRAP_interpolateLayers(iSplit,NxSplit,l_My,l_Mz,h_C,h_P,h_left,h_right);
	 if(iLayer<= 118) CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_C->JxBeam,"C");
//	 int err_int = cudaGetLastError();
	 double *t;
// 	 cudaMalloc((void **)&t,l_My*l_Mz*sizeof(double));
// 	 int errc = cudaMemcpy(t,h_P->JxBeam,l_My*l_Mz*sizeof(double),cudaMemcpyDeviceToDevice);
	 printf("before call  to setLayersPC h_C %p %d h_P %p %d \n ",h_C,h_C->Np,h_P,h_P->Np);
	// exit(0);
	 setLayersPC(h_C,h_P);
//#endif	 
	 
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 33");
	 cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"loop 3"); 
	  
         printf("rank %d after setLayers \n",GetRank());
	 //exit(0);
//         if(GetRank() == 0) exit(0);
	  
#ifdef CUDA_WRAP_FFTW_ALLOWED	 
         for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
            for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
               long np = GetN(iLayer+1,j,k);
               long nc = np-1;
               long nYZ = GetNyz(j,k);
               double fRhoBeamC = p_CellArray[nc].f_RhoBeam;
               double fRhoBeamP = p_CellArray[np].f_RhoBeam;
               double fJxBeamC = p_CellArray[nc].f_JxBeam;
               double fJxBeamP = p_CellArray[np].f_JxBeam;
               p_CellLayerC[nYZ].f_RhoBeam = (fRhoBeamC*(NxSplit-iSplit) + fRhoBeamP*iSplit)/NxSplit;
               p_CellLayerP[nYZ].f_RhoBeam = (fRhoBeamC*(NxSplit-iSplit-1.) + fRhoBeamP*(iSplit+1.))/NxSplit;
               p_CellLayerC[nYZ].f_JxBeam = (fJxBeamC*(NxSplit-iSplit) + fJxBeamP*iSplit)/NxSplit;
               p_CellLayerP[nYZ].f_JxBeam = (fJxBeamC*(NxSplit-iSplit-1.) + fJxBeamP*(iSplit+1.))/NxSplit;
#ifdef CUDA_WRAP_LOOP_PARALLEL_DEBUG	       
	       printf("inter %5d %5d %15.5e %15.5e %15.5e %15.5e\n",j,k,fJxBeamC,fJxBeamP,(fJxBeamC*(NxSplit-iSplit) + fJxBeamP*iSplit)/NxSplit,
		                                                                               (fJxBeamC*(NxSplit-iSplit-1.) + fJxBeamP*(iSplit+1.))/NxSplit);
#endif	       
            }
         }
#endif         
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 4");

         gettimeofday(&tvs15,NULL); 
         if(iLayer <= 118 && iSplit >= 0)
	 {
	    int z = 0;  
	//    CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_C->JxBeam,"JxBeam");
	//    CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerP,h_P->JxBeam,"JxBeam");
	//    CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_C->RhoBeam,"RhoBeam");
	//    CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerP,h_P->RhoBeam,"RhoBeam");
	 }

         //printf("rank %d before move \n",GetRank());
//         if(GetRank() == 0) exit(0);


         CUDA_WRAP_printLayerParticles(h_P,"P0");
	 CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host P0",this->GetControlDomain()->p_Cntrl->GetNstep());
	 CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
         //printf("rank %d right before move \n",GetRank());
         cuLayerPrintCentre(h_C,-1,this,p_CellLayerC,"C-loop 5");

         if((GetRank() < GetSize() - 1) && (iLayer == l_Mx -1))
         {
#ifndef CUDA_WRAP_FFTW_ALLOWED		   
            CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_C,h_P);
            CUDA_WRAP_copyLayerParticles(h_C,h_P);
#else
            for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
                for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
                    long nYZ = GetNyz(j,k);
	            Cell &ccc_c =  p_CellLayerC[nYZ];
	            Cell &ccc_p =  p_CellLayerP[nYZ];
		    //p_CellLayerC[nYZ] = p_CellLayerP[nYZ];
		    
	            double fds[20];
		    //printf("%15.5e %15.5e %15.5e %15.5e %15.5e %15.5e \n",fds[0],fds[1],fds[2],fds[3],fds[4],fds[5]);
                    fds[0] = ccc_p.GetEx();
                    fds[1] = ccc_p.GetEy();
                    fds[2] = ccc_p.GetEz();
                    fds[3] = ccc_p.GetBx();
                    fds[4] = ccc_p.GetBy();
                    fds[5] = ccc_p.GetBz();
                    fds[6] = ccc_p.GetJx();
                    fds[7] = ccc_p.GetJy();
                    fds[8] = ccc_p.GetJz();
		    fds[9] = ccc_p.GetDens(-1);
		    
		    // printf("%3d %3d %15.5e %15.5e %15.5e %15.5e %15.5e %15.5e \n",j,k,fds[0],fds[1],fds[2],fds[3],fds[4],fds[5]);
	    
	            ccc_c.SetAll(fds);
                 }
            }	    
#endif
         }

         cuLayerPrintCentre(h_P,-10,this,p_CellLayerP,"P before MoveSplitLayer");
         cuLayerPrintCentre(h_C,-11,this,p_CellLayerC,"C before MoveSplitLayer");
         
         
//         cuMoveSplitParticles(iLayer,iSplit);
         if(iLayer % 10 == 0)
         {
             printf("MoveSplitLayer %5d %03d +++++++++++++++++++++++\n",
        		        iLayer,iSplit);
         }
         MoveSplitLayer(iLayer,iSplit);
         cuLayerPrintCentre(h_P,-100,this,p_CellLayerP,"P after MoveSplitLayer");
         cuLayerPrintCentre(h_C,-110,this,p_CellLayerC,"C after MoveSplitLayer");
         //printf("rank %d after move \n",GetRank());

	  CUDA_WRAP_printLayerParticles(h_P,"after");
	  CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host after",this->GetControlDomain()->p_Cntrl->GetNstep());
        // if(iLayer<= 118) CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_C->Ex,"ex:after move");

         //printf("rank %d after move \n",GetRank());
    //     if(GetRank() == 0) exit(0);
         
#ifdef CUDA_WRAP_FFTW_ALLOWED	 
         for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
            for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
               long nYZ = GetNyz(j,k);
               p_CellLayerP[nYZ] = p_CellLayerC[nYZ];
            }
         }
#endif         
         if(iLayer<= 118) 
         {
            CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
         }
         CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_P,h_C);
         if(iLayer<= 118) 
         {
            CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
         }	 
	 CUDA_WRAP_printLayerParticles(h_P,"C");
#ifndef CUDA_WRAP_FFTW_ALLOWED		 
	 CUDA_WRAP_copyLayerParticles(h_P,h_C);
 
	 CUDA_WRAP_printLayerParticles(h_P,"P");
	 CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host P");
	 
	 if(iLayer<= 118)
	 {
	   CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_C->Ex,"C_ex:");
	   CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_P->Ex,"P_ex:");
	 }
#endif		 
	 gettimeofday(&tvs2,NULL);
        /* printf("rank %d split %e 15 %e 12 %e\n",GetRank(),
                                    (tvs2.tv_sec - tvs1.tv_sec)+1e-6*(tvs2.tv_usec - tvs1.tv_usec),(tvs15.tv_sec - tvs1.tv_sec)+1e-6*(tvs15.tv_usec - tvs1.tv_usec),
 	                            (tvs12.tv_sec - tvs1.tv_sec)+1e-6*(tvs12.tv_usec - tvs1.tv_usec)  
	);*/
      //printf("rank %d split %d \n",GetRank(),iSplit);
//      if(GetRank() == 0) exit(0);

      }
      //printf("rank %d end split \n",GetRank());
//      if(GetRank() == 0) exit(0);

      
#ifdef CUDA_WRAP_FFTW_ALLOWED
      
      for (int k=-l_dMz; k<l_Mz+l_dMz-1; k++) {
         for (int j=-l_dMy; j<l_My+l_dMy-1; j++) {
            long nc = GetN(iLayer,j,k);
            long nYZ = GetNyz(j,k);
            p_CellArray[nc] = p_CellLayerP[nYZ];
         }
      }
//      if(iLayer == 1)
//      {
//         CUDA_WRAP_getLayerFromMesh(this,p_CellArray,1,l_My,l_Mz,&host_send_layer,iLayer);
//	       printf("out 1st pa rticle %e Np %d\n",host_send_layer->particles[0].f_Y,host_send_layer->Np);
//      }


#endif      
      //printf("before fields copy l_Mx rank %d %d %d \n",GetRank(),l_Mx,iLayer);
#ifndef CUDA_WRAP_FFTW_ALLOWED	      
      CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_P->Ey,"before fields copy");
     

      CUDA_WRAP_copyLayerFields(iLayer,l_Mx,l_My,l_Mz,h_P->Ex,h_P->Ey,h_P->Ez,h_P->Bx,h_P->By,h_P->Bz);
      
      if(GetRank() == 1) CUDA_DEBUG_print3DmatrixLayer(d_Ey3D,iLayer,l_My,l_Mz,"after copyLayer");
#endif 
      
#ifdef CUDA_WRAP_TEST_TRANSVERSE_FIELD      
      CUDA_WRAP_writeMatrixFromDevice(l_My,l_Mz,Hy(),Hz(),h_P->Ey,iLayer,"Ey");
#endif      

#ifndef CUDA_WRAP_FFTW_ALLOWED      
      if(iLayer<= 118) 
      {
         CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
	 CUDA_WRAP_check_hidden_currents3D(this,p_CellArray,iLayer,l_My,l_Mz,h_P->JxBeam,"JxBeam");
      }
      CUDA_WRAP_copyLayerDeviceToDevice(l_My,l_Mz,Np,h_layers[iLayer],h_P);
      
      CUDA_WRAP_copyLayerParticles(h_layers[iLayer],h_P);
      
      CUDA_WRAP_printLayerParticles(h_layers[iLayer],"layer particles");
#endif      
//      cuLayerPrintCentre(h_layers[iLayer],iLayer,this,p_CellArray);

      cuLayerPrintCentre(h_layers[iLayer],iLayer,this,p_CellArray,"layer particles");

#ifndef CUDA_WRAP_FFTW_ALLOWED      
      if(iLayer<= 118) 
      {
        CUDA_WRAP_check_hidden_currents3D(this,p_CellArray,iLayer,l_My,l_Mz,h_P->JxBeam,"JxBeam");
	CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_layers[iLayer]->Ex,"3D:");
	CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_C,h_P);
      }
      
      if(GetRank() == 1) CUDA_DEBUG_print3DmatrixLayer(d_Ey3D,iLayer,l_My,l_Mz,"after Layer118");
#endif      
      
      gettimeofday(&tvl2,NULL);
      //printf("rank %d loop %e \n",GetRank(),(tvl2.tv_sec - tvl1.tv_sec)+1e-6*(tvl2.tv_usec - tvl1.tv_usec));
   }
   cout << " maxVx = " << maxVx ;
#endif   
      printf("after loop particle %e \n",host_send_layer->particles[0].f_Y);
   
   //printf("before send Layer %d ********************************************************************************** \n",GetRank());
  // if(GetRank() == 1) exit(0);
   CUDA_WRAP_printLayerParticles(h_layers[1],"before send layer");
   
   
#ifndef CPU_COMPUTING
   host_send_layer = h_layers[1]; 
#endif
   
   
 //  printf("before send 1st particle %e \n",host_send_layer->particles[0].f_Y);
   
   Np = CUDA_WRAP_getLayerParticlesNumber(this,p_CellArray,1,l_My,l_Mz,"before sendLayer");
   //printf("before send layer Np %d \n",Np);

   if(GetRank() > 0) SendLayer(host_send_layer,l_My,l_Mz,host_send_layer->Np);
   cuLayerPrintCentre(h_layers[1],1,this,p_CellArray,"after sendLayer");
   
   if(GetRank() == 1) CUDA_DEBUG_print3DmatrixLayer(d_Ey3D,l_Mx/2,l_My,l_Mz,"after sendLayer");

   //printf("after send Layer %d *********************************************************************************** \n",GetRank());
   
   //exit(0);
   
   gettimeofday(&tv2,NULL);
   //printf("allLayers %e \n",(tv2.tv_sec - tv1.tv_sec)+1e-6*(tv2.tv_usec - tv1.tv_usec));
//  if(GetRank() == 0) exit(0);
    
};

//---Mesh:: ---------------------------------------------->
void Mesh::MoveSplitLayer(int iLayer,int iSplit) 
{
   long i, j, k, n;
   double part = 1.;
   int iFullStep = 0;
   double maxEx =0.;
   double maxEy =0.;
   double maxEz =0.;
   double maxRho = 0.;
   double minRho = 0.;
   double totalJy, totalJz;
   cudaLayer *h_cl,*h_pl;
   double g_time = 0.0,p_time = 0.0,i_time = 0.0;
   struct timeval tg1,tg2,tp1,tp2,ti1,ti2,t1,t2;
   cudaLayer *h_C,*h_P;
   printf("in MoveSplitLayer");
   //printf("rank %d in move \n",GetRank());
//   if(GetRank() == 0 && (iSplit == 1)) exit(0);

   if (iLayer == (l_Mx-1))
   {
      CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"in_MoveSplitLayer",this->GetControlDomain()->p_Cntrl->GetNstep());
   }
   
   gettimeofday(&t1,NULL);
   gettimeofday(&tg1,NULL);

#ifndef CUDA_WRAP_FFTW_ALLOWED   
   getLayersPC(&h_C,&h_P);
#endif   
//    cuLayerPrintCentre(h_P,-50,this,p_CellLayerP,"P before guess ");
//    cuLayerPrintCentre(h_C,-51,this,p_CellLayerC,"C before guess ");
   
   printf("before guess %d\n",GetRank());
   GuessFieldsHydroLinLayerSplit(iLayer,iSplit);
   printf("after guess %d\n",GetRank());
   //cuLayerPrintCentre(h_C,iLayer,this,p_CellArray);
   
//    cuLayerPrintCentre(h_P,-52,this,p_CellLayerP,"P after guess ");
//    cuLayerPrintCentre(h_C,-53,this,p_CellLayerC,"C after guess ");
//

   //gettimeofday(&tg2,NULL);
   //g_time = (tg2.tv_sec - tg1.tv_sec)+(tg2.tv_usec - tg1.tv_usec)*1e-6;
   printf("rank %d guess \n",GetRank());
//   if(GetRank() == 0 && (iSplit == 1)) exit(0);
   
    getLayersPC(&h_cl,&h_pl);
   CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   printf("check all hidden fields \n");
   //#ifdef CUDA_WRAP_FFTW_ALLOWED     
   //ExchangeFieldsSplit(iLayer);
   
  // return;
    
   i= iLayer;
   j = k = 0;
   n = GetNyz(  j,  k);
   Cell &c = p_CellLayerC[n];
   double Exc = c.f_Ex;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         n = GetNyz(  j,  k);
         Cell &ccc = p_CellLayerC[n];
         maxEx = max(maxEx, fabs(ccc.f_Ex));
      }
   }
//#endif   
   double frac_rude,frac_ideal;
   
   printf("rank %d before iter loop \n",GetRank());

   int niter = nIter();
   for (int iter=0; iter<niter; iter++) {
       printf("rank %d iteration %d begins \n",GetRank(),iter);

      iFullStep = 0;
//#ifdef CUDA_WRAP_FFTW_ALLOWED        
      ClearCurrentsSplit();
//#endif      
      CUDA_WRAP_clearCurrents(l_My,l_Mz,1);
//#ifdef CUDA_WRAP_FFTW_ALLOWED        
      ClearRhoSplit();
//#endif      
      printf("113 \n");
      if(iLayer <= 118 )
      {
	int zz = 0;
      }
      getLayersPC(&h_cl,&h_pl);
     CUDA_WRAP_printLayerParticles(h_pl,"before 1");
     CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host before 1",this->GetControlDomain()->p_Cntrl->GetNstep());
     printf(" print particle list host\n");
     CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
     if(iLayer<= 118 )
     {
//      CUDA_DEBUG_printDdevice_matrix(l_My,l_Mz,h_pl->Ex,"before move");
     }
     
     printf("rank %d iteration %d begins particles \n",GetRank(),iter);
//         cuLayerPrintCentre(h_P,-40,this,p_CellLayerP,"P before MoveParticlesLayerSplit");
//         cuLayerPrintCentre(h_C,-41,this,p_CellLayerC,"C before  MoveParticlesLayerSplit");
      //exit(0);
      ControlPoint(this,"_before_MoveParticlesLayerSplit");
      cudaLayer *h_pl_dp,*h_pl,*d_pl;
      cudaLayer *h_cl_dp,*h_cl,*d_cl;
      CUDA_WRAP_copy_from_CellArray2Layer(this,
                                      this->get_p_CellLayerP(),
                                      &h_pl,iLayer);
      CUDA_WRAP_copy_from_CellArray2Layer(this,
                                            this->get_p_CellLayerC(),
                                            &h_cl,iLayer);
      copyLayerFromHostToDevice(&h_pl_dp,h_pl);
      copyLayerStructureFromHostToDevice(&d_pl,h_pl_dp);
      copyLayerFromHostToDevice(&h_cl_dp,h_cl);
            copyLayerStructureFromHostToDevice(&d_cl,h_cl_dp);
      setLayersPC(d_cl,d_pl);
      MoveParticlesLayerSplit(iLayer, iSplit,iFullStep, part);//,h_pl,d_pl,h_cl,d_cl);
      string where = "MoveParticlesLayerSplit_Player";
      plasmaControlPoint(this,p_CellLayerP,where.c_str());


//         cuLayerPrintCentre(h_P,-42,this,p_CellLayerP,"P after MoveParticlesLayerSplit");
//         cuLayerPrintCentre(h_C,-43,this,p_CellLayerC,"C after MoveParticlesLayerSplit");
      
     printf("rank %d iteration %d begins after particles \n",GetRank(),iter);
      gettimeofday(&tp2,NULL);
      p_time += (tp2.tv_sec - tp1.tv_sec)+(tp2.tv_usec - tp1.tv_usec)*1e-6;
     CUDA_WRAP_printLayerParticles(h_pl,"after 1");
     CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host_after_1",this->GetControlDomain()->p_Cntrl->GetNstep());
       CUDA_WRAP_printLayerParticles(h_pl,"no");
      getLayersPC(&h_cl,&h_pl);
#ifndef CUDA_WRAP_FFTW_ALLOWED       
      CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jx,"Jx");
      CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jy,"Jy");
      CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jz,"Jz");
      CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Rho,"Rho");
#endif      
      
//      CUDA_WRAP_compare_device_array(l_My*l_Mz,,d_a1,&frac_ideal,&frac_rude,s1,where,details_flag);
//#ifdef CUDA_WRAP_FFTW_ALLOWED        
      ExchangeCurrentsSplit(iLayer);
//#endif      
      //   ExchangeRho(iLayer);

      printf("rank %d iteration %d before iterate\n ",GetRank(),iter);
      
//       cuLayerPrintCentre(h_P,-1000,this,p_CellLayerP,"P before Iterate");
//       cuLayerPrintCentre(h_C,-1001,this,p_CellLayerC,"C before Iterate");

     printf("before iterate1 %d\n",GetRank());
      IterateFieldsHydroLinLayerSplit(iLayer,iSplit,iter);
     printf("rank %d after iterate\n ",GetRank());
     // printf("after  iterate1 %d\n",GetRank()); 
      //cuLayerPrintCentre(h_C,iLayer,this,p_CellArray);

      CUDA_WRAP_copy_from_CellArray2Layer(this,p_CellArray,&h_pl,iLayer);
      //copyLayerFromHostToDevice(&h_P,p_CellLayerP);

      cuLayerPrintCentre(h_P,-1002,this,p_CellLayerP,"P after Iterate");
      cuLayerPrintCentre(h_C,-1003,this,p_CellLayerC,"C after Iterate");

      printf("rank %d iteration %d after iterate\n ",GetRank(),iter);
       gettimeofday(&ti2,NULL);
       i_time += (ti2.tv_sec - ti1.tv_sec)+(ti2.tv_usec - ti1.tv_usec)*1e-6;
//#ifdef CUDA_WRAP_FFTW_ALLOWED         
      ExchangeFieldsSplit(iLayer);
//#endif      
   };
//#ifdef CUDA_WRAP_FFTW_ALLOWED  
   ClearCurrentsSplit();
//#endif   
#ifndef CUDA_WRAP_FFTW_ALLOWED    
   CUDA_WRAP_clearCurrents(l_My,l_Mz,1);
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Rho,"Rho");
#endif    
   
   part = 1.;
   iFullStep = 1;
   
#ifndef CUDA_WRAP_FFTW_ALLOWED    
   CUDA_WRAP_printLayerParticles(h_pl,"before 2");
   CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host before 2"); 
    CUDA_WRAP_printParticleListFromHost(this,p_CellLayerC,l_Mx-1,l_My,l_Mz,"host бефоре2 C");
   if(iLayer<= 118)
   {
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   }
#endif   
   gettimeofday(&tp1,NULL);
           //printf("rank %d begins particles2 \n",GetRank());
   MoveParticlesLayerSplit(iLayer,iSplit, iFullStep, part);
        //printf("rank %d after particles2 \n",GetRank());
   gettimeofday(&tp2,NULL);
   p_time += (tp2.tv_sec - tp1.tv_sec)+(tp2.tv_usec - tp1.tv_usec)*1e-6;
#ifndef CUDA_WRAP_FFTW_ALLOWED    
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jx,"Jx");
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jy,"Jy");
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Jz,"Jz");
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Rho,"Rho");   
    CUDA_WRAP_printLayerParticles(h_pl,"yes");
    CUDA_WRAP_printLayerParticles(h_cl,"yesC");
     CUDA_WRAP_printParticleListFromHost(this,p_CellLayerP,l_Mx-1,l_My,l_Mz,"host yes");
     CUDA_WRAP_printParticleListFromHost(this,p_CellLayerC,l_Mx-1,l_My,l_Mz,"host yesC");
   CUDA_WRAP_check_hidden_currents(this,iLayer,l_My,l_Mz,p_CellLayerC,h_cl->Rho,"Rho");
#endif   
//#ifdef CUDA_WRAP_FFTW_ALLOWED     
   ExchangeCurrentsSplit(iLayer);
//#endif   
//   ExchangeRho(iLayer);
#ifndef CUDA_WRAP_FFTW_ALLOWED    
   if(iLayer<= 118)
   {
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   }
#endif   
   gettimeofday(&ti1,NULL);
       //printf("rank %d  before iterate2\n ",GetRank());

 //  printf("before iterate %d\n",GetRank());
   double err = IterateFieldsHydroLinLayerSplit(iLayer,iSplit,-1);
         //printf("rank %d after iterate\n ",GetRank());
 //  printf("after  iterate %d\n",GetRank());
   //cuLayerPrintCentre(h_C,iLayer,this,p_CellArray);
   
   cuLayerPrintCentre(h_C,iLayer,this,p_CellArray,"after  iterate %d\n");
   gettimeofday(&ti2,NULL);
   i_time += (ti2.tv_sec - ti1.tv_sec)+(ti2.tv_usec - ti1.tv_usec)*1e-6;
   if(iLayer<= 118)
   {
      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   }  
//#ifdef CUDA_WRAP_FFTW_ALLOWED     
   ExchangeFieldsSplit(iLayer);
//#endif   
   gettimeofday(&t2,NULL);
   //printf("end move: proc %/*3*/d Layer %3d guess %e iterate %e particles %e total %e\n",g_time,i_time,p_time,(t2.tv_sec - t1.tv_sec)+(t2.tv_usec - t1.tv_usec)*1e-6,GetRank(),iLayer);

   return;
}

int Mesh::getLayerParticles(int iLayer)
{
   int num = 0;
   int i,k,j;

   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {

         i=iLayer;
         int ip = i+1;
         long ncc = GetNyz(j,  k);

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


         p = pcc.p_Particles;

         if (p==NULL)
            continue;

         p_PrevPart = NULL;
         while(p)
         {
            Particle *p_next = p->p_Next;
	        num++;
	        p = p_next;
         }
//         printf("cell %10d %5d %5d particles %15d \n",i,j,k,num);
      }
   }
   return num;
}

//---Mesh:: ---------------------------------------------->
void Mesh::MoveParticlesLayerSplit(int iLayer,int iSplit, int iFullStep, double part)
{
   double Vx = 0.;
   int np = 0;
   l_Processed = 0;
   f_GammaMax  = 1.;
   double ElaserPhoton = 1.2398e-4/domain()->GetWavelength();
   int isort;
   long i, j, k, n;
   double ts = Ts();
   double hx = HxSplit()*part;
   double hy = Hy();
   double hz = Hz();
   

   printf("in :MoveParticlesLayerSplit  \n ");
   //exit(0); 
//#ifndef CUDA_WRAP_FFTW_ALLOWED   
   //getLayersPC(&h_cl,&h_pl);
//#endif   
   //exit(0);
 //  CUDA_WRAP_printLayerParticles(h_pl,"IN particle ");
   if((iLayer<= 118))  
   {
      //CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   }
   
//#ifndef CUDA_WRAP_FFTW_ALLOWED 
   printf("before  particlesPrepareAtLayer \n ");
   printf("after  particlesPrepareAtLayer \n "); 
   //exit(0);
//#endif   
//    CUDA_WRAP_print_ particles__before_MoveParticlesLayerSplit_Player_0000000001.datlayer(d_pl,Np,"layerFormed",l_My,l_Mz);
   printf("after d_pl formed\n");
 //  exit(0);
   int nsorts = domain()->GetNsorts();
#ifdef CUDA_WRAP_PARTICLE_HOST_COMPUTATIONS
   i = iLayer;
   j = l_My/2.;
   k = l_Mz/2.;
   double xco = X(i) + domain()->p_Cntrl->GetPhase();
   double yco = Y(j) - domain()->GetYlength()/2.;
   double zco = Z(k) - domain()->GetZlength()/2.;
   

   double dens = 0.;
   
   for (isort=0; isort<nsorts; isort++) {
      Specie* spec = domain()->GetSpecie(isort);
      if (spec->IsBeam()) continue;
      dens += spec->Density(xco,yco,zco)*spec->GetQ2M();
   };

/*
   for (k=0; k<l_Mz; k++)
   {
      for (j=0; j<l_My; j++)
      {
         long nccc = GetNyz(j,  k);
         Cell &ccc = p_CellLayerC[nccc];
         Cell &pcc = p_CellLayerP[nccc];
         ccc.f_Jx = pcc.f_Jx + hx*fabs(dens)*(ccc.f_Ex+pcc.f_Ex)/2.;
         ccc.f_Jy = pcc.f_Jy + hx*fabs(dens)*(ccc.f_Ey+pcc.f_Ey)/2.;
         ccc.f_Jz = pcc.f_Jz + hx*fabs(dens)*(ccc.f_Ez+pcc.f_Ez)/2.;
         Vx = 0.5*(ccc.f_Jx + pcc.f_Jx);
         maxVx = max(maxVx,fabs(Vx));

         if (Vx > 0. && j==l_My/2 && k==l_Mz/2) {
            double dummy = 0.;
         };

         ccc.f_Jx *= 1./(1.-Vx);
         ccc.f_Jy *= 1./(1.-Vx);
         ccc.f_Jz *= 1./(1.-Vx);
      }
   }

   return;
*/
#endif


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
   
#ifdef CUDA_WRAP_PARTICLE_HOST_COMPUTATIONS   

   for (k=0; k<l_Mz; k++)
   {
      printf("particles k %d \n",k);	   
      for (j=0; j<l_My; j++)
      {
	 
         i=iLayer;
         int ip = i+1;
         long ncc = GetNyz(j,  k);

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

         p = pcc.p_Particles;

         if (p==NULL)
            continue;

         p_PrevPart = NULL;
         while(p)
         {
            Particle *p_next = p->p_Next;
	    
            isort = p->GetSort();
            if (isort > 0) {
               int ttest = 0;
            }
            if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
               double check1=0;
            };
            create_h_plasma_particles(this->getLayerParticles(iLayer));
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,0,(double)j);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,1,(double)k);  
            l_Processed++;
            double weight = p->f_Weight;
            double xp  = p->f_X;
            double yp  = p->f_Y;
            double zp  = p->f_Z;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,2,weight);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,3,xp);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,4,yp);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,5,zp);  

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
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,6,px);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,7,py);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,8,pz);  
            double pxp = px;
            double pyp = py;
            double pzp = pz;
            double gammap = sqrt(1. + px*px + py*py + pz*pz);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,9,gammap);  

            Vx = px / gammap;
            maxVx = max(maxVx,fabs(Vx));
            double q2m = p->f_Q2m;
            if(q2m > 0)
            {
               int qq = 0;
            }

            double Vxp = Vx;
            double Vyp = py/gammap;
            double Vzp = pz/gammap;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,9,Vxp);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,10,Vyp);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,11,Vzp);  
	    

            double y_est = j + yp + Vyp/(1.-Vxp)*hx/hy;
            double z_est = k + zp + Vzp/(1.-Vxp)*hx/hz;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,12,y_est);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,13,z_est);  

            while (y_est > l_My) y_est -= l_My;
            while (y_est < 0)    y_est += l_My;
            while (z_est > l_Mz) z_est -= l_Mz;
            while (z_est < 0)    z_est += l_Mz;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,14,y_est);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,15,z_est);  

            int j_est = y_est;
            int k_est = z_est;

            double ym = y_est - j_est;
            double zm = y_est - z_est;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,16,ym);  
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,17,zm);  

            if (ym + yp != 0.) {
               double dummy = 0.;
            }
/*
            ym = yp;
            zm = zp;

            j_est = j;
            k_est = k;
*/
/*
            long nccc = npcc - 1;
            long ncpc = nppc - 1;
            long nccp = npcp - 1;
            long ncpp = nppp - 1;

            if (j_est != j || k_est !=k) {
               nccc = GetN(i,  j,  k);
               ncpc = GetN(i,  j+1,k);
               nccp = GetN(i,  j,  k+1);
               ncpp = GetN(i,  j+1,k+1);
            }
*/

            Cell &ccc = p_CellLayerC[ncc];
            Cell &cpc = p_CellLayerC[npc];
            Cell &ccp = p_CellLayerC[ncp];
            Cell &cpp = p_CellLayerC[npp];
            Cell &cmc = p_CellLayerC[nmc];
            Cell &ccm = p_CellLayerC[ncm];
            Cell &cmm = p_CellLayerC[nmm];
            Cell &cmp = p_CellLayerC[nmp];
            Cell &cpm = p_CellLayerC[npm];

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
/*
            double ys = yp - 0.5;
            double zs = zp - 0.5;  

            double yms = ym - 0.5;
            double zms = zm - 0.5;  

            double ayc = 1.-ys*ys;
            double aym = 0.5*(-ys + ys*ys);
            double ayp = 0.5*( ys + ys*ys);
            double azc = 1.-zs*zs;
            double azm = 0.5*(-zs + zs*zs);
            double azp = 0.5*( zs + zs*zs);

            double myc = 1.-yms*yms;
            double mym = 0.5*(-yms + yms*yms);
            double myp = 0.5*( yms + yms*yms);
            double mzc = 1.-zms*zms;
            double mzm = 0.5*(-zms + zms*zms);
            double mzp = 0.5*( zms + zms*zms);

            double apcc = ayc*azc;
            double appc = ayp*azc;
            double apcp = ayc*azp;
            double appp = ayp*azp;
            double appm = ayp*azm;
            double apmp = aym*azp;
            double apmc = aym*azc;
            double apcm = ayc*azm;
            double apmm = aym*azm;

            double accc = myc*mzc;
            double acpc = myp*mzc;
            double accp = myc*mzp;
            double acpp = myp*mzp;
            double acpm = myp*mzm;
            double acmp = mym*mzp;
            double acmc = mym*mzc;
            double accm = myc*mzm;
            double acmm = mym*mzm;
*/
            double ex, ey, ez;
            double exp, eyp, ezp;
            double bxp, byp, bzp;
            double exm, eym, ezm;
            double bxm, bym, bzm;

            double bx=0.;
            double by=0.;
            double bz=0.;

            exp = pcc.f_Ex;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,18,exp);
//            exp = apcc*pcc.f_Ex + appc*ppc.f_Ex + apcp*pcp.f_Ex + appp*ppp.f_Ex;
	    if(np == 14253)
	    {
	       int i14 = 0;
	    }
	    

            eyp = pcc.f_Ey;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,19,eyp);
//            eyp = apcc*pcc.f_Ey + appc*ppc.f_Ey + apcp*pcp.f_Ey + appp*ppp.f_Ey;

            ezp = pcc.f_Ez;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,20,ezp);
//            ezp = apcc*pcc.f_Ez + appc*ppc.f_Ez + apcp*pcp.f_Ez + appp*ppp.f_Ez;

            bxp = pcc.f_Bx;
//            bxp = apcc*pcc.f_Bx + appc*ppc.f_Bx + apcp*pcp.f_Bx + appp*ppp.f_Bx;
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,21,bxp);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,22,accc);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,23,appc);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,24,apcp);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,25,appp);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,26,pcc.f_Bx);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,27,ppc.f_Bx);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,28,pcp.f_Bx);
            CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,29,ppp.f_Bx);
            byp = pcc.f_By;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,30,byp);
//            byp = apcc*pcc.f_By + appc*ppc.f_By + apcp*pcp.f_By + appp*ppp.f_By;

            bzp = pcc.f_Bz;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,31,bzp);	    
//            bzp = apcc*pcc.f_Bz + appc*ppc.f_Bz + apcp*pcp.f_Bz + appp*ppp.f_Bz;

            exm = ccc.f_Ex;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,32,exm);	    
//            exm = accc*ccc.f_Ex + acpc*cpc.f_Ex + accp*ccp.f_Ex + acpp*cpp.f_Ex;

            eym = ccc.f_Ey;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,33,eym);	    
//            eym = accc*ccc.f_Ey + acpc*cpc.f_Ey + accp*ccp.f_Ey + acpp*cpp.f_Ey;

            ezm = ccc.f_Ez;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,34,ezm);	    
//            ezm = accc*ccc.f_Ez + acpc*cpc.f_Ez + accp*ccp.f_Ez + acpp*cpp.f_Ez;

            bxm = ccc.f_Bx;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,35,bxm);	    
//            bxm = accc*ccc.f_Bx + acpc*cpc.f_Bx + accp*ccp.f_Bx + acpp*cpp.f_Bx;

            bym = ccc.f_By;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,36,bym);	    
//            bym = accc*ccc.f_By + acpc*cpc.f_By + accp*ccp.f_By + acpp*cpp.f_By;

            bzm = ccc.f_Bz;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,37,bzm);	    
//            bzm = accc*ccc.f_Bz + acpc*cpc.f_Bz + accp*ccp.f_Bz + acpp*cpp.f_Bz;

/*
            exp =
               apcc*pcc.f_Ex + appc*ppc.f_Ex + apcp*pcp.f_Ex + appp*ppp.f_Ex +
               appm*ppm.f_Ex + apmp*pmp.f_Ex + apmc*pmc.f_Ex + apcm*pcm.f_Ex + apmm*pmm.f_Ex;

            eyp =
               apcc*pcc.f_Ey + appc*ppc.f_Ey + apcp*pcp.f_Ey + appp*ppp.f_Ey +
               appm*ppm.f_Ey + apmp*pmp.f_Ey + apmc*pmc.f_Ey + apcm*pcm.f_Ey + apmm*pmm.f_Ey;

            ezp =
               apcc*pcc.f_Ez + appc*ppc.f_Ez + apcp*pcp.f_Ez + appp*ppp.f_Ez +
               appm*ppm.f_Ez + apmp*pmp.f_Ez + apmc*pmc.f_Ez + apcm*pcm.f_Ez + apmm*pmm.f_Ez;

            bxp =
               apcc*pcc.f_Bx + appc*ppc.f_Bx + apcp*pcp.f_Bx + appp*ppp.f_Bx +
               appm*ppm.f_Bx + apmp*pmp.f_Bx + apmc*pmc.f_Bx + apcm*pcm.f_Bx + apmm*pmm.f_Bx;

            byp =
               apcc*pcc.f_By + appc*ppc.f_By + apcp*pcp.f_By + appp*ppp.f_By +
               appm*ppm.f_By + apmp*pmp.f_By + apmc*pmc.f_By + apcm*pcm.f_By + apmm*pmm.f_By;

            bzp =
               apcc*pcc.f_Bz + appc*ppc.f_Bz + apcp*pcp.f_Bz + appp*ppp.f_Bz +
               appm*ppm.f_Bz + apmp*pmp.f_Bz + apmc*pmc.f_Bz + apcm*pcm.f_Bz + apmm*pmm.f_Bz;

            exm =
               accc*ccc.f_Ex + acpc*cpc.f_Ex + accp*ccp.f_Ex + acpp*cpp.f_Ex +
               acpm*cpm.f_Ex + acmp*cmp.f_Ex + acmc*cmc.f_Ex + accm*ccm.f_Ex + acmm*cmm.f_Ex;

            eym =
               accc*ccc.f_Ey + acpc*cpc.f_Ey + accp*ccp.f_Ey + acpp*cpp.f_Ey +
               acpm*cpm.f_Ey + acmp*cmp.f_Ey + acmc*cmc.f_Ey + accm*ccm.f_Ey + acmm*cmm.f_Ey;

            ezm =
               accc*ccc.f_Ez + acpc*cpc.f_Ez + accp*ccp.f_Ez + acpp*cpp.f_Ez +
               acpm*cpm.f_Ez + acmp*cmp.f_Ez + acmc*cmc.f_Ez + accm*ccm.f_Ez + acmm*cmm.f_Ez;

            bxm =
               accc*ccc.f_Bx + acpc*cpc.f_Bx + accp*ccp.f_Bx + acpp*cpp.f_Bx +
               acpm*cpm.f_Bx + acmp*cmp.f_Bx + acmc*cmc.f_Bx + accm*ccm.f_Bx + acmm*cmm.f_Bx;

            bym =
               accc*ccc.f_By + acpc*cpc.f_By + accp*ccp.f_By + acpp*cpp.f_By +
               acpm*cpm.f_By + acmp*cmp.f_By + acmc*cmc.f_By + accm*ccm.f_By + acmm*cmm.f_By;

            bzm =
               accc*ccc.f_Bz + acpc*cpc.f_Bz + accp*ccp.f_Bz + acpp*cpp.f_Bz +
               acpm*cpm.f_Bz + acmp*cmp.f_Bz + acmc*cmc.f_Bz + accm*ccm.f_Bz + acmm*cmm.f_Bz;

*/
            ex = 0.5*(exp+exm);
            ey = 0.5*(eyp+eym);
            ez = 0.5*(ezp+ezm);

            bx = 0.5*(bxp+bxm);
            by = 0.5*(byp+bym);
            bz = 0.5*(bzp+bzm);
	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,38,ex);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,39,ey);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,40,ez);

	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,22,bx);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,23,by);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,24,bz);	    
	    
            double ex1 = ex;
            double ey1 = ey;
            double ez1 = ez;


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

            ex *= q2m*hx/2.;
            ey *= q2m*hx/2.;
            ez *= q2m*hx/2.;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,41,ex);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,42,ey);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,43,ez);	    
/*
            px += 2*ex;
            py += 2*ey;
            pz += 2*ez;
*/

            px += ex/(1.-Vx);
            py += ey/(1.-Vx);
            pz += ez/(1.-Vx);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,44,px);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,45,py);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,46,pz);	  	    

            double gamma = sqrt(1. + px*px + py*py + pz*pz);     //!!!!!!

            if (f_GammaMax < gamma)
               f_GammaMax = gamma;

            double gamma_r = 1./gamma;																	 //!!!!!!

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
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,47,p3x);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,48,p3y);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,49,p3z);	  	    

            p3x *= co;
            p3y *= co;
            p3z *= co;

            double px_new = p3y*bz - p3z*by;
            double py_new = p3z*bx - p3x*bz;
            double pz_new = p3x*by - p3y*bx;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,50,px_new);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,51,py_new);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,52,pz_new);	  	    
	    

            px += ex/(1.-Vx) + px_new;
            py += ey/(1.-Vx) + py_new;
            pz += ez/(1.-Vx) + pz_new;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,53,px);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,54,py);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,55,pz);	 	    
/*
            px += ex + px_new;
            py += ey + py_new;
            pz += ez + pz_new;
*/
            gamma = sqrt(1. + px*px + py*py + pz*pz);
            double Vxm = px/gamma;
            double Vym = py/gamma;
            double Vzm = pz/gamma;

            Vx = 0.5*(Vxm+Vxp);
            maxVx = max(maxVx,fabs(Vx));
            double Vy = 0.5*(Vym+Vyp);
            double Vz = 0.5*(Vzm+Vzp);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,56,Vx);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,57,Vy);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,58,Vz);	 
	    
            isort = p->GetSort();

            djx = weight*djx0[isort]*Vxm;
            djy = weight*djy0[isort]*Vym;
            djz = weight*djz0[isort]*Vzm;
            double drho = weight*drho0[isort];
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,59,djx);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,60,djy);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,61,djz);	 
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,62,drho);	    
	    
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
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,63,djx);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,64,djy);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,65,djz);	 
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,66,drho);
	    
            double dy = Vy*hx/hy;
            double dz = Vz*hx/hz;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,67,dy);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,68,dz);

	    dy = dy/(1.-Vx);
            dz = dz/(1.-Vx);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,69,dy);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,70,dz);

            double partdx = 0.;
            double step = 1.;
            // --- first half-step

            if (j==l_My/2 && k==l_Mz/2) {
               double dummy = 0;
            };

            // Particle pusher cel per cell//////////////////////////////////////


            int j_jump = j;
            int k_jump = k;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,71,(double)j_jump);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,72,(double)k_jump);
	    
            xtmp = 0;
            ytmp = yp;
            ztmp = zp;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,73,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,74,ztmp);
	    
            if (fabs(dy)>1. || fabs(dz)>1.) {
               if (fabs(dy) > fabs(dz)) {
                  step = partdx = fabs(dy);
               } else {
                  step = partdx = fabs(dz);
               };
            }
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,75,step);	    
            

            if (partdx < 1.) {
               partdx = step = 1.;
            }
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,76,partdx);	    
            

            while (partdx>0.) {
               if (partdx > 1.) {
                  partdx -= 1.;
                  part_step = 1./step;
               } else {
                  part_step = partdx/step;
                  partdx = 0.;
               }
               xtmp = 0.;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,77,partdx);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,78,part_step);
	       

               ytmp += dy*part_step + j_jump;
               ztmp += dz*part_step + k_jump;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,79,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,80,ztmp);
	       

               while (ytmp > l_My) ytmp -= l_My;
               while (ytmp < 0) ytmp += l_My;
               while (ztmp > l_Mz) ztmp -= l_Mz;
               while (ztmp < 0) ztmp += l_Mz;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,81,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,82,ztmp);
					 

               int j_jump = ytmp;
               int k_jump = ztmp;
               ytmp -= j_jump;
               ztmp -= k_jump;
               if (ytmp < 0. || ytmp > 1. || ztmp < 0. || ztmp > 1.) {
                  double checkpoint21 = 0.;
               };
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,83,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,84,ztmp);
               xtmp = 0;

               int itmp = iLayer;
               int jtmp = j_jump;
               int ktmp = k_jump;
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,85,(double)j_jump);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,86,(double)k_jump);
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,94,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,95,ztmp);	    
	    
               int ntmp = GetN(itmp,jtmp,ktmp);

               if (fabs(djy) > 0.) {
                  int check = 0;
               };
               long ncc_check = GetNyz(jtmp,  ktmp);  
               CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,87,djx);
               CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,88,djy);
               CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,89,djz);
               CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,90,drho);
	       
	       if(np == 256)
	       {
                  int i256 = 0;		 
	       }
	       
               CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,91,(double)(ncc_check- (l_dMy + l_sizeY *  l_dMz + 2*l_dMy*ktmp)));
	       //printf("rank %d b_deposit %3d %3d %3d y %e z %e rho %e \n ",GetRank(),iLayer,j,k,xtmp,ytmp,drho*part_step);
               DepositCurrentsInCellSplit(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
                  djx*part_step, djy*part_step, djz*part_step, drho*part_step,np);
	       
	      // printf("deposit np %d drho %e \n",np,drho);
	       
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,96,ytmp);	    
	    CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,97,ztmp);	       
	       
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
            
            DepositCurrentsInCell(p, isort, itmp, jtmp, ktmp, Vx, Vy, Vz, xtmp, ytmp, ztmp, 
               djx, djy, djz, drho);
/////////////////////// end of one cell pusher ///////////////////
*/
            if (iFullStep) {
               xtmp = 0.;
               p->SetP(px,py,pz);
	       CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,92,ytmp);
	       CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,93,ztmp);
               p->SetX(xtmp,ytmp,ztmp);
               long nnew = GetNyz(jtmp,ktmp);
               Cell &cnew = p_CellLayerC[nnew];
               p->p_Next = cnew.p_Particles;
               cnew.p_Particles = p;
               pcc.p_Particles = p_next;
            }
             CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,99,(double)np);
            np++;
            p = p_next;

            if (j==l_My/3 && k==l_Mz/3 && i==l_Mx/2) {
               double check1=0;
            };
         }
      }
   }
   long totalNe = domain()->GetSpecie(0)->GetNp();
   //   cout << "We have " << totalNe << " electrons \n";
//   cout << "Max Vx = " << maxVx << endl;
   
   if((iLayer<= 118)) 
   {
//      CUDA_WRAP_check_all_hidden_fields(this,iLayer,l_My,l_Mz,p_CellLayerC,p_CellLayerP,h_cl,h_pl);
   }
   //printf("deposit np %d \n",np);
#endif
   int nstep = this->domain()->p_Cntrl->GetNstep();
   cudaLayer *d_cl, *d_pl;


   int Np = CUDA_WRAP_get_particles_number(this,this->get_p_CellLayerP());
   //particlesPrepareAtLayer(this,p_CellLayerP,p_CellLayerC,iLayer,l_My,l_Mz);
   printf("before  cuMoveSplitParticles\n  ");
   cuMoveSplitParticles(iLayer,iSplit,Np,l_Mx,l_My,l_Mz,hx,hy,hz,
                                      djx0,djy0,djz0,drho0,nsorts,iFullStep,nstep);
   printf("after cuMoveSplitParticles \n");
#ifdef CUDA_WRAP_PARTICLE_HOST_COMPUTATIONS
   delete[] djx0;
   delete[] djy0;
   delete[] djz0;
   delete[] drho0;
   delete[] iAtomTypeArray;
#endif   

}

//---Mesh::DepositCurrentsInCellSplit ---------------------------------------------->
void Mesh::DepositCurrentsInCellSplit(
                                 Particle *p, int isort, 
                                 int i, int j, int k, 
                                 double Vx, double Vy, double Vz, 
                                 double x, double y, double z, 
                                 double djx, double djy, double djz, double drho,int np)
{
   long ncc = GetNyz(j,  k);

   long npc = ncc + 1;
   long ncp = ncc + l_sizeY;
   long npp = ncp + 1;
   long nmc = ncc - 1;
   long ncm = ncc - l_sizeY;
   long nmm = ncm - 1;
   long nmp = ncp - 1;
   long npm = npc - l_sizeY;

   Cell &ccc = p_CellLayerC[ncc];
   Cell &cpc = p_CellLayerC[npc];
   Cell &ccp = p_CellLayerC[ncp];
   Cell &cpp = p_CellLayerC[npp];
   Cell &cmc = p_CellLayerC[nmc];
   Cell &ccm = p_CellLayerC[ncm];
   Cell &cmm = p_CellLayerC[nmm];
   Cell &cmp = p_CellLayerC[nmp];
   Cell &cpm = p_CellLayerC[npm];

   x = 0.;
   double ayc = 1.-y;
   double ayp = y;
   double azc = 1.-z;
   double azp = z;

   double accc = ayc*azc;
   double acpc = ayp*azc;
   double accp = ayc*azp;
   double acpp = ayp*azp;
   
   //printf("                     rank %d in_deposit %3d %3d y %10.3e z %10.3e rho %10.3e  \n",GetRank(),j,k,x,y,drho);

/*
   double ys = y - 0.5;
   double zs = z - 0.5;

   double axc = 1.;
   double ayc = 1.-ys*ys;
   double aym = 0.5*(-ys + ys*ys);
   double ayp = 0.5*( ys + ys*ys);
   double azc = 1.-zs*zs;
   double azm = 0.5*(-zs + zs*zs);
   double azp = 0.5*( zs + zs*zs);

   double accc = ayc*azc;
   double acpc = ayp*azc;
   double accp = ayc*azp;
   double acpp = ayp*azp;
   double acpm = ayp*azm;
   double acmp = aym*azp;
   double acmc = aym*azc;
   double accm = ayc*azm;
   double acmm = aym*azm;
*/
   double weight = fabs(drho);
   
   ccc.f_Jx += djx;
   ccc.f_Jy += djy;
   ccc.f_Jz += djz;
   ccc.f_Dens += drho;

   CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,111,ccc.f_Jx);
   CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,112,ccc.f_Jy);
   CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,113,ccc.f_Jz);
   CUDA_WRAP_write_plasma_value(np,PLASMA_VALUES_NUMBER,114,ccc.f_Dens);
/*
   ccc.f_Jx += djx*accc;
   ccc.f_Jy += djy*accc;
   ccc.f_Jz += djz*accc;

   cpc.f_Jx += djx*acpc;
   cpc.f_Jy += djy*acpc;
   cpc.f_Jz += djz*acpc;

   ccp.f_Jx += djx*accp;
   ccp.f_Jy += djy*accp;
   ccp.f_Jz += djz*accp;

   cpp.f_Jx += djx*acpp;
   cpp.f_Jy += djy*acpp;
   cpp.f_Jz += djz*acpp;


   cmc.f_Jx += djx*acmc;
   cmc.f_Jy += djy*acmc;
   cmc.f_Jz += djz*acmc;

   ccm.f_Jx += djx*accm;
   ccm.f_Jy += djy*accm;
   ccm.f_Jz += djz*accm;

   cmm.f_Jx += djx*acmm;
   cmm.f_Jy += djy*acmm;
   cmm.f_Jz += djz*acmm;

   cpm.f_Jx += djx*acpm;
   cpm.f_Jy += djy*acpm;
   cpm.f_Jz += djz*acpm;

   cmp.f_Jx += djx*acmp;
   cmp.f_Jy += djy*acmp;
   cmp.f_Jz += djz*acmp;
*/
}


//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeFieldsSplit(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      Cell &cc0 = p_CellLayerC[GetNyz(0,k)];
      Cell &cp1 = p_CellLayerC[GetNyz(l_My,k)];
      Cell &cc1 = p_CellLayerC[GetNyz(-1,k)];
      Cell &cp0 = p_CellLayerC[GetNyz(l_My-1,k)];
      for (int idim=0; idim<FLD_DIM; idim++) {
         cc1.f_Fields[idim] = cp0.f_Fields[idim];
         cp1.f_Fields[idim] = cc0.f_Fields[idim];
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      Cell &cc0 = p_CellLayerC[GetNyz(j,0)];
      Cell &cp1 = p_CellLayerC[GetNyz(j,l_Mz)];
      Cell &cc1 = p_CellLayerC[GetNyz(j,-1)];
      Cell &cp0 = p_CellLayerC[GetNyz(j,l_Mz-1)];

      for (int idim=0; idim<FLD_DIM; idim++) {
         cc1.f_Fields[idim] = cp0.f_Fields[idim];
         cp1.f_Fields[idim] = cc0.f_Fields[idim];
      }
   }
};

//---Mesh:: ---------------------------------------------->
void Mesh::ExchangeCurrentsSplit(int iLayer) 
{
   for (int k=-1; k<l_Mz+1; k++) {
      Cell &cc0 = p_CellLayerC[GetNyz(0,k)];
      Cell &cp1 = p_CellLayerC[GetNyz(l_My,k)];
      Cell &cc1 = p_CellLayerC[GetNyz(-1,k)];
      Cell &cp0 = p_CellLayerC[GetNyz(l_My-1,k)];
      for (int idim=0; idim<CURR_DIM; idim++) {
         cc0.f_Currents[idim] += cp1.f_Currents[idim];
         cp0.f_Currents[idim] += cc1.f_Currents[idim];
         cp1.f_Currents[idim] = cc1.f_Currents[idim] = 0.; 
//         cc0.f_Currents[idim] = cp0.f_Currents[idim] = 0.;
      }
   }
   for (int j=-1; j<l_My+1; j++) {
      Cell &cc0 = p_CellLayerC[GetNyz(j,0)];
      Cell &cp1 = p_CellLayerC[GetNyz(j,l_Mz)];
      Cell &cc1 = p_CellLayerC[GetNyz(j,-1)];
      Cell &cp0 = p_CellLayerC[GetNyz(j,l_Mz-1)];
      for (int idim=0; idim<CURR_DIM; idim++) {
         cc0.f_Currents[idim] += cp1.f_Currents[idim];
         cp0.f_Currents[idim] += cc1.f_Currents[idim];
         cp1.f_Currents[idim] = cc1.f_Currents[idim] = 0.; 
//         cc0.f_Currents[idim] = cp0.f_Currents[idim] = 0.;
      }
   }
};
