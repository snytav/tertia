#include <stdio.h>

#include <stdlib.h>


#include "vlpl3d.h"

#include <sys/time.h>

#include "CUDA_WRAP/cuBeam.h"
#include "CUDA_WRAP/beam_copy.h"
#include "CUDA_WRAP/diagnostic_print.h"
#include "CUDA_WRAP/cuLayers.h"

#include "para.h"

#include "run_control.h"

int beamPrepareFirstCall = 1;
int Np = 0;

//---Domain:: --------------------->

int Domain::Step(void)
{
  
  printf("begin step rank %d \n",GetRank());

   //	out_Flog << "Domain::Step begins \n";

   int itmp = 0;
   p_Cntrl->Step();
   p_Cntrl->f_Phase += GetTs();
	if ( GetMPP()->GetnPE() == 0 )	{
	  //		cout << "\rt="<<GetPhase()<<"      ";
   }
   
   if (p_Cntrl->l_Nstep == 50) {
      out_Flog << "Right step \n";
   }
   
///////////////////////////////////////////////////////////////////
 
   
   if(beamPrepareFirstCall == 1)
   {
      puts("before beam prepare");
#ifndef CUDA_WRAP_FFTW_ALLOWED      
      CUDA_WRAP_beam_prepare(l_Xsize,l_Ysize,l_Zsize,p_M,p_M->p_CellArray);
#endif      
      Np = getBeamNp();
      printf("rank %d after prepare Np %d \n",GetRank(),Np);
      beamPrepareFirstCall = 0;
   }
   
#ifndef PARALLEL_ONLY   
///////////////////////////////////////////////////////////////////   
   struct timeval tv1,tv2,tvc1,tvc2;
      
   CUDA_WRAP_printBeamParticles(p_M,p_Cntrl->l_Nstep,"AfterMove");
 //  puts("BEFORE BEAM");
#ifdef CUDA_WRAP_COMPUTE_BEAM_ON_HOST   
   p_M->MoveBeamParticles();
 //  puts("AFTER BEAM");
#endif

   CUDA_WRAP_printBeamDensity3D(p_M,p_Cntrl->l_Nstep,"AfterMove");
      
   //puts("AFTER BEAM");
   
///////////////////////////////////////////////////////////////////
 //  CUDA_WRAP_compare3DFields(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray);
   gettimeofday(&tvc1,NULL);   
   printf("before beam move\n");
   CUDA_WRAP_beam_move(Np,l_Xsize,l_Ysize,l_Zsize,p_M->Hx(),p_M->Hy(),p_M->Hz(),p_M->Ts());
   printf("after beam move\n");
   gettimeofday(&tvc2,NULL);   
#endif

   printf("before sending particles rank %d \n",GetRank());
   
   //SendBeamParticles(&Np);
   printf("after sending particles rank %d \n",GetRank());



#ifndef PARALLEL_ONLY
   
   CUDA_WRAP_compareBeamCurrents(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray);
   
#ifdef COPY_BEAM_FROM_HOST
   CUDA_WRAP_copyBeamToArray(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray,&d_RhoBeam3D,&d_JxBeam3D);
#endif   
   
   
   
///////////////////////////////////////////////////////////////////   
   
//   p_M->ClearFields();
 //  Exchange(SPACK_JB);
//   Exchange(SPACK_PB);
#endif  

   CUDA_WRAP_diagnose(    l_Xsize,    l_Ysize,    l_Zsize,p_M->Hx(),p_M->Hy(),p_Cntrl->l_Nstep,p_M,p_M->p_CellArray);
   printf("long after sending particles rank %d \n",GetRank());
   gettimeofday(&tv1,NULL);
   p_M->MoveAllSplitLayers();
   gettimeofday(&tv2,NULL);
    printf("size in step %d %d %d \n",l_Xsize/2,l_Ysize,l_Zsize); 
    if(GetRank() == 1) CUDA_DEBUG_print3DmatrixLayer(d_Ey3D,l_Xsize/2,l_Ysize,l_Zsize,"after allLayer");

   printf("rank %d after AllLayers \n",GetRank());   
   //exit(0);
  // ParallelExit();
   
   printf(" field %e beam %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec),(tvc2.tv_sec-tvc1.tv_sec)+1e-6*(tvc2.tv_usec-tvc1.tv_usec));
   
 
   
//   CUDA_WRAP_diagnose(l_Xsize,l_Ysize,l_Zsize,p_M->Hx(),p_M->Hy(),p_Cntrl->l_Nstep,p_M,p_M->p_CellArray);

   printf("rank %d after diagnose \n",GetRank());   
   
   itmp = Check();
   /*
   if (HybridIncluded()) {
      p_M->MoveFieldsHydro();
   } else {
      p_M->MoveAllLayers();
   };
   printf(" field %e beam %e \n",(tv2.tv_sec-tv1.tv_sec)+1e-6*(tv2.tv_usec-tv1.tv_usec),(tvc2.tv_sec-tvc1.tv_sec)+1e-6*(tvc2.tv_usec-tvc1.tv_usec));
   */
//nt CUDA_WRAP_diagnose(int l_Xsize,int l_Ysize,int l_Zsize,double hx, double hy,int step,Mesh *mesh,Cell *p_CellArray);
   CUDA_WRAP_diagnose(    l_Xsize,    l_Ysize,    l_Zsize,p_M->Hx(),p_M->Hy(),p_Cntrl->l_Nstep,p_M,p_M->p_CellArray);
   //exit(0);
    /*


   CUDA_WRAP_compare3DFields(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray);
 //  CUDA_WRAP_copy3DFields(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray);
   CUDA_WRAP_compare3DFields(p_M,l_Xsize,l_Ysize,l_Zsize,p_M->p_CellArray);
   /*
   for (int n_iteration=0; n_iteration<1; n_iteration++) {
      p_M->MoveFields();
      p_M->SeedFrontParticles();
      p_M->MoveParticles();
      Exchange(SPACK_J);
      Exchange(SPACK_P);
   };
   */
/*   Exchange(SPACK_P);
   Exchange(SPACK_F);
   double t1 =  p_Cntrl->GetCPU();
   itmp = Check();
*/
/*
#ifdef V_MPI
   int root = 0;

   int ierr = MPI_Bcast(&itmp, sizeof(int),
      MPI_BYTE, root, MPI_COMM_WORLD);

   switch (ierr)
   {
   case MPI_SUCCESS:
      // No error
      break;
   case MPI_ERR_COMM:
      out_Flog << "Invalid communicator. A common error is to use a null communicator in a call. ierr = " << ierr << ", PE = " << GetmyPE() << endl;
      break;
   case MPI_ERR_COUNT:
      out_Flog << "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. ierr = " << ierr << ", PE = " << GetmyPE() << endl;
      break;
   case MPI_ERR_TYPE:
      out_Flog << "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). ierr = " << ierr << ", PE = " << GetmyPE() << endl;  					
      break;
   case MPI_ERR_BUFFER:
      out_Flog << "Invalid buffer pointer. Usually a null buffer where one is not valid. ierr = " << ierr << ", PE = " << GetmyPE() << endl;  					  					
      break;
   case MPI_ERR_ROOT:
      out_Flog << "Invalid root. The root must be specified as a rank in the communicator. Ranks must be between zero and the size of the communicator minus one. ierr = " << ierr << ", PE = " << GetmyPE() << endl;  					  					 	
      break;
   default:
      out_Flog << "Unknown error. ierr = " << ierr << ", PE = " << GetmyPE() << endl;				
   };

   //  if (ierr) out_Flog<<"Error step::broadcast. ierr="<<ierr<<", PE"<<GetmyPE()<<endl;
#endif
*/
   printf("rank %d end step \n",GetRank());   


   return 0;
}

//---Domain:: --------------------->

int Domain::GroupSteps(void)
{
   printf("begin group step %d \n",GetRank());
  /*
#ifdef V_MPI
  if (p_BufferMPI) {
    out_Flog << "detaching Buffer of size " << l_BufferMPIsize << endl;
    out_Flog.flush();
    MPI_Buffer_detach(p_BufferMPI, &l_BufferMPIsize);
    delete[] p_BufferMPI;
  }
  l_BufferMPIsize = 100000000 + MPI_BSEND_OVERHEAD;
  p_BufferMPI = new char[l_BufferMPIsize];
  out_Flog << "attaching Buffer of size " << l_BufferMPIsize << endl;
  out_Flog.flush();
  int err_buf_att = MPI_Buffer_attach(p_BufferMPI, l_BufferMPIsize);
  switch (err_buf_att)
    {
    case MPI_SUCCESS:
      out_Flog << "Buffer of size " << l_BufferMPIsize << " attached" << endl;
      out_Flog.flush();
      // No error
      break;
    case MPI_ERR_BUFFER:
      cerr << "Invalid buffer pointer. Usually a null buffer where one is not valid. err_buf_att = " << err_buf_att << endl;
      exit(1);
      break;
    case MPI_ERR_INTERN:
      cerr << "An internal error has been detected. This is fatal. Please send a bug report to mpi-bugs@mcs.anl.gov. err_buf_att = " << err_buf_att << endl;
      exit(1);
      break;
    default:
      cerr << "Unknown error. Domain::BroadCast MPI_Buffer_attach. err_buf_att = " << err_buf_att << endl;				
      exit(1);
    };
#endif
 */
  int itmp = 0;
  int i = 0;

  //	out_Flog << "Domain::GroupStep point is before of Diagnose() \n";
  printf("be dia %d \n",GetRank());
  
 // Diagnose();

  //	out_Flog << "Domain::GroupStep point is past of Diagnose() \n";

  for (i=0; i<(p_Cntrl->i_Ndiagnose); i++) {
    printf("before step %d itmp %d rank %d p_Cntrl->i_Ndiagnose %d iters %d \n",i,itmp,GetRank(),p_Cntrl->i_Ndiagnose,
                                                                        p_Cntrl->i_Ndiagnose+GetRank());
    itmp = Step();
    
   // if (itmp) return itmp;
    printf("after step %d itmp %d Ndia %d rank %d \n",i,itmp,p_Cntrl->i_Ndiagnose,GetRank());
  }
  printf("rank %d end group step %d \n",GetRank(),itmp);
//  ParallelExit();
  
  
  
  return 1;
}

//---Domain:: --------------------->

int Domain::Run(void)
{
#ifdef CUDA_WRAP_HIGH_PARALLEL_DEBUG  
  printf("before Group %d Xlen %d \n",GetRank(),f_Xlength);
#endif  
  SetXSize(&l_Xsize,f_Xlength);
  
  int itmp = 0;
  while(-1)
    {
#ifdef CUDA_WRAP_HIGH_PARALLEL_DEBUG      
      printf("before Group %d Xlen %d \n",GetRank(),f_Xlength);
#endif      
       
      if (itmp = GroupSteps())
	break;
      printf("in RUN %d itmp %d \n",GetRank(),itmp); 
      
    }
  return itmp;
}

//---Domain::Check --------------------->

int Domain::Check(void) 
{
  
  puts("in Check");
  if (p_Cntrl->PhaseSave()) {
    Save();
  }

  printf("p_Cntrl->PhaseMovie() %d \n",p_Cntrl->PhaseMovie());
  
  if (p_Cntrl->PhaseMovie()) {
    //    p_M->SaveCadrMovie(p_MovieFile);
    p_M->SaveCadrMovie2(p_MovieFile);
  }

  if (p_Cntrl->PhaseFieldFilter()) {
//    p_M->FilterFieldX();
//    Exchange(SPACK_F);
  }

  if (p_Cntrl->PhaseMovieH5()) {
    //    p_M->SaveCadrMovie(p_MovieFile);
    p_M->Save_Movie_Frame_H5(p_Cntrl->GetMovieFrameH5());
  }

  if (p_Cntrl->PhaseStop()) {
    return -1;
  }

  if (p_Cntrl->CPUStop()) {
    return -1;
  }

  return 0;
}
