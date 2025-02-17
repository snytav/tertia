#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include "vlpl3d.h"
#include "para.h"

//Domain::p_D = NULL;
Domain* Domain::p_D = NULL;
int MakeNewDirectory(char* directory);

//---------------------------- Domain::Domain -----------------------
Domain::Domain (char *infile, int rank) : NList("Domain")
{
   int i = 0;
   p_BufferMPI = NULL;
   l_BufferMPIsize = 0;

   Domain::p_D = this;

   str_SName = ".save";
   str_DName = ".dat";
   str_LogName = ".log";

   str_DataDirectory = ".";
   str_LogDirectory = ".";
   str_MovieDirectory = ".";

   int ierr=0;
/*
   if (rank == NIL) {
      ierr = MakeNewDirectory(str_DataDirectory);
      ierr = MakeNewDirectory(str_LogDirectory);
      ierr = MakeNewDirectory(str_MovieDirectory);
   }
#ifdef V_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
*/

   AddEntry("Hx", &f_Hx);
   AddEntry("NxSplit", &i_NxSplit,1);
   AddEntry("nIter", &i_nIter,1); // number of iterations in MoveLayer
   AddEntry("Hy", &f_Hy);
   AddEntry("Hz", &f_Hz);
   AddEntry("Xlength", &f_Xlength);
   AddEntry("Ylength", &f_Ylength);
   AddEntry("Zlength", &f_Zlength);
   AddEntry("Ts", &f_Ts, 1.);
   AddEntry("TsIni", &f_TsIni, 1.);
   AddEntry("TimeIni", &f_TimeIni, 0.);
   AddEntry("Nspecies", &i_Nsorts, 1);
   AddEntry("Npulses", &i_Npulses, 0);
   AddEntry("NMovieFrames", &i_NMovieFrames, 0);
   AddEntry("NMovieFramesH5", &i_NMovieFramesH5, 0);
   AddEntry("Wavelength", &f_Wavelength, 0.001);

   AddEntry("l_Xsize",&l_Xsize);
   AddEntry("l_Ysize",&l_Ysize);
   AddEntry("l_Zsize",&l_Zsize);
   AddEntry("l_dX",&l_dX);
   AddEntry("l_dY",&l_dY);
   AddEntry("l_dZ",&l_dZ);

   AddEntry("f_Ts2Hx",&f_Ts2Hx);
   AddEntry("f_Ts2Hy",&f_Ts2Hy);
   AddEntry("f_Ts2Hz",&f_Ts2Hz);

   AddEntry("f_BTs2Hx",&f_BTs2Hx);
   AddEntry("f_BTs2Hy",&f_BTs2Hy);
   AddEntry("f_BTs2Hz",&f_BTs2Hz);

   AddEntry("BxExternal",&f_BxExternal, 0.);
   AddEntry("ByExternal",&f_ByExternal, 0.);
   AddEntry("BzExternal",&f_BzExternal, 0.);

   AddEntry("MaxwellSolver",&i_MaxwellSolver, 1); // 0 - Yee, 1 - NDF
   AddEntry("Hybrid",&i_Hybrid, 0); // should we use a hybrid solver?

   p_File = NULL;

   sprintf(str_FileName,"%s//v%d.log",str_LogDirectory,rank);
   out_Flog.open(str_FileName, ios::out);
   //  out_Fig8.open("fig8.dat", ios::out);

   if (rank==NIL)
   {
      p_File = fopen(infile,"rt");
      if (p_File == NULL)
      {
         out_Flog << "Error. Domain: No such file " << infile << "\n";
         exit (-1);
      }
   }

   if (p_File)
   {
      rewind(p_File);
      read(p_File);
   }

#ifdef V_MPI

   CBuffer *buf = new CBuffer;
   buf->reset();
   pack_nls(buf);
   BroadCast(buf);
   if (rank) unpack_nls(buf);
   delete buf;

#endif

   p_MPP = new Partition("MPP_partition",p_File);
   p_MPP->Init();

   if (f_Hx<=0. || f_Xlength<=f_Hx ||
      f_Hy<=0. || f_Ylength<=f_Hy ||
      f_Hz<=0. || f_Zlength<=f_Hz) {
         cout << "Wrong step or domain size! \n";
         exit( -1);
   }

   if (i_NxSplit<1) i_NxSplit = 1;
   f_HxSplit = f_Hx/i_NxSplit;
   if (i_nIter<0) i_nIter = 0;
   f_Ts2Hx = f_Ts/f_Hx;
   f_Ts2Hy = f_Ts/f_Hy;
   f_Ts2Hz = f_Ts/f_Hz;
   f_BTs2Hx = double(f_Ts2Hx/2.);
   f_BTs2Hy = double(f_Ts2Hy/2.);
   f_BTs2Hz = double(f_Ts2Hz/2.);

   l_Xsize = long(f_Xlength/f_Hx)/GetSize()+1;
#ifdef CUDA_WRAP_PARALLEL_DEBUG   
   printf("lXsize %d size %d \n",l_Xsize,GetSize());
#endif   
   
   f_Xlength = l_Xsize*f_Hx;
   l_dX = 2;
   l_Ysize = long(f_Ylength/f_Hy)+1;
   f_Ylength = l_Ysize*f_Hy;
   l_dY = 2;
   l_Zsize = long(f_Zlength/f_Hz)+1;
   f_Zlength = l_Zsize*f_Hz;
   l_dZ = 2;

   p_CGS = new UnitsCGS;
   p_Cntrl = new Controls("Controls",p_File);

   if (p_Cntrl->Reload()) {
      int np = nPE();
      long ldump = Load(np);

   } else {

      if (i_Npulses > 0) {
         pa_Pulses = new Pulse*[i_Npulses];
      } else {
         pa_Pulses = NULL;
      }

      if (i_Nsorts < 1) i_Nsorts = 1;
      pa_Species = new Specie*[i_Nsorts];
      pa_Species[0] = new Specie("Electrons",p_File);

      out_Flog << "Electron specie created"<<endl;

      char name[128];
      for (i=1; i<i_Nsorts; i++)
      {
         sprintf(name,"Specie%d",i);
         pa_Species[i] = new IonSpecie(i, name, p_File);
      }

      if (i_NMovieFrames > 0) {
         p_MovieFrame = new MovieFrame("Movie",p_File,i_NMovieFrames);
      } else {
         p_MovieFrame = NULL;
      }

      if (i_NMovieFramesH5 > 0) {
         p_MovieFrameH5 = new MovieFrame("MovieHDF5",p_File,i_NMovieFramesH5);
      } else {
         p_MovieFrameH5 = NULL;
      }

      l_NProcessed = 0;

      FILE *ftmp = NULL;
      ftmp = p_File;
      if (GetmyPE()==0 && ftmp==NULL) exit(-20);
      if (GetmyPE()!=0 && ftmp!=NULL) exit(-21);

      p_BndXm = new Boundary("Boundary_Xm",ftmp,XDIR+MDIR,p_MPP->p_XmPE);
      p_BndXp = new Boundary("Boundary_Xp",ftmp,XDIR+PDIR,p_MPP->p_XpPE);
      p_BndYm = new Boundary("Boundary_Ym",ftmp,YDIR+MDIR,p_MPP->p_YmPE);
      p_BndYp = new Boundary("Boundary_Yp",ftmp,YDIR+PDIR,p_MPP->p_YpPE);
      p_BndZm = new Boundary("Boundary_Zm",ftmp,ZDIR+MDIR,p_MPP->p_ZmPE);
      p_BndZp = new Boundary("Boundary_Zp",ftmp,ZDIR+PDIR,p_MPP->p_ZpPE);

      p_Synchrotron = new Synchrotron("Synchrotron",p_File);

      Cell::sf_DimFields = 1.;
      Cell::sf_DimDens = 1.;
      Cell::sf_DimCurr = 1.; 

      long mx = l_Xsize/p_MPP->GetXpart();
      long ofX = mx*iPE();
      long my = l_Ysize/p_MPP->GetYpart();

      long ofY = my*jPE();
      long mz = l_Zsize/p_MPP->GetZpart();
      long ofZ = mz*kPE();

      p_M = new Mesh(mx, ofX, my, ofY, mz, ofZ);

      for (i=0; i<i_Npulses; i++)
      {
         sprintf(name,"Pulse%d",i);
         pa_Pulses[i] = new Pulse(name, p_File);
      }
      p_M->MakeIt();
      if (nPE()==0) {
	cout << "Mesh is prepared on PE " << nPE() << endl; 
      };
      for (i=0; i<i_Npulses; i++) p_M->InitPulse(pa_Pulses[i]);
      if (nPE()==0) {
	cout << "Pulses created on PE " << nPE() << endl; 
      };
      Exchange(SPACK_F);
      if (nPE()==0) {
	cout << "Fields exchanged " << nPE() << endl; 
      };
   }

#ifdef X_ACCESS 
   plot = new Plot(file);
#endif

   printf("in domain rank %d \n",GetRank());
   sprintf(str_MovieFile,"%s//mov%d.dat",str_MovieDirectory,GetRank());

   if(p_M->MovieWriteEnabled())
      p_MovieFile=fopen(str_MovieFile,"wb");

   if (nPE()==0) {
     cout << "Movie file created " << str_MovieFile << endl; 
   };

   p_PoyntingFile = NULL;
   if (nPE()==0) {
    sprintf(str_FileName,"%s//poynting.dat",str_MovieDirectory);

//    p_PoyntingFile = fopen(str_FileName,"wt");

     cout << "Poynting file created " << str_FileName << endl; 
   };

   out_Flog << "Domain is created"<<endl;
}

//---------------------------- Domain::MPP -----------------------
int Domain::Xpartition(void)   {return p_MPP->GetXpart();};
int Domain::Ypartition(void)   {return p_MPP->GetYpart();};
int Domain::Zpartition(void)   {return p_MPP->GetZpart();};
int Domain::nPEs(void)   {return p_MPP->GetnPEs();};
int Domain::nPE(void)   {return p_MPP->GetnPE();};
int Domain::iPE(void)   {return p_MPP->GetiPE();};
int Domain::jPE(void)   {return p_MPP->GetjPE();};
int Domain::kPE(void)   {return p_MPP->GetkPE();};

//---------------------------- Domain::GetPulse -----------------------
Pulse* Domain::GetPulse(int ipulse)
{
   return pa_Pulses[ipulse];
}

//---------------------------- Domain::BroadCast -----------------------
void Domain::BroadCast(CBuffer *b) 
{
#ifdef V_MPI
   int root = 0;
#ifdef _DEBUG
   printf("Start BroadCast\nlen = %d\n\n", b->getlen());
#endif

   int ierr = MPI_Bcast(b->getbuf(), b->getlen(),
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
      out_Flog << "Unknown error Domain::BroadCast MPI_Bcast. ierr = " << ierr << ", PE = " << GetmyPE() << endl;				
   };

#ifdef _DEBUG
   printf("End BroadCast\n\n");
#endif
#endif
}

//---------------------------- Domain::GetSpecie -----------------------
Specie* Domain::GetSpecie(int sort) 
{
   if (sort <0 || sort >=i_Nsorts) {
      out_Flog << "GetSpecie: error sort="<<sort<<endl;
      out_Flog.flush();
      exit(sort);
   }
   return pa_Species[sort];
}
//---------------------------- Domain::Edges -----------------------
int Domain::XmEdge(void){return p_MPP->XmEdge();};
int Domain::XpEdge(void){return p_MPP->XpEdge();};

int Domain::YmEdge(void){return p_MPP->YmEdge();};
int Domain::YpEdge(void){return p_MPP->YpEdge();};

int Domain::ZmEdge(void){return p_MPP->ZmEdge();};
int Domain::ZpEdge(void){return p_MPP->ZpEdge();};

//---------------------------- Domain::GetmyPE -----------------------
int Domain::GetmyPE(void) {return p_MPP->GetnPE();};

//---------------------------- Domain::Add2Specie -----------------------
long       Domain::Add2Specie(int sort) 
{
   if (sort <0 || sort >= i_Nsorts) {
      out_Flog << "Add2Specie: error sort="<<sort<<endl;
      out_Flog.flush();
      exit(sort);
   }
   return pa_Species[sort]->Add();
}


//---------------------------- Domain::RemoveFromSpecie -----------------------
long       Domain::RemoveFromSpecie(int sort) 
{
   if (sort <0 || sort >= i_Nsorts) exit(-1);
   return pa_Species[sort]->Remove();
}

//---------------------------- Domain::GetPhase -----------------------
double Domain::GetPhase()
{
   return p_Cntrl->GetPhase();
}

//---------------------------- Domain::GetBufMPP -----------------------
CBuffer*	 Domain::GetBufMPP()
{
   return p_MPP->GetBuf();
};

//---------------------------- Domain::ResetBufMPP -----------------------
void			 Domain::ResetBufMPP()
{
   p_MPP->GetBuf()->reset();
};
//--------------------------------  double GetTs()------------------------
double Domain::GetTs() {
   if (p_Cntrl->GetPhase() < f_TimeIni) {
      return f_TsIni;
   } else {
      return f_Ts;
   };
}


//---------------------------- Domain::~Domain() -----------------------
Domain::~Domain()
{
#ifdef X_ACCESS
   delete plot;
#endif
   if(out_Flog)
      out_Flog.close();

   if(p_File)
      fclose(p_File);

   if(p_MovieFile)
      fclose(p_MovieFile);

   for (int j=0; j<i_Npulses; j++)
   {
      delete pa_Pulses[j];
   };

   delete p_M;
   delete p_BndZp;
   delete p_BndZm;
   delete p_BndYp;
   delete p_BndYm;
   delete p_BndXp;
   delete p_BndXm;

   delete p_MovieFrame;
   delete p_MovieFrameH5;

   for (int k=0; k<i_Nsorts; k++)
   {
      delete pa_Species[k];
   };

   delete p_Cntrl;
   delete p_CGS;
   delete p_MPP;
};


