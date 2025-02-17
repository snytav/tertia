#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "vlpl3d.h"
#define _DEBUG

//---------------------------- Partition::Partition -----------------------
Partition::Partition (char *nm, FILE *f) : NList (nm)
{
  p_D = Domain::p_D;
  ia_X = ia_Y = ia_Z = NULL;
  fa_Loading = NULL;
  p_Buf = new CBuffer;

#ifdef V_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &i_myPE);
  MPI_Comm_size(MPI_COMM_WORLD, &i_nPEs);
#endif
#ifndef V_MPI
  i_myPE = 0;
  i_nPEs = 1;
#endif

  AddEntry("Xpartition", &i_Xpart, 1);
  AddEntry("Ypartition", &i_Ypart, 1);
  AddEntry("Zpartition", &i_Zpart, 1);
  AddEntry("mksElectron",&f_mksElectron, 2.5);
  AddEntry("mksIon",&f_mksIon, 3.);
  AddEntry("mksCell",&f_mksCell, 1.);

  if (f)
    {
      rewind(f);
      read(f);
  }

#ifdef V_MPI
  p_Buf->reset();
  pack_nls(p_Buf);
  p_D->BroadCast(p_Buf);
  if (f==NULL)
    unpack_nls(p_Buf);
#endif

  if (i_nPEs != i_Xpart*i_Ypart*i_Zpart)
    {
      p_D->out_Flog << "Error Partition: nPEs = " << i_nPEs <<
	" i_Xpart = " << i_Xpart <<
	" i_Ypart = " << i_Ypart <<
	" i_Zpart = " << i_Zpart << "\n";
      exit(-1);
    }

  pa_PEs = new Processor[i_nPEs];
  ia_X = new int[i_nPEs];
  ia_Y = new int[i_nPEs];
  ia_Z = new int[i_nPEs];
}

//---Partition::------------------->
void Partition::Init(void)
{
  for (int k=0; k<i_Zpart; k++) 
    for (int j=0; j<i_Ypart; j++) 
      for (int i=0; i<i_Xpart; i++) {
	int n = GetnPE(i,j,k);
	pa_PEs[n].Set(this, i, j, k );
      }
  p_myPE = pa_PEs+i_myPE;
  p_XmPE = p_myPE->p_Xm;
  p_YmPE = p_myPE->p_Ym;
  p_ZmPE = p_myPE->p_Zm;
  p_XpPE = p_myPE->p_Xp;
  p_YpPE = p_myPE->p_Yp;
  p_ZpPE = p_myPE->p_Zp;
}

//---Partition::------------------->
void Partition::Xloading(void)
{
}
//---Partition::------------------->
void Partition::Yloading(int iPE)
{}

//---Partition::------------------->
void Partition::Zloading(int iPE, int jPE)
{
}

//---Partition::------------------->

float Partition::GetCellLoading(int i, int j, int k) {  
  return 1.;
}

//---Partition::------------------->
void Partition::Balance(void)
{
}

//---Partition::------------------->
void Partition::Receive(Processor *FromPE, int what)
{
  int fromN = FromPE->GetnPE();
  int myN = p_myPE->GetnPE();
  if ( fromN != myN) {
#ifdef V_MPI
    p_Buf->reset();
    MPI_Status status;
    int msgtag = int(what);
    long nsend = 0;
	
#ifdef _DEBUG
    p_D->out_Flog << "Start recv1 what=" << what << " from " <<fromN << endl;
#endif
    MPI::Request receive_rqst =  MPI::COMM_WORLD.Irecv(&nsend, sizeof(long), 
						       MPI::BYTE,
						       fromN, msgtag+1);

    receive_rqst.Wait();
#ifdef _DEBUG
    p_D->out_Flog << "end recv1 nsend = " << nsend << endl;
#endif
    p_Buf->Adjust(nsend);
    //    {
#ifdef _DEBUG
    p_D->out_Flog << "start recv2" << endl;
#endif
    MPI::Request receive_rqst1 =  MPI::COMM_WORLD.Irecv(p_Buf->getbuf(), 
							nsend, 
							MPI::BYTE,
							fromN, msgtag);
    receive_rqst1.Wait();
    p_Buf->setpos(nsend);
#ifdef _DEBUG
    p_D->out_Flog << "end recv2 nsend = " << nsend << endl;
#endif
    //    MPI_Barrier(MPI_COMM_WORLD);

#endif
#ifndef V_MPI
    p_D->out_Flog << "Error Partition:recv3 jointPE = "<<fromN<<endl;
#endif
  }  
  else {
  }
}

//---Partition::------------------->
void Partition::Send(Processor *ToPE, int what)
{
  int toN = ToPE->GetnPE();
  int myN = p_myPE->GetnPE();
#ifdef _DEBUG
  p_D->out_Flog << "Partition: Send what="<<what<<" to "<< toN <<endl;
#endif
  if (toN != myN) {
#ifdef V_MPI
    int msgtag = int(what);
    long nsend = p_Buf->getpos();
    long bufCRC = p_Buf->FindCRC(nsend);
    int ierr = 0;

#ifdef _DEBUG
    p_D->out_Flog << "start send1 nsend = " << nsend << endl;
#endif
    MPI::COMM_WORLD.Isend(&nsend, sizeof(long), MPI::BYTE,
		     toN, msgtag+1);
#ifdef _DEBUG
    p_D->out_Flog << "end send1 nsend = " << nsend << endl;
#endif		

#ifdef _DEBUG
    p_D->out_Flog << "start send2 p_Buf->b_pos = " << p_Buf->getpos() << endl;
#endif
	
    MPI::COMM_WORLD.Isend(p_Buf->getbuf(), p_Buf->getpos(), MPI::BYTE,
		     toN, msgtag);

#endif
#ifndef V_MPI
    p_D->out_Flog << "Error Partition:send3 jointPE = "<<toN<<endl;
#endif
  }  
  else {
  }
}

//---------------------------- Processor::Processor -----------------------
void Processor::Set(Partition *prt, int i, int j, int k) 
{
  p_MPP = prt;
  p_D = Domain::p_D;
  i_nPE = p_MPP->GetnPE(i,j,k); 
  i_iPE = i;       i_jPE = j;        i_kPE = k;      
  p_Xm = p_MPP->GetPE(p_MPP->XCycle(i-1),j,k);
  p_Ym = p_MPP->GetPE(i,p_MPP->YCycle(j-1),k);
  p_Zm = p_MPP->GetPE(i,j,p_MPP->ZCycle(k-1));
  p_Xp = p_MPP->GetPE(p_MPP->XCycle(i+1),j,k);
  p_Yp = p_MPP->GetPE(i,p_MPP->YCycle(j+1),k);
  p_Zp = p_MPP->GetPE(i,j,p_MPP->ZCycle(k+1));
}

//---------------------------- Processor::Edges -----------------------

int Processor::XpEdge() { return i_iPE==p_MPP->GetXpart()-1;};
int Processor::YpEdge() { return i_jPE==p_MPP->GetYpart()-1;};
int Processor::ZpEdge() { return i_kPE==p_MPP->GetZpart()-1;};

//---------------------------- Processor::GetMesh -----------------------
Mesh* Processor::GetMesh()
{
  return p_D->GetMesh();
}
