#include <stdio.h>
#include <stdlib.h>


#include "vlpl3d.h"
//#define _DEBUG

#ifdef _DEBUG
ofstream CRCprotokol;
#endif
static long nsrtmp = 0;

//---------------------------- Partition::Partition -----------------------
Partition::Partition (char *nm, FILE *f) : NList (nm)
{
  p_D = Domain::p_D;
  ia_X = ia_Y = ia_Z = NULL;
  fa_Loading = NULL;
  p_Buf = new CBuffer;
  p_Buftmp = new CBuffer;

#ifdef V_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &i_myPE);
  MPI_Comm_size(MPI_COMM_WORLD, &i_nPEs);
#endif
#ifndef V_MPI
  i_myPE = 0;
  i_nPEs = 1;
#endif
#ifdef _DEBUG
  char fname[128];
  sprintf(fname,"CRCexch%d.out",i_myPE);
  CRCprotokol.open(fname, ios::out);
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

double Partition::GetCellLoading(int i, int j, int k) {  
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
    p_D->out_Flog << "---------- Start recv1 what=" << int(what)
		  << " from " <<fromN 
		  << " msgtag=" << msgtag+1024 << endl; p_D->out_Flog.flush();
#endif
    int ierr = MPI_Recv(&nsend, 1, MPI_LONG,
    			fromN, msgtag+1024, MPI_COMM_WORLD, &status);
#ifdef _DEBUG
    CRCprotokol <<nsrtmp<< " Receiving "<<nsend<<" Bytes from "<<fromN<<endl;
    //    int ierr = MPI_Recv(&nsend, 1, MPI_LONG,
    //			fromN, msgtag+1, MPI_COMM_WORLD, &status);
    p_D->out_Flog << "finished recv1 nsend = " << nsend << endl; p_D->out_Flog.flush();
#endif
    p_Buf->reset();
    p_Buf->Adjust(nsend);

#ifdef _DEBUG
    p_D->out_Flog << "start recv2 msgtag=" << msgtag 
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif
    ierr = MPI_Recv(p_Buf->getbuf(), nsend, MPI_CHAR, fromN, msgtag, MPI_COMM_WORLD, &status);

#ifdef _DEBUG
    for (long i=0; i<nsend; i++) {
      CRCprotokol <<*(p_Buf->getbuf()+i);
    };
    CRCprotokol << endl;
    p_D->out_Flog << "end recv2 nsend=" << nsend 
		  << " from " << fromN << endl; p_D->out_Flog.flush();
    p_D->out_Flog << "Starting recv CRC, msgtag=" << msgtag+2048  
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif

    p_Buf->setpos(nsend);
    long bufCRCrec=0;

    ierr = MPI_Recv(&bufCRCrec, 1, MPI_LONG,
		    fromN, msgtag+2048, MPI_COMM_WORLD, &status);

    long bufCRC = p_Buf->FindCRC(nsend);

#ifdef _DEBUG
    p_D->out_Flog << "Received CRC=" << bufCRCrec  
		  << " from " << fromN << endl; p_D->out_Flog.flush();
    CRCprotokol << "bufCRC="<<bufCRC<<" bufCRCrec="<<bufCRCrec<<endl;
#endif
    if (bufCRC != bufCRCrec) {
      p_D->out_Flog << "Error Partition:recv4 bufCRCrec = "<<bufCRCrec << " counted "<<bufCRC 
		    << " msgtag=" << msgtag << " from " << fromN <<endl; p_D->out_Flog.flush();
    }
#endif
#ifndef V_MPI
    p_D->out_Flog << "Error Partition:recv3 jointPE = "<<fromN<<endl; p_D->out_Flog.flush();
#endif
  }  
  else {
  }
}

//---Partition::------------------->
void Partition::Send(Processor *ToPE, int what)
{
  nsrtmp++;
  int toN = ToPE->GetnPE();
  int myN = p_myPE->GetnPE();
#ifdef _DEBUG
  p_D->out_Flog << "Partition: Send what="<<int(what)
		<< " to "<< toN <<endl; p_D->out_Flog.flush();
#endif
  if (toN != myN) {
#ifdef V_MPI
    int msgtag = int(what);
    long nsend = p_Buf->getpos();
    long bufCRC = p_Buf->FindCRC(nsend);
    int ierr = 0;
	
    p_Buftmp->reset();
    p_Buftmp->npush(p_Buf->getbuf(), p_Buf->getpos());
    long nsendtmp = p_Buftmp->getpos();
    long bufCRCtmp = p_Buftmp->FindCRC(nsendtmp);
    if (bufCRC != bufCRCtmp) cerr << "CRC tmp Error! \n";

#ifdef _DEBUG
    p_D->out_Flog << "start send1 nsend = " << nsend 
		  << " msgtag="<< msgtag+1024 << " to "<< toN<< endl; 
    p_D->out_Flog.flush();
    p_D->out_Flog << "CRC=" << bufCRC << " CRCtmp="<< bufCRCtmp << endl;
    CRCprotokol <<nsrtmp << " Sending "<<nsend<<" Bytes to "<<toN<<endl;
#endif
 
    MPI_Request req;
    /*    ierr = MPI_Isend(&nsendtmp, 1, MPI_LONG,
		     toN, msgtag+1024, MPI_COMM_WORLD, &req);
    MPI_Request_free (&req);
    */
    ierr = MPI_Bsend(&nsendtmp, 1, MPI_LONG,
		     toN, msgtag+1024, MPI_COMM_WORLD);
#ifdef _DEBUG
    p_D->out_Flog << "end send1 nsend = " << nsend
		  << " msgtag="<< msgtag+1024 << " to "<< toN<< endl; 
    p_D->out_Flog.flush();
    p_D->out_Flog << "start send2 p_Buf->b_pos = " << p_Buf->getpos() 
		  << " msgtag="<< msgtag << " to "<< toN 
		  << endl; p_D->out_Flog.flush();
#endif
    /*
    ierr = MPI_Isend(p_Buftmp->getbuf(), nsend, MPI_CHAR,
		     toN, msgtag, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
    */
    ierr = MPI_Bsend(p_Buftmp->getbuf(), nsend, MPI_CHAR,
		     toN, msgtag, MPI_COMM_WORLD);


#ifdef _DEBUG
    for (long i=0; i<nsend; i++) {
      CRCprotokol <<*(p_Buftmp->getbuf()+i);
    };
    CRCprotokol << endl;
    p_D->out_Flog << "end send2 p_Buf->b_pos = " << p_Buf->getpos() 
		  << " msgtag="<< msgtag << " to "<< toN 
		  << endl; p_D->out_Flog.flush();
    p_D->out_Flog << "Sending CRC=" << bufCRC << " to "<< toN << " msgtag=" 
		  << msgtag+2048 << endl; p_D->out_Flog.flush();			    p_D->out_Flog << "end send2" << endl; 
    p_D->out_Flog.flush();
#endif			 			
    /*
    ierr = MPI_Isend(&bufCRC, 1, MPI_LONG,
		     toN, msgtag+2048, MPI_COMM_WORLD, &req);
    MPI_Request_free (&req);
    */
    ierr = MPI_Bsend(&bufCRC, 1, MPI_LONG,
		     toN, msgtag+2048, MPI_COMM_WORLD);

#ifdef _DEBUG
    CRCprotokol << "bufCRC="<<bufCRC<<endl;
    p_D->out_Flog << "Sent CRC=" << bufCRC << " to "<< toN << " msgtag=" 
		  << msgtag+2048 << endl; p_D->out_Flog.flush();
    p_D->out_Flog << "end send2" << endl; p_D->out_Flog.flush();
#endif			 			

#endif
#ifndef V_MPI
    p_D->out_Flog << "Error Partition:send3 jointPE = "<<toN<<endl; p_D->out_Flog.flush();
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
