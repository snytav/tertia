#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "vlpl3d.h"
//#define _DEBUG

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
    p_D->out_Flog << "---------- Start recv1 what=" << int(what)
		  << " from " <<fromN 
		  << " msgtag=" << msgtag+1024 << endl; p_D->out_Flog.flush();
#endif
    MPI_Request req1, req, req2;
    int ierr = MPI_Irecv(&nsend, sizeof(int), MPI_BYTE,
    			fromN, msgtag+1024, MPI_COMM_WORLD, &req1);
    MPI_Wait(&req1,&status);

    //    int ierr = MPI_Recv(&nsend, 1, MPI_LONG,
    //			fromN, msgtag+1, MPI_COMM_WORLD, &status);
#ifdef _DEBUG
    p_D->out_Flog << "finished recv1 nsend = " << nsend << endl; p_D->out_Flog.flush();
#endif
    switch (ierr)
      {
      case MPI_SUCCESS:
	// No error
	break;
      case MPI_ERR_COMM:
	p_D->out_Flog << "Invalid communicator. A common error is to use a null communicator in a call. Error Partition:recv1 ierr = " << ierr  << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_COUNT:
	p_D->out_Flog << "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. Error Partition:recv1 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_TYPE:
	p_D->out_Flog << "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). Error Partition:recv1 ierr = " << ierr << endl; p_D->out_Flog.flush();  					
	break;
      case MPI_ERR_RANK:
	p_D->out_Flog << "Invalid source or destination rank. Ranks must be between zero and the size of the communicator minus one; ranks in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE. Error Partition:recv1 ierr = " << ierr << endl; p_D->out_Flog.flush();  					  					
	break;
      case MPI_ERR_TAG:
	p_D->out_Flog << "Invalid tag argument. Tags must be non-negative; tags in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG. The largest tag value is available through the the attribute MPI_TAG_UB. Error Partition:recv1 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      default:
	p_D->out_Flog << "Unknown error Partition:recv1. ierr = " << ierr << endl; p_D->out_Flog.flush();				
      };

			
    //    if (ierr) p_D->out_Flog << "Error Partition:recv1 ierr ="<<ierr<<endl; p_D->out_Flog.flush();

    //    if (nsend < p_Buf->b_len)
    p_Buf->Adjust(nsend);
    //    {
#ifdef _DEBUG
    p_D->out_Flog << "start recv2 msgtag=" << msgtag 
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif
    ierr = MPI_Irecv(p_Buf->getbuf(), nsend, MPI_BYTE, fromN, msgtag, MPI_COMM_WORLD, &req);
    MPI_Wait(&req,&status);
    //	 }
    //    else
    //    {
    //	   p_D->out_Flog << "Don`t start recv2" << endl; p_D->out_Flog.flush();
    //    	ierr = nsend;
    //    }

    switch (ierr)
      {
      case MPI_SUCCESS:
	// No error
	break;
      case MPI_ERR_COMM:
	p_D->out_Flog << "Invalid communicator. A common error is to use a null communicator in a call. Error Partition:recv2 ierr = " << ierr  << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_COUNT:
	p_D->out_Flog << "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. Error Partition:recv2 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_TYPE:
	p_D->out_Flog << "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). Error Partition:recv2 ierr = " << ierr << endl; p_D->out_Flog.flush();  					
	break;
      case MPI_ERR_RANK:
	p_D->out_Flog << "Invalid source or destination rank. Ranks must be between zero and the size of the communicator minus one; ranks in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE. Error Partition:recv2 ierr = " << ierr << endl; p_D->out_Flog.flush();  					  					
	break;
      case MPI_ERR_TAG:
	p_D->out_Flog << "Invalid tag argument. Tags must be non-negative; tags in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG. The largest tag value is available through the the attribute MPI_TAG_UB. Error Partition:recv2 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      default:
	p_D->out_Flog << "Unknown error Partition:recv2. ierr = " << ierr << endl; p_D->out_Flog.flush();				
      };


    //    if (ierr) p_D->out_Flog << "Error Partition:recv2 ierr ="<<ierr<<endl; p_D->out_Flog.flush();

    //    p_D->out_Flog << "Partition: received "<<nsend<<" bytes from "
    //		 <<fromN<<endl; p_D->out_Flog.flush();
    p_Buf->setpos(nsend);
#ifdef _DEBUG
    p_D->out_Flog << "end recv2 nsend=" << nsend 
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif
    //    MPI_Barrier(MPI_COMM_WORLD);
    long bufCRCrec=0;
#ifdef _DEBUG
    p_D->out_Flog << "Starting recv CRC, msgtag=" << msgtag+2048  
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif

    ierr = MPI_Irecv(&bufCRCrec, sizeof(long), MPI_BYTE,
		    fromN, msgtag+2048, MPI_COMM_WORLD, &req2);
    MPI_Wait(&req2,&status);

#ifdef _DEBUG
    p_D->out_Flog << "Received CRC=" << bufCRCrec  
		  << " from " << fromN << endl; p_D->out_Flog.flush();
#endif
    long bufCRC = p_Buf->FindCRC(nsend);
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

#ifdef _DEBUG
    p_D->out_Flog << "start send1 nsend = " << nsend 
		  << " msgtag="<< msgtag+1024 << " to "<< toN<< endl; 
    p_D->out_Flog.flush();
#endif
    MPI_Request req1, req, req2;
    ierr = MPI_Isend(&nsend, sizeof(int), MPI_BYTE,
		     toN, msgtag+1024, MPI_COMM_WORLD, &req1);
    MPI_Request_free(&req1);
#ifdef _DEBUG
    p_D->out_Flog << "end send1 nsend = " << nsend
		  << " msgtag="<< msgtag+1024 << " to "<< toN<< endl; 
    p_D->out_Flog.flush();
#endif		
		
    switch (ierr)
      {
      case MPI_SUCCESS:
	// No error
	break;
      case MPI_ERR_COMM:
	p_D->out_Flog << "Invalid communicator. A common error is to use a null communicator in a call. Error Partition:send1 ierr = " << ierr  << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_COUNT:
	p_D->out_Flog << "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. Error Partition:send1 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_TYPE:
	p_D->out_Flog << "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). Error Partition:send1 ierr = " << ierr << endl; p_D->out_Flog.flush();  					
	break;
      case MPI_ERR_RANK:
	p_D->out_Flog << "Invalid source or destination rank. Ranks must be between zero and the size of the communicator minus one; ranks in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE. Error Partition:send1 ierr = " << ierr << endl; p_D->out_Flog.flush();  					  					
	break;
      case MPI_ERR_TAG:
	p_D->out_Flog << "Invalid tag argument. Tags must be non-negative; tags in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG. The largest tag value is available through the the attribute MPI_TAG_UB. Error Partition:send1 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      default:
	p_D->out_Flog << "Unknown error Partition:send1. ierr = " << ierr << endl; p_D->out_Flog.flush();				
      };
    /*
      MPI_Buffer_detach(p_b, &buf_size);		
      free(p_b);

      buf_size = p_Buf->b_pos + MPI_BSEND_OVERHEAD;
      p_b = (char*)malloc(buf_size);
      err_buf_att = MPI_Buffer_attach( p_b, buf_size);
      switch (err_buf_att)
      {
      case MPI_SUCCESS:
      // No error
      break;
      case MPI_ERR_BUFFER:
      p_D->out_Flog << "Invalid buffer pointer. Usually a null buffer where one is not valid. err_buf_att = " << err_buf_att << endl; p_D->out_Flog.flush();
      break;
      case MPI_ERR_INTERN:
      p_D->out_Flog << "An internal error has been detected. This is fatal. Please send a bug report to mpi-bugs@mcs.anl.gov. err_buf_att = " << err_buf_att << endl; p_D->out_Flog.flush();
      break;
      default:
      p_D->out_Flog << "Unknown error. Domain::BroadCast MPI_Buffer_attach. err_buf_att = " << err_buf_att << endl; p_D->out_Flog.flush();				
      };
    */		     		
    //    if (ierr) p_D->out_Flog << "Error Partition:send1 ierr ="<<ierr<<endl; p_D->out_Flog.flush();
#ifdef _DEBUG
    p_D->out_Flog << "start send2 p_Buf->b_pos = " << p_Buf->getpos() 
		  << " msgtag="<< msgtag << " to "<< toN 
		  << endl; p_D->out_Flog.flush();
#endif
	
    ierr = MPI_Isend(p_Buf->getbuf(), p_Buf->getpos(), MPI_BYTE,
		     toN, msgtag, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
#ifdef _DEBUG
    p_D->out_Flog << "end send2 p_Buf->b_pos = " << p_Buf->getpos() 
		  << " msgtag="<< msgtag << " to "<< toN 
		  << endl; p_D->out_Flog.flush();
#endif

    switch (ierr)
      {
      case MPI_SUCCESS:
	// No error
	break;
      case MPI_ERR_COMM:
	p_D->out_Flog << "Invalid communicator. A common error is to use a null communicator in a call. Error Partition:send2 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_COUNT:
	p_D->out_Flog << "Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. Error Partition:send2 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      case MPI_ERR_TYPE:
	p_D->out_Flog << "Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). Error Partition:send2 ierr = " << ierr << endl; p_D->out_Flog.flush();  					
	break;
      case MPI_ERR_RANK:
	p_D->out_Flog << "Invalid source or destination rank. Ranks must be between zero and the size of the communicator minus one; ranks in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_SOURCE. Error Partition:send2 ierr = " << ierr << endl; p_D->out_Flog.flush();  					  					
	break;
      case MPI_ERR_TAG:
	p_D->out_Flog << "Invalid tag argument. Tags must be non-negative; tags in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG. The largest tag value is available through the the attribute MPI_TAG_UB. Error Partition:send2 ierr = " << ierr << endl; p_D->out_Flog.flush();
	break;
      default:
	p_D->out_Flog << "Unknown error Partition:send2. ierr = " << ierr << endl; p_D->out_Flog.flush();				
      };

#ifdef _DEBUG
    p_D->out_Flog << "Sending CRC=" << bufCRC << " to "<< toN << " msgtag=" 
		  << msgtag+2048 << endl; p_D->out_Flog.flush();			    p_D->out_Flog << "end send2" << endl; 
    p_D->out_Flog.flush();
#endif			 			
    ierr = MPI_Isend(&bufCRC, sizeof(long), MPI_BYTE,
		     toN, msgtag+2048, MPI_COMM_WORLD, &req2);
    MPI_Request_free(&req2);

#ifdef _DEBUG
    p_D->out_Flog << "Sent CRC=" << bufCRC << " to "<< toN << " msgtag=" 
		  << msgtag+2048 << endl; p_D->out_Flog.flush();			
    p_D->out_Flog << "end send2" << endl; p_D->out_Flog.flush();			
#endif			 			
    //    if (ierr) p_D->out_Flog << "Error Partition:send2 ierr ="<<ierr<<endl; p_D->out_Flog.flush();
    /*
      MPI_Buffer_detach(p_b, &buf_size);
      free(p_b);
    */
    //    p_D->out_Flog << "Partition: sent "<<nsend<<" bytes to "
    //		 <<toN<<endl; p_D->out_Flog.flush();
    //MPI_Barrier(MPI_COMM_WORLD);
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
