#include <stdio.h>
#include <stdlib.h>


#include "vlpl3d.h"

// #define _DEBUG
static long lpcnt = 0;
static char where = 0;

//---Domain:: --------------------->

void Domain::Exchange(int what)
{
   if ( GetMPP()->GetXpart() == 2 )	{
      //		out_Flog << "Xpart == 2 : Ordered exchange mode.\n";
      if ( GetMPP()->GetnPE() < p_BndXm->MatePE()->GetnPE() )	{
         p_M->Send(p_BndXm, what);
         p_M->Send(p_BndXp,what);
         p_M->Receive(p_BndXp,what);
         p_M->Receive(p_BndXm,what);
      }	else	{
         p_M->Receive(p_BndXp,what);
         p_M->Receive(p_BndXm,what);
         p_M->Send(p_BndXm, what);
         p_M->Send(p_BndXp,what);
      }
   }
   else {
      //		out_Flog << "Xpart != 2 : Normal exchange mode.\n";
      //		out_Flog << "BndXm : "; out_Flog.flush();
      p_M->Send(p_BndXm, what);
      p_M->Receive(p_BndXp,what);
      p_M->Send(p_BndXp,what);
      p_M->Receive(p_BndXm,what);
   }

   if ( GetMPP()->GetYpart() == 2 )	{
      //		out_Flog << "Ypart == 2 : Ordered exchange mode.\n";
      if ( GetMPP()->GetnPE() < p_BndYm->MatePE()->GetnPE() )	{
         p_M->Send(p_BndYm, what);
         p_M->Send(p_BndYp,what);
         p_M->Receive(p_BndYp,what);
         p_M->Receive(p_BndYm,what);
      }	else	{
         p_M->Receive(p_BndYp,what);
         p_M->Receive(p_BndYm,what);
         p_M->Send(p_BndYm, what);
         p_M->Send(p_BndYp,what);
      }
   }
   else {
      //		out_Flog << "Ypart != 2 : Normal exchange mode.\n";
      //		out_Flog << "BndYm : "; out_Flog.flush();
      p_M->Send(p_BndYm, what);
      p_M->Receive(p_BndYp,what);
      p_M->Send(p_BndYp,what);
      p_M->Receive(p_BndYm,what);
   }

   if ( GetMPP()->GetZpart() == 2 )	{
      //		out_Flog << "Zpart == 2 : Ordered exchange mode.\n";
      if ( GetMPP()->GetnPE() < p_BndZm->MatePE()->GetnPE() )	{
         p_M->Send(p_BndZm, what);
         p_M->Send(p_BndZp,what);
         p_M->Receive(p_BndZp,what);
         p_M->Receive(p_BndZm,what);
      }	else	{
         p_M->Receive(p_BndZp,what);
         p_M->Receive(p_BndZm,what);
         p_M->Send(p_BndZm, what);
         p_M->Send(p_BndZp,what);
      }
   }
   else {
      //		out_Flog << "Zpart != 2 : Normal exchange mode.\n";
      //		out_Flog << "BndZm : "; out_Flog.flush();
      p_M->Send(p_BndZm, what);
      p_M->Receive(p_BndZp,what);
      p_M->Send(p_BndZp,what);
      p_M->Receive(p_BndZm,what);
   }

#ifdef _DEBUG
   out_Flog << "End recv p_BndZm" << endl;
   out_Flog << "End receiving" << endl;
   out_Flog.flush();
#endif
}

//---Mesh::Send --------------------->

void Mesh::Send(Boundary *bnd, int what)
{
   if (what & SPACK_P) SetCellNumbers();

   if (what == SNOPACK)
      return;

   int fin = FINISH;

   //  CBuffer *buf = domain()->GetMPP()->p_Buf;
   CBuffer *buf = domain()->GetBufMPP();
   buf->reset();
   long length = buf->getlen();
   long pos = buf->getpos();
   long out = buf->getout();
#ifdef _DEBUG
   domain()->out_Flog << "Mesh::Send  p_buf\n";
   domain()->out_Flog.flush();
#endif
   int fcnd = bnd->GetFcnd();
   int pcnd = bnd->GetPcnd();
   double x=0., y=0., z=0.;

   where = bnd->Where();
   if (where & XDIR)
   {
#ifdef _DEBUG
      domain()->out_Flog << "Mesh::Send  where & XDIR\n";
      domain()->out_Flog.flush();
#endif
      if (what & SPACK_F || what & SPACK_H ) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++)
            {
               long i = 0;
               long i1 = 0;
               if (where & MDIR) {
                  i=0;
               } else {
                  i=l_Mx-1;
               }

               if (what & SPACK_E)
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               if (what & SPACK_B)
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               if (what & SPACK_H)
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);

               if (where & MDIR) {
                  i=1; 
               }
               else {
                  i=l_Mx-2;
               }

               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H)
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);
            }
         }
      }

      if (what & SPACK_J) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++) {
               long i = 0;
               if (where & MDIR) {
                  i=-1;
               } else { 
                  i=l_Mx;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J );

               if (where & MDIR) {
                  i=-2;
               } else {
                  i=l_Mx+1;
               }

               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J);
            }
         }
      }

      if (what & SPACK_JB) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++) {
               long i = 0;
               if (where & MDIR) {
                  i=-1;
               } else { 
                  i=l_Mx;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB );

               if (where & MDIR) {
                  i=-2;
               } else {
                  i=l_Mx+1;
               }

               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB);
            }
         }
      }
      if (what & SPACK_P) {
         length = buf->getlen();
         pos = buf->getpos();
         long i = 0;
         if (where & MDIR) {
            i=-1;
         } else {
            i=l_Mx;
         }

         x = FullI(i);
         for (long k=-2; k<l_Mz+2; k++) {
            z = FullK(k);
            for (long j=-2; j<l_My+2; j++) {
               y = FullJ(j);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackP(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }

         *buf << fin;
         pos = buf->getpos();
         length = buf->getlen();
      }
      if (what & SPACK_PB) {
         length = buf->getlen();
         pos = buf->getpos();
         long i = 0;
         if (where & MDIR) {
            i=-1;
         } else {
            i=l_Mx;
         }

         x = FullI(i);
         for (long k=-2; k<l_Mz+2; k++) {
            z = FullK(k);
            for (long j=-2; j<l_My+2; j++) {
               y = FullJ(j);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackPB(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }

         *buf << fin;
         pos = buf->getpos();
         length = buf->getlen();
      }
   } 
   else if (where & YDIR)
   {
#ifdef _DEBUG
      domain()->out_Flog << "Mesh::Send  where & YDIR\n";
      domain()->out_Flog.flush();
#endif
      if (what & SPACK_F || what & SPACK_H ) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=0;
               } else {
                  j=l_My-1;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               }

               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);
               }

               if (where & MDIR) {
                  j=1;
               } else {
                  j=l_My-2;
               }

               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);
               }
            }
         }
      }

      if (what & SPACK_J) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=-1;
               } else {
                  j=l_My;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J);
               if (where & MDIR) {
                  j=-2;
               } else {
                  j=l_My+1;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J);
            }
         }
      }

      if (what & SPACK_JB) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=-1;
               } else {
                  j=l_My;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB);
               if (where & MDIR) {
                  j=-2;
               } else {
                  j=l_My+1;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB);
            }
         }
      }

      if (what & SPACK_P) {
         long j = 0;
         if (where & MDIR) {
            j=-1;
         } else {
            j=l_My;
         }
         y = FullJ(j);
         for (long k=-2; k<l_Mz+2; k++) {
            z = FullK(k);
            for (long i=-2; i<l_Mx+2; i++) {
               x = FullI(i);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackP(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }
         *buf << fin; 
      }

      if (what & SPACK_PB) {
         long j = 0;
         if (where & MDIR) {
            j=-1;
         } else {
            j=l_My;
         }
         y = FullJ(j);
         for (long k=-2; k<l_Mz+2; k++) {
            z = FullK(k);
            for (long i=-2; i<l_Mx+2; i++) {
               x = FullI(i);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackPB(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }
         *buf << fin; 
      }
   } else if (where & ZDIR) {
#ifdef _DEBUG
      domain()->out_Flog << "Mesh::Send  where & ZDIR\n";
      domain()->out_Flog.flush();
#endif
      if (what & SPACK_F || what & SPACK_H ){
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=0;
               } else {
                  k=l_Mz-1;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);
               }

               if (where & MDIR) {
                  k=1;
               } else {
                  k=l_Mz-2;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_H);
               }
            }
         }
      }

      if (what & SPACK_J) {
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=-1;
               }  else {
                  k=l_Mz;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J);
               if (where & MDIR) {
                  k=-2;
               } else { 
                  k=l_Mz+1;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_J);
            }
         }
      }

      if (what & SPACK_JB) {
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=-1;
               }  else {
                  k=l_Mz;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB);
               if (where & MDIR) {
                  k=-2;
               } else { 
                  k=l_Mz+1;
               }
               p_CellArray[GetN(i,j,k)].PackF(buf,fcnd,SPACK_JB);
            }
         }
      }

      if (what & SPACK_P) {
         long k = 0;
         if (where & MDIR) {
            k=-1;
         } else {
            k=l_Mz;
         }
         z = FullK(k);
         for (long j=-2; j<l_My+2; j++) {
            y = FullJ(j);
            for (long i=-2; i<l_Mx+2; i++) {
               x = FullI(i);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackP(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }
         *buf << fin;
      }

      if (what & SPACK_PB) {
         long k = 0;
         if (where & MDIR) {
            k=-1;
         } else {
            k=l_Mz;
         }
         z = FullK(k);
         for (long j=-2; j<l_My+2; j++) {
            y = FullJ(j);
            for (long i=-2; i<l_Mx+2; i++) {
               x = FullI(i);
               lpcnt+=p_CellArray[GetN(i,j,k)].PackPB(bnd,buf,x,y,z,pcnd,ALLSORTS);
            }
         }
         *buf << fin;
      }
   }  else {
#ifdef _DEBUG
      domain()->out_Flog << "Mesh::Send  exit(where)\n";
      domain()->out_Flog.flush();
#endif
      exit(where);
   }

   Processor *pe = bnd->MatePE();
#ifdef _DEBUG
   domain()->out_Flog << "Mesh::Send  domain()->GetMPP()->Send(pe,what);\n" ;
   domain()->out_Flog.flush();
#endif
   domain()->GetMPP()->Send(pe,what);
}

//---Mesh::Receive --------------------->

void Mesh::Receive(Boundary *bnd, int what)
{
   if (what == SNOPACK) return;

   Processor *pe = bnd->MatePE();
   domain()->GetMPP()->Receive(pe,what);

   CBuffer *buf = domain()->GetMPP()->p_Buf;
   int fcnd = bnd->GetFcnd();
   int pcnd = bnd->GetPcnd();
   double x=0., y=0., z=0.;

   where = bnd->Where();
   if (where & XDIR) {
      if (what & SPACK_F || what & SPACK_H) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++) {
               long i = 0;
               if (where & MDIR) {
                  i=-1; 
               } else {
                  i=l_Mx;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }

               if (where & MDIR) {
                  i=-2; 
               } else {
                  i=l_Mx+1;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }
            }
         }
      }
      if (what & SPACK_J) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++) {
               long i = 0;
               if (where & MDIR) {
                  i=0; 
               } else {
                  i=l_Mx-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
               if (where & MDIR) {
                  i=1;
               } else {
                  i=l_Mx-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
            }
         }
      }
      if (what & SPACK_JB) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long j=-2; j<l_My+2; j++) {
               long i = 0;
               if (where & MDIR) {
                  i=0; 
               } else {
                  i=l_Mx-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
               if (where & MDIR) {
                  i=1;
               } else {
                  i=l_Mx-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
            }
         }
      }
      if (what & SPACK_P) {
         UnPackP(buf,pcnd);
      }
      if (what & SPACK_PB) {
         UnPackPB(buf,pcnd);
      }
   } else if (where & YDIR) {
      if (what & SPACK_F  || what & SPACK_H) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=-1;
               } else {
                  j=l_My;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }

               if (where & MDIR) {
                  j=-2;
               } else {
                  j=l_My+1;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }
            }
         }
      }
      if (what & SPACK_J) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=0;
               } else {
                  j=l_My-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
               if (where & MDIR) {
                  j=1;
               } else {
                  j=l_My-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
            }
         }
      }
      if (what & SPACK_JB) {
         for (long k=-2; k<l_Mz+2; k++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long j = 0;
               if (where & MDIR) {
                  j=0;
               } else {
                  j=l_My-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
               if (where & MDIR) {
                  j=1;
               } else {
                  j=l_My-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
            }
         }
      }
      if (what & SPACK_P) {
         UnPackP(buf,pcnd);
      }
      if (what & SPACK_PB) {
         UnPackPB(buf,pcnd);
      }
   } else if (where & ZDIR) {
      if (what & SPACK_F || what & SPACK_H) {
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=-1;
               } else {
                  k=l_Mz;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }

               if (where & MDIR) {
                  k=-2;
               } else {
                  k=l_Mz+1;
               }
               if (what & SPACK_E) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_E);
               }
               if (what & SPACK_B) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_B);
               }
               if (what & SPACK_H) {
                  p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_H);
               }
            }
         }
      }
      if (what & SPACK_J) {
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=0;
               } else {
                  k=l_Mz-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
               if (where & MDIR) {
                  k=1;
               } else {
                  k=l_Mz-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_J);
            }
         }
      }
      if (what & SPACK_JB) {
         for (long j=-2; j<l_My+2; j++) {
            for (long i=-2; i<l_Mx+2; i++) {
               long k = 0;
               if (where & MDIR) {
                  k=0;
               } else {
                  k=l_Mz-1;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
               if (where & MDIR) {
                  k=1;
               } else {
                  k=l_Mz-2;
               }
               p_CellArray[GetN(i,j,k)].UnPackF(buf,fcnd,SPACK_JB);
            }
         }
      }
      if (what & SPACK_P) {
         UnPackP(buf,pcnd);
      }
      if (what & SPACK_PB) {
         UnPackPB(buf,pcnd);
      }
   } else {
      exit(where);
   }
}

//---Mesh::UnPackP --------------------->

long Mesh::UnPackP(CBuffer* buf, int cnd)
{
   long i, j, k;
   double x, y, z, xc, yc, zc;
   long lcnt = 0;
   int psort=0;
   Cell *cptr=NULL;
   double weight = 0.;
   Particle *p = NULL;

   if (buf==NULL) return 0;
   while (-1) {
      *buf >> psort; 
      if (psort == FINISH) break;
      if (psort == 0) { //Electron
         p = new Electron();
      } else {
#ifdef _DEBUG
         domain()->out_Flog << "Mesh::UnPackP psort = " << psort << endl;
         domain()->out_Flog.flush();
#endif
         p = new Ion(NULL, psort);
      }
      lcnt++;
      p->UnPack(buf);

      long lccc = p->l_Cell;
      i = GetI_from_CellNumber(lccc);
      j = GetJ_from_CellNumber(lccc);
      k = GetK_from_CellNumber(lccc);

      p->GetX(x,y,z);
      if (x<0||x>1 || y<0||y>1 || z<0||z>1) {
         domain()->out_Flog << "Wrong UnPack "<<lpcnt<<" particles: x="
            <<x<<" y="<<y<<" z="<<z<<"\n";
         domain()->out_Flog.flush();
         exit(-11);
      }

      if (where & XDIR ) {
         while (i<0)    i += l_Mx;
         while (i>=l_Mx) i -= l_Mx;
         if (where & MDIR) {
            if (i>0) {
               domain()->out_Flog << "Wrong UnPackP X MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (i<l_Mx-1) {
               domain()->out_Flog << "Wrong UnPackP X PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      } else if (where & YDIR ) {
         while (j<0)    j += l_My;
         while (j>=l_My) j -= l_My;
         if (where & MDIR) {
            if (j>0) {
               domain()->out_Flog << "Wrong UnPackP Y MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (j<l_My-1) {
               domain()->out_Flog << "Wrong UnPackP Y PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      } else if (where & ZDIR ) {
         while (k<0)    k += l_Mz;
         while (k>=l_Mz) k -= l_Mz;
         if (where & MDIR) {
            if (k>0) {
               domain()->out_Flog << "Wrong UnPackP Z MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (k<l_Mz-1) {
               domain()->out_Flog << "Wrong UnPackP Z PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      }

      if (i<-l_dMx || i>l_Mx+l_dMx-1 || j<-l_dMy || j>l_My+l_dMy-1 || k<-l_dMz || k>l_Mz+l_dMz-1) {
         domain()->out_Flog << "Wrong UnPackP i="
            <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
         domain()->out_Flog.flush();
         exit(where);
      }

      if (cnd==0) {
         cptr = &GetCell(i,j,k);
         cptr->AddParticle(p);
      }   else {
         domain()->out_Flog << "Wrong UnPack particlescondition: x="
            <<x<<" y="<<y<<" z="<<z<<" cnd="<<cnd<<"\n";
         domain()->out_Flog.flush();
         exit (-15);
      }
      lcnt++;
   }
   return lcnt;
}



//---Mesh::UnPackPB --------------------->

long Mesh::UnPackPB(CBuffer* buf, int cnd)
{
   long i, j, k;
   double x, y, z, xc, yc, zc;
   long lcnt = 0;
   int psort=0;
   Cell *cptr=NULL;
   double weight = 0.;
   Particle *p = NULL;

   if (buf==NULL) return 0;
   while (-1) {
      *buf >> psort; 
      if (psort == FINISH) break;
      if (psort == 0) { //Electron
         p = new Electron();
      } else {
#ifdef _DEBUG
         domain()->out_Flog << "Mesh::UnPackPB psort = " << psort << endl;
         domain()->out_Flog.flush();
#endif
         p = new Ion(NULL, psort);
      }
      lcnt++;
      p->UnPack(buf);

      long lccc = p->l_Cell;
      i = GetI_from_CellNumber(lccc);
      j = GetJ_from_CellNumber(lccc);
      k = GetK_from_CellNumber(lccc);

      p->GetX(x,y,z);
      if (x<0||x>1 || y<0||y>1 || z<0||z>1) {
         domain()->out_Flog << "Wrong UnPack "<<lpcnt<<" particles: x="
            <<x<<" y="<<y<<" z="<<z<<"\n";
         domain()->out_Flog.flush();
         exit(-11);
      }

      if (where & XDIR ) {
         while (i<0)    i += l_Mx;
         while (i>=l_Mx) i -= l_Mx;
         if (where & MDIR) {
            if (i>0) {
               domain()->out_Flog << "Wrong UnPackP X MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (i<l_Mx-1) {
               domain()->out_Flog << "Wrong UnPackP X PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      } else if (where & YDIR ) {
         while (j<0)    j += l_My;
         while (j>=l_My) j -= l_My;
         if (where & MDIR) {
            if (j>0) {
               domain()->out_Flog << "Wrong UnPackP Y MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (j<l_My-1) {
               domain()->out_Flog << "Wrong UnPackP Y PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      } else if (where & ZDIR ) {
         while (k<0)    k += l_Mz;
         while (k>=l_Mz) k -= l_Mz;
         if (where & MDIR) {
            if (k>0) {
               domain()->out_Flog << "Wrong UnPackP Z MDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         } else {
            if (k<l_Mz-1) {
               domain()->out_Flog << "Wrong UnPackP Z PDIR i="
                  <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
               domain()->out_Flog.flush();
               exit(where);
            }
         }
      }

      if (i<-l_dMx || i>l_Mx+l_dMx-1 || j<-l_dMy || j>l_My+l_dMy-1 || k<-l_dMz || k>l_Mz+l_dMz-1) {
         domain()->out_Flog << "Wrong UnPackP i="
            <<i<<" j="<<j<<" k="<<k<<" lccc="<<lccc<<"\n";
         domain()->out_Flog.flush();
         exit(where);
      }

      if (cnd==0) {
         cptr = &GetCell(i,j,k);
         cptr->AddBeamParticle(p);
      }   else {
         domain()->out_Flog << "Wrong UnPack particlescondition: x="
            <<x<<" y="<<y<<" z="<<z<<" cnd="<<cnd<<"\n";
         domain()->out_Flog.flush();
         exit (-15);
      }
      lcnt++;
   }
   return lcnt;
}
