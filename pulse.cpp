#include "vlpl3d.h"
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

//---------------------------- Pulse::Pulse -----------------------
Pulse::Pulse (char *nm, FILE *f, int num) : NList (nm)
{
   i_Pulse = num;
   sprintf(str_B,"XM");  // pulse comes from XM by default

   AddEntry("a0", &f_Anorm);
   AddEntry("Xpol", &f_Xpol, 0.);
   AddEntry("Ypol", &f_Ypol, 0.);
   AddEntry("Zpol", &f_Zpol, 0.);
   AddEntry("Length", &f_Length);
   AddEntry("Ywidth", &f_Ywidth);
   AddEntry("Zwidth", &f_Zwidth);
   AddEntry("RiseTime", &f_Rise, 0.);
   AddEntry("DropTime", &f_Drop, 0.);
   AddEntry("Xcenter", &f_Xcenter, 0.);
   AddEntry("Ycenter", &f_Ycenter, 0.);
   AddEntry("Zcenter", &f_Zcenter, 0.);
   AddEntry("Yphase", &f_Yphase, 0.);
   AddEntry("Zphase", &f_Zphase, PI/2.);
   AddEntry("Kx", &f_Kx, 1.);
   AddEntry("Ky", &f_Ky, 0.);
   AddEntry("Kz", &f_Kz, 0.);
   AddEntry("FromBoundary", str_B, "%s2");
   AddEntry("Tprofile", &i_Tprofile, 0);
   AddEntry("Lprofile", &i_Lprofile, 0);

   AddEntry("f_Kxy",&f_Kxy);
   AddEntry("f_Kxyz",&f_Kxyz);
   AddEntry("f_A",&f_A);
   AddEntry("f_OmegaCGS",&f_OmegaCGS);
   AddEntry("f_NcCGS", &f_NcCGS);
   AddEntry("f_Omega", &f_Omega, 1.);
   AddEntry("f_Nc", &f_Nc, 1.);
   AddEntry("i_From",&i_From);
   AddEntry("i_Pulse",&i_Pulse);

   AddEntry("YcenterOscillationAmplitude", &f_YcenterOscillationAmplitude, 0.);
   AddEntry("YcenterOscillationPeriod", &f_YcenterOscillationPeriod, 1.);
   AddEntry("YcenterOscillationPhase", &f_YcenterOscillationPhase, 0.);

   AddEntry("ZcenterOscillationAmplitude", &f_ZcenterOscillationAmplitude, 0.);
   AddEntry("ZcenterOscillationPeriod", &f_ZcenterOscillationPeriod, 1.);
   AddEntry("ZcenterOscillationPhase", &f_ZcenterOscillationPhase,  PI/2.);

   if (f)
   {
      rewind(f);
      read(f);
   }

#ifdef V_MPI
   CBuffer *buf = domain()->GetMPP()->GetBuf();
   buf->reset();
   pack_nls(buf);
   domain()->BroadCast(buf);
   if (f==NULL)
      unpack_nls(buf);
#endif

   f_A = f_Anorm*(PI*domain()->GetTs());
   f_Kxy = sqrt(f_Kx*f_Kx+f_Ky*f_Ky);
   f_Kxyz = sqrt(f_Kxy*f_Kxy+f_Kz*f_Kz);
   if (f_Kxyz == 0.) f_Kxyz = 1.;

   switch (str_B[0])
   {
   case 'X':
   case 'x':
      if (str_B[1]=='M' || str_B[1]=='m')
      {
         i_From = TOXM;
         p_B = domain()->GetBndXm();
         domain()-> out_Flog << "i_From = " << "TOXM" << endl;
      }
      else
         if (str_B[1]=='P' || str_B[1]=='p')
         {
            i_From = TOXP;
            p_B = domain()->GetBndXp();
         }
         else
            exit(-1);
      break;

   case 'Y':
   case 'y':
      if (str_B[1]=='M' || str_B[1]=='m')
      {
         i_From = TOYM;
         p_B = domain()->GetBndYm();
      }
      else
         if (str_B[1]=='P' || str_B[1]=='p')
         {
            i_From = TOYP;
            p_B = domain()->GetBndYp();
         }
         else
            exit(-2);
      break;

   case 'Z':
   case 'z':
      if (str_B[1]=='M' || str_B[1]=='m')
      {
         i_From = TOZM;
         p_B = domain()->GetBndZm();
      }
      else
         if (str_B[1]=='P' || str_B[1]=='p')
         {
            i_From = TOZP;
            p_B = domain()->GetBndZp();
         }
         else
            exit(-3);
      break;

   default:
      exit (-123);
   }
}

//-- Pulse:: ---------------------------------------------->
double Pulse::Form(double xco, double yco, double zco, int pol)
{
   // pol = 1...6: Ex ... Bz;

   Form(xco, yco, zco);

   switch (pol)
   {
   case EX:
      return f_Ex;

   case EY:
      return f_Ey;

   case EZ:
      return f_Ez;

   case BX:
      return f_Bx;

   case BY:
      return f_By;

   case BZ:
      return f_Bz;

   default:
      cout << "Error Pulse::form. Wrong pol = "<<pol<<"\n";
      return 0.;
   }

   return 0.;
}

//-- Pulse:: ---------------------------------------------->
double* Pulse::Form(double x, double y, double z)
{

   x += -f_Xcenter;
   y += -f_Ycenter - domain()->GetYlength()/2.;
   z += -f_Zcenter - domain()->GetZlength()/2.;

   // Pulse axis rotation
   double x1, y1, x2, z2;
   if (f_Kxy>0.) {
      x1 = (f_Kx*x + f_Ky*y)/f_Kxy;
      y1 = (-f_Ky*x + f_Kx*y)/f_Kxy;
   } else {
      x1 = x;
      y1 = y;
   }
   if (f_Kxyz>0.) {
      x2 = (f_Kxy*x1 + f_Kz*z)/f_Kxyz;
      z2 = (-f_Kz*x1 + f_Kxy*z)/f_Kxyz;
   } else {
      x2 = x1;
      z2 = z;
   }

   double xco = x2;
   double yco = y1;
   double zco = z2;
   //Pulse is now directed at at f_Kx, f_Ky, f_Kz;

   double phase = domain()->GetPhase();
   double hx = domain()->GetHx();
   double hy = domain()->GetHy();
   double hz = domain()->GetHz();
   double ts = domain()->GetTs();
   double dy = 0.;
   double dz = 0.;

   double phie = xco - phase + (f_Kx*hx + f_Ky*hy)/(2.*f_Kxy);
   double phib = xco - phase + ts/2.;

   yco = yco;
   zco = zco;
   double pT = ProfileT(phie,yco,zco);
   double pL = ProfileL(phie,yco,zco);

   double ample = f_A*pL*pT;
   double amplb = f_A*pL*pT;

   phie *= f_Omega;

   //f_Ex = -f_A*pL*(f_Ypol*dy*sin(phie*2.*PI)-f_Zpol*dz*cos(phie*2.*PI))/2./PI;
   double Ex2 = 0.;
   double Ey2 = ample*f_Ypol*cos(phie*2.*PI+f_Yphase);
   double Ez2 = ample*f_Zpol*cos(phie*2.*PI+f_Zphase);

   //f_Bx = -f_A*pL*(f_Ypol*dy*sin(phie*2.*PI)+f_Zpol*dz*cos(phie*2.*PI))/2./PI;
   double Bx2 = 0.;
   double Bz2 = amplb*f_Ypol*cos(phie*2.*PI+f_Yphase);
   double By2 = -amplb*f_Zpol*cos(phie*2.*PI+f_Zphase);

   double Ex1, Ey1, Ez1, Bx1, By1, Bz1;

   //Now we must rotate back the pulse fields!

   if (f_Kxyz>0.) {
      Ex1 = (f_Kxy*Ex2 - f_Kz*Ez2)/f_Kxyz;
      Ez1 = (f_Kz*Ex2 + f_Kxy*Ez2)/f_Kxyz;
      Bx1 = (f_Kxy*Bx2 - f_Kz*Bz2)/f_Kxyz;
      Bz1 = (f_Kz*Bx2 + f_Kxy*Bz2)/f_Kxyz;
   } else {
      Ex1 = Ex2;
      Ez1 = Ez2;
      Bx1 = Bx2;
      Bz1 = Bz2;
   }
   Ey1 = Ey2;
   By1 = By2;

   if (f_Kxy>0.) {
      f_Ex = (f_Kx*Ex1 - f_Ky*Ey1)/f_Kxy;
      f_Ey = (f_Ky*Ex1 + f_Kx*Ey1)/f_Kxy;
      f_Bx = (f_Kx*Bx1 - f_Ky*By1)/f_Kxy;
      f_By = (f_Ky*Bx1 + f_Kx*By1)/f_Kxy;
   } else {
      f_Ex = Ex1;
      f_Ey = Ey1;
      f_Bx = Bx1;
      f_By = By1;
   }
   f_Ez = Ez1;
   f_Bz = Bz1;

   return f_Fields;
}

/*
//-- Pulse:: ---------------------------------------------->
double Pulse::ProfileL(double xco, double yco, double zco)
{
if (f_Length<=0) return 0.;
double arg = xco/f_Length;
double tmp = 0.;
if ( arg>-1. && arg<1.) {
tmp = cos(0.5*PI*arg);
}
return tmp;
}
*/

//-- Pulse:: ---------------------------------------------->
double Pulse::ProfileL(double xco, double yco, double zco)
{
   if (f_Length<=0) return 0.;
   double arg = 0.;
   double tmpx = 0.;
   arg = xco/f_Length;

   switch(i_Lprofile) {
  case 0:
  default:
     if ( arg<-2. || arg>2.) {
        return 0.;
     }
     tmpx = exp(-arg*arg);
     return tmpx;
  case 1:
     if ( arg<-2. || arg>2.) {
        return 0.;
     }
     tmpx = arg*arg*exp(-arg*arg);
     return tmpx;
     break;
  case 2:
     if ( arg<-1. || arg>1.) {
        return 0.;
     }
     tmpx = (arg*arg-1);
     tmpx *= tmpx;
     return tmpx;
     break;
   }

   //  double arg = xco/f_Length;
   //  double tmp = 0.;
   //  if ( arg>-1. && arg<1.) {
   //    tmp = cos(0.5*PI*arg);
   //  }
   return tmpx;
}


/*
//-- Pulse:: ---------------------------------------------->
double Pulse::ProfileT(double xco, double yco, double zco)
{
double arg = 0.;
double tmp = 0.;
double tmpy = 0.;
double tmpz = 0.;
if (f_Ywidth>0) {
arg = yco/f_Ywidth;
if ( arg>-1. && arg<1.) {
tmpy = cos(0.5*PI*arg);
}
} else tmpy = 1.;
arg = 0.;
if (f_Zwidth>0) {
arg = zco/f_Zwidth;
if ( arg>-1. && arg<1.) {
tmpz = cos(0.5*PI*arg);
}
}  else tmpz = 1.;
if (tmpz > 0.) {
tmp = tmpy*tmpz;
}
return tmp;
}
*/

// Gauss
//-- Pulse:: ---------------------------------------------->
double Pulse::ProfileT(double xco, double yco, double zco)
{
   double argy = 0.;
   double argz = 0.;
   double arg = 0.;
   double tmp = 0.;
   double phase = domain()->GetPhase();

   if (f_YcenterOscillationAmplitude != 0.) {
      yco -= f_YcenterOscillationAmplitude
         *cos(2*PI*phase/f_YcenterOscillationPeriod + f_YcenterOscillationPhase);
   }

   if (f_ZcenterOscillationAmplitude != 0.) {
      zco -= f_ZcenterOscillationAmplitude
         *cos(2*PI*phase/f_ZcenterOscillationPeriod + f_ZcenterOscillationPhase);
   }

   if (f_Ywidth>0) {
      argy = yco/f_Ywidth;
   }
   if (f_Zwidth>0) {
      argz = zco/f_Zwidth;
   }
   double arg2 = argy*argy + argz*argz; 

   switch(i_Tprofile) {
  case 0:
  default:
     if ( arg2<-4. || arg2>4.) {
        return 0.;
     }
     tmp = exp(-arg2);
     return tmp;
  case 1:
     if ( arg2<-4. || arg2>4.) {
        return 0.;
     }
     tmp = arg2*exp(-arg2);
     return tmp;
     break;
   }
   return tmp;
}


//-- Mesh:: ---------------------------------------------->
void Mesh::Radiate(Pulse *parray[],int npulses)
{
   long i, j, k;
   double fld[FLD_DIM];
   for (i = 0; i < FLD_DIM; i++) {
      fld[i] = 0;
   }

   double PoyntingXm_Pulses = 0.;
   double PoyntingXm = 0.;
   double PoyntingXp_Pulses = 0.;
   double PoyntingXp = 0.;

   if (domain()->GetBndXm()->GetFcnd())
   {
      //      domain()-> out_Flog << "RADIATE GetBndXm()->GetFcnd() " << endl;
      i = 0;
      for (k=-2; k<l_Mz+2; k++) 
         for (j=-2; j<l_My+2; j++)
         {
            Cell &c = p_CellArray[GetN(i,j,k)];
            if (k>=0 && k<l_Mz && j>=0 && j<l_My) {
               PoyntingXm -= c.GetEyG()*c.GetBzG() - c.GetEzG()*c.GetByG();
            };
            c.SetFields(fld);  // fld[] =0 ?
         }
   }

   if (domain()->GetBndXp()->GetFcnd())
   {
      //	 domain()-> out_Flog << "RADIATE GetBndXp()->GetFcnd() " << endl;
      i=l_Mx-1;
      for (k=-2; k<l_Mz+2; k++) 
         for (j=-2; j<l_My+2; j++) {
            Cell &c = p_CellArray[GetN(i,j,k)];
            Cell &cp = p_CellArray[GetN(i+1,j,k)];
            if (k>=0 && k<l_Mz && j>=0 && j<l_My) {
               PoyntingXp += c.GetEyG()*c.GetBzG() - c.GetEzG()*c.GetByG();
            };
            c.f_Ez *= 0.2;
            c.f_Ey *= 0.2;
//            cp.SetFields(fld);
         }
   }

   if (MaxwellSolver()==2) {
      if (domain()->GetBndYm()->GetFcnd())
      {
         //      domain()-> out_Flog << "RADIATE GetBndXm()->GetFcnd() " << endl;
         j = 0;
         for (k=-2; k<l_Mz+2; k++) 
            for (i=-2; i<l_Mx+2; i++)
            {
               Cell &c = p_CellArray[GetN(i,j,k)];
               //	    c.f_Ex = 0.;
               //	    c.f_Ez = 0.;
            }
      }

      if (domain()->GetBndZm()->GetFcnd())
      {
         //      domain()-> out_Flog << "RADIATE GetBndXm()->GetFcnd() " << endl;
         k = 0;
         for (j=-2; j<l_My+2; j++) 
            for (i=-2; i<l_Mx+2; i++)
            {
               Cell &c = p_CellArray[GetN(i,j,k)];
               //	    c.f_Ex = 0.;
               //	    c.f_Ey = 0.;
            }
      } 
   }

   //  domain()-> out_Flog << "RADIATE npulses = " << npulses << endl;
   for (int ip=0; ip<npulses; ip++)
   {
      Pulse *pulse = parray[ip];
      int fcnd = pulse->GetBnd()->GetFcnd();
      if (fcnd == NIL)
         continue;

      char from = pulse->GetBnd()->Where();
      if (from & XDIR)
      {
         i = 0;
         if (from & MDIR)
            i=0;
         else
            i=l_Mx-1;

         double x = X(i);
         for (k=-2; k<l_Mz+2; k++)
         {
            double z = Z(k);
            for (j=-2; j<l_My+2; j++)
            {
               double y = Y(j);
               pulse->Form(x,y,z);
               p_CellArray[GetN(i,j,k)].AddFields(pulse->GetFields());
            }
         }
      }
      else
         if (from & YDIR)
         {
            long j = 0;
            if (from & MDIR)
               j=0;
            else
               j=l_My-1;
            double y = Y(j);
            for (long k=-2; k<l_Mz+2; k++)
            {
               double z = Z(k);
               for (long i=-2; i<l_Mx+2; i++)
               {
                  double x = X(i);
                  pulse->Form(x,y,z);
                  p_CellArray[GetN(i,j,k)].AddFields(pulse->GetFields());
               }
            }
         }
         else
            if (from & ZDIR)
            {
               long k = 0;
               if (from & MDIR) k=0; else k=l_Mz-1;
               double z = Z(k);
               for (long j=-2; j<l_My+2; j++) {
                  double y = Y(j);
                  for (long i=-2; i<l_Mx+2; i++) {
                     double x = X(i);
                     pulse->Form(x,y,z);
                     p_CellArray[GetN(i,j,k)].AddFields(pulse->GetFields());
                  }
               }
            } else {
               exit(from);
            }
   }
   if (domain()->GetBndXm()->GetFcnd())
   {
      //      domain()-> out_Flog << "RADIATE GetBndXm()->GetFcnd() " << endl;
      i = 0;
      for (k=0; k<l_Mz; k++) 
         for (j=0; j<l_My; j++)
         {
            Cell &c = p_CellArray[GetN(i,j,k)];
            PoyntingXm_Pulses += c.GetEyG()*c.GetBzG() - c.GetEzG()*c.GetByG();
         }
   }

   if (domain()->GetBndXp()->GetFcnd())
   {
      //	 domain()-> out_Flog << "RADIATE GetBndXp()->GetFcnd() " << endl;
      i=l_Mx-1;
      for (k=0; k<l_Mz; k++) 
         for (j=0; j<l_My; j++) {
            Cell &c = p_CellArray[GetN(i,j,k)];
            PoyntingXp_Pulses -= c.GetEyG()*c.GetBzG() - c.GetEzG()*c.GetByG();
         }
   }

    double Poyntings[5];
    Poyntings[4] = domain()->GetPhase();
    double fnorm = Hy()*Hz()*2.*PI*2.*PI;
    Poyntings[0] = PoyntingXm*fnorm;
    Poyntings[1] = PoyntingXm_Pulses*fnorm;
    Poyntings[2] = PoyntingXp*fnorm;
    Poyntings[3] = PoyntingXp_Pulses*fnorm;
    int root = 0;

#ifdef V_MPI
    int ierr = 0;
    int nproc = domain()->nPEs();
    double *recvbuf = new double[4*nproc];
    int sendcnt = 4;
    if (domain()->nPE() == root) {
      //      cout << "Starting to gather from" << nproc << " processors" << endl;
    }
    ierr = MPI_Gather(Poyntings, sendcnt, MPI_DOUBLE, 
                recvbuf, sendcnt, MPI_DOUBLE, 
                root, MPI_COMM_WORLD);

    if (domain()->nPE() == root) {
      //      cout << "Gathered." << endl;
      for (int npe=1; npe<nproc; npe++) {
	//	cout << "Adding proc #" <<npe << endl;
          for (int ndir=0; ndir<sendcnt; ndir++) {
            Poyntings[ndir] += recvbuf[npe*sendcnt+ndir];
          };   
       };
    }
    delete[] recvbuf;
#endif

    if (domain()->nPE() == root) {
       fprintf(domain()->GetPoyntingFile(),"%g %g %g %g %g\n",
          Poyntings[4],Poyntings[0],Poyntings[1],Poyntings[2],Poyntings[3]);
       fflush(domain()->GetPoyntingFile());
    };

}


//-- Mesh:: ---------------------------------------------->
void Mesh::InitPulse(Pulse *pulse)
{
   if (domain()->GetCntrl()->Reload()) {
      return;
   }
   int fcnd = pulse->GetBnd()->GetFcnd();
   char from = pulse->GetBnd()->Where();

   for (long k=0; k<l_Mz; k++) {
      double z = Z(k);
      for (long j=0; j<l_My; j++) {
         double y = Y(j);
         for (long i=0; i<l_Mx; i++) {
            double x = X(i);
            Cell &c = p_CellArray[GetN(i,j,k)];
            c.AddFields(pulse->Form(x,y,z));
         }
      }
   }
}
