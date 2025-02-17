#include "vlpl3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
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
float Pulse::Form(float xco, float yco, float zco, int pol)
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
float* Pulse::Form(float x, float y, float z)
{

   x += -f_Xcenter;
   y += -f_Ycenter - domain()->GetYlength()/2.;
   z += -f_Zcenter - domain()->GetZlength()/2.;

   // Pulse axis rotation
   float x1, y1, x2, z2;
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

   float xco = x2;
   float yco = y1;
   float zco = z2;
   //Pulse is now directed at at f_Kx, f_Ky, f_Kz;

   float phase = domain()->GetPhase();
   float hx = domain()->GetHx();
   float hy = domain()->GetHy();
   float hz = domain()->GetHz();
   float ts = domain()->GetTs();
   float dy = 0.;
   float dz = 0.;

   float phie = xco - phase + (f_Kx*hx + f_Ky*hy)/(2.*f_Kxy);
   float phib = xco - phase + ts/2.;

   yco = yco;
   zco = zco;
   float pT = ProfileT(phie,yco,zco);
   float pL = ProfileL(phie,yco,zco);

   float ample = f_A*pL*pT;
   float amplb = f_A*pL*pT;

   phie *= f_Omega;

   //f_Ex = -f_A*pL*(f_Ypol*dy*sin(phie*2.*PI)-f_Zpol*dz*cos(phie*2.*PI))/2./PI;
   float Ex2 = 0.;
   float Ey2 = ample*f_Ypol*cos(phie*2.*PI+f_Yphase);
   float Ez2 = ample*f_Zpol*cos(phie*2.*PI+f_Zphase);

   //f_Bx = -f_A*pL*(f_Ypol*dy*sin(phie*2.*PI)+f_Zpol*dz*cos(phie*2.*PI))/2./PI;
   float Bx2 = 0.;
   float Bz2 = amplb*f_Ypol*cos(phie*2.*PI+f_Yphase);
   float By2 = -amplb*f_Zpol*cos(phie*2.*PI+f_Zphase);

   float Ex1, Ey1, Ez1, Bx1, By1, Bz1;

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
float Pulse::ProfileL(float xco, float yco, float zco)
{
if (f_Length<=0) return 0.;
float arg = xco/f_Length;
float tmp = 0.;
if ( arg>-1. && arg<1.) {
tmp = cos(0.5*PI*arg);
}
return tmp;
}
*/

//-- Pulse:: ---------------------------------------------->
float Pulse::ProfileL(float xco, float yco, float zco)
{
   if (f_Length<=0) return 0.;
   float arg = 0.;
   float tmpx = 0.;
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

   //  float arg = xco/f_Length;
   //  float tmp = 0.;
   //  if ( arg>-1. && arg<1.) {
   //    tmp = cos(0.5*PI*arg);
   //  }
   return tmpx;
}


/*
//-- Pulse:: ---------------------------------------------->
float Pulse::ProfileT(float xco, float yco, float zco)
{
float arg = 0.;
float tmp = 0.;
float tmpy = 0.;
float tmpz = 0.;
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
float Pulse::ProfileT(float xco, float yco, float zco)
{
   float argy = 0.;
   float argz = 0.;
   float arg = 0.;
   float tmp = 0.;
   float phase = domain()->GetPhase();

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
   float arg2 = argy*argy + argz*argz; 

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
   float fld[FLD_DIM];
   for (i = 0; i < FLD_DIM; i++)
      fld[i] = 0;

   if (domain()->GetBndXm()->GetFcnd())
   {
      //      domain()-> out_Flog << "RADIATE GetBndXm()->GetFcnd() " << endl;
      i = 0;
      for (k=-2; k<l_Mz+2; k++) 
         for (j=-2; j<l_My+2; j++)
         {
//            p_CellArray[GetN(i,j,k)].SetFields(fld);  // fld[] =0 ?
            Cell &c = p_CellArray[GetN(i,j,k)];
            Cell &cp = p_CellArray[GetN(i+1,j,k)];
            float s2 = sqrt(2.)/2.;
            float Fpz = c.f_Ez - s2*(c.f_By-c.f_Bx);
            float Fmz = c.f_Ez + s2*(c.f_By-c.f_Bx);
            float Fpy = s2*(c.f_Ey-c.f_Ex) + c.f_Bz;
            float Fmy = s2*(c.f_Ey-c.f_Ex) - c.f_Bz;
            float E45 = Fpy/2.;
            float B45 = -Fpz/2.;

            c.f_Ey = s2*E45;
            c.f_Ex = -s2*E45;
            c.f_By = s2*B45;
            c.f_Bx = -s2*B45;

//            c.f_Ez = Fpz/2.;
//            c.f_Ey = Fpy/2.;
//            c.f_Bz = Fpy/2.;
//            c.f_By = -Fpz/2.;
         }
   }

   if (domain()->GetBndXp()->GetFcnd())
   {
      //	 domain()-> out_Flog << "RADIATE GetBndXp()->GetFcnd() " << endl;
      i=l_Mx-2;
      for (k=-2; k<l_Mz+2; k++) 
         for (j=-2; j<l_My+2; j++) {
            Cell &c = p_CellArray[GetN(i,j,k)];
            Cell &cp = p_CellArray[GetN(i+1,j,k)];
            float Fpz = c.f_Ez - cp.f_By;
            float Fmz = c.f_Ez + cp.f_By;
            float Fpy = c.f_Ey + cp.f_Bz;
            float Fmy = c.f_Ey - cp.f_Bz;
            c.f_Ez = Fmz/2.;
            c.f_Ey = Fmy/2.;
            cp.f_Bz = -Fmy/2.;
            cp.f_By = Fmz/2.;
//            c.f_Ez = 0.;
//            c.f_Ey = 0.;
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

         float x = X(i);
         for (k=-2; k<l_Mz+2; k++)
         {
            float z = Z(k);
            for (j=-2; j<l_My+2; j++)
            {
               float y = Y(j);
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
            float y = Y(j);
            for (long k=-2; k<l_Mz+2; k++)
            {
               float z = Z(k);
               for (long i=-2; i<l_Mx+2; i++)
               {
                  float x = X(i);
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
               float z = Z(k);
               for (long j=-2; j<l_My+2; j++) {
                  float y = Y(j);
                  for (long i=-2; i<l_Mx+2; i++) {
                     float x = X(i);
                     pulse->Form(x,y,z);
                     p_CellArray[GetN(i,j,k)].AddFields(pulse->GetFields());
                  }
               }
            } else {
               exit(from);
            }
   }
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
      float z = Z(k);
      for (long j=0; j<l_My; j++) {
         float y = Y(j);
         for (long i=0; i<l_Mx; i++) {
            float x = X(i);
            Cell &c = p_CellArray[GetN(i,j,k)];
            c.AddFields(pulse->Form(x,y,z));
         }
      }
   }
}
