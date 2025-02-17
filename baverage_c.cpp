
#include "vlpl3d.h"

//----------------------------Mesh::AverageBfield---------------------
void Mesh::AverageBfield(void) 
{
   long i, j, k;

   float a1 = 0.565;
   float a2 = -0.065;
   float a8 = float(0.125);

   for (k=-l_dMz; k<l_Mz+l_dMz; k++) {
      for (j=-l_dMy; j<l_My+l_dMy; j++) {
         i = -l_dMx+1;
         long lc = GetN(i, j, k);
         long lcm = lc-1;
         long lcp = lc+1;
         long lcpp = lc+2;
         for (i=-l_dMx+1; i<l_Mx; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
             Cell &cpp = p_CellArray[lcpp];
            c.f_Jy = a1*(c.f_By + cp.f_By) + a2*(cpp.f_By + cm.f_By);
            c.f_Jz = a1*(c.f_Bz + cp.f_Bz) + a2*(cpp.f_Bz + cm.f_Bz);
            c.f_BxAv = c.f_ByAv = c.f_BzAv = 0.;
            lc++; lcm++; lcp++; lcpp++;
         }
      }
   }

   for (k=-l_dMz; k<l_Mz+l_dMz; k++) {
      for (j=-l_dMy+1; j<l_My; j++) {
         long i = -l_dMx;
         long lc = GetN(i, j, k);
         long lcm = GetN(i, j-1, k);
         long lcpp = GetN(i, j+2, k);
         long lcp = GetN(i, j+1, k);
         for (i=-l_dMx; i<l_Mx+l_dMx; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
            Cell &cpp = p_CellArray[lcpp];
            c.f_Jx  = a1*(c.f_Bx + cp.f_Bx) + a2*(cpp.f_Bx + cm.f_Bx);
            c.f_BzAv = a1*(c.f_Jz + cp.f_Jz) + a2*(cpp.f_Jz + cm.f_Jz);
            lc++; lcm++; lcp++; lcpp++;
         }
      }
   }

   for (k=-l_dMz+1; k<l_Mz; k++) {
      for (j=-l_dMy; j<l_My+l_dMy; j++) {
         long i = -l_dMx;
         long lc = GetN(i, j, k);
         long lcm = GetN(i, j, k-1);
         long lcpp = GetN(i, j, k+2);
         long lcp = GetN(i, j, k+1);
         for (i=-l_dMx; i<l_Mx+l_dMx; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
            Cell &cpp = p_CellArray[lcpp];
            c.f_BxAv = a1*(c.f_Jx + cm.f_Jx) + a2*(cp.f_Jx + cpp.f_Jx);
            c.f_ByAv = a1*(c.f_Jy + cm.f_Jy) + a2*(cp.f_Jy + cpp.f_Jy);
            lc++; lcm++; lcp++; lcpp++;
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////

void Mesh::AverageEfield(void) 
{
   return;
   long i, j, k;

   float a1 = 0.565;
   float a2 = -0.065;
   float a8 = float(0.125);

   for (k=-l_dMz; k<l_Mz+l_dMz; k++) {
      for (j=-l_dMy; j<l_My+l_dMy; j++) {
         i = -l_dMx+2;
         long lc = GetN(i, j, k);
         long lcm = lc-1;
         long lcp = lc+1;
         long lcmm = lc-2;
         for (i=-l_dMx+2; i<l_Mx+1; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
            Cell &cmm = p_CellArray[lcmm];
            c.f_Jx0 = a1*(c.f_Ex + cm.f_Ex) + a2*(cp.f_Ex + cmm.f_Ex);
            lc++; lcm++; lcp++; lcmm++;
         }
      }
   }

   for (k=-l_dMz; k<l_Mz+l_dMz; k++) {
      for (j=-l_dMy+2; j<l_My+1; j++) {
         long i = -l_dMx;
         long lc = GetN(i, j, k);
         long lcm = GetN(i, j-1, k);
         long lcmm = GetN(i, j-2, k);
         long lcp = GetN(i, j+1, k);
         for (i=-l_dMx; i<l_Mx+l_dMx; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
            Cell &cmm = p_CellArray[lcmm];
            c.f_Jy0 = a1*(c.f_Ey + cm.f_Ey) + a2*(cp.f_Ey + cmm.f_Ey);
            lc++; lcm++; lcp++; lcmm++;
         }
      }
   }

   for (k=-l_dMz+2; k<l_Mz+1; k++) {
      for (j=-l_dMy; j<l_My+l_dMy; j++) {
         long i = -l_dMx;
         long lc = GetN(i, j, k);
         long lcm = GetN(i, j, k-1);
         long lcmm = GetN(i, j, k-2);
         long lcp = GetN(i, j, k+1);
         for (i=-l_dMx; i<l_Mx+l_dMx; i++) {
            Cell &c = p_CellArray[lc];
            Cell &cm = p_CellArray[lcm];
            Cell &cp = p_CellArray[lcp];
            Cell &cmm = p_CellArray[lcmm];
            c.f_Jz0 = a1*(c.f_Ez + cm.f_Ez) + a2*(cp.f_Ez + cmm.f_Ez);
            lc++; lcm++; lcp++; lcmm++;
         }
      }
   }
}
