
#include "vlpl3d.h"
#include "cell3d.h"

//----------------------------Mesh::AverageBfield---------------------

void Mesh::AverageBfield(void) 
{
#ifdef S_BAVERAGE
  float a1 = m_Stencil.f_Bav1;
  float a2 = m_Stencil.f_Bav2;
  float a3 = m_Stencil.f_Bav3;
  float a8 = float(0.125);

  long i, j, k;

  for (k=0; k<l_Mz+1; k++) {
    for (j=0; j<l_My+1; j++) {
      i=-1;
      long nccc = GetN(i,  j,  k);
      long npcc = GetN(i+1,j,  k);
      long ncpc = GetN(i,  j+1,k);
      long nppc = GetN(i+1,j+1,k);
      long nccp = GetN(i,  j,  k+1);
      long npcp = GetN(i+1,j,  k+1);
      long ncpp = GetN(i,  j+1,k+1);
      long nppp = GetN(i+1,j+1,k+1);
      for (i=0; i<=l_Mx+1; i++) {
	Cell &ccc = p_CellArray[++nccc];
	Cell &pcc = p_CellArray[++npcc];
	Cell &cpc = p_CellArray[++ncpc];
	Cell &ppc = p_CellArray[++nppc];
	Cell &ccp = p_CellArray[++nccp];
	Cell &pcp = p_CellArray[++npcp];
	Cell &cpp = p_CellArray[++ncpp];
	Cell &ppp = p_CellArray[++nppp];

	ccc.f_BxAv = 
	  a8*((ccc.f_Bx + pcc.f_Bx) +
	      (cpc.f_Bx + ppc.f_Bx) +
	      (ccp.f_Bx + pcp.f_Bx) +
	      (cpp.f_Bx + ppp.f_Bx));
	ccc.f_ByAv = 
	  a8*((ccc.f_By + pcc.f_By) +
	      (cpc.f_By + ppc.f_By) +
	      (ccp.f_By + pcp.f_By) +
	      (cpp.f_By + ppp.f_By));
	ccc.f_BzAv = 
	  a8*((ccc.f_Bz + pcc.f_Bz) +
	      (cpc.f_Bz + ppc.f_Bz) +
	      (ccp.f_Bz + pcp.f_Bz) +
	      (cpp.f_Bz + ppp.f_Bz));
      }
    }
  }
#endif //S_BAVERAGE
}
