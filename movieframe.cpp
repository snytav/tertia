#include "vlpl3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

#include "vlpl3d.h"

//---------------------------- Pulse::Pulse -----------------------
MovieFrame::MovieFrame (char *nm, FILE *f, int num) : NList (nm)
{
  int i=0;
  char FrameNames[16][16];
  if (num > 16)
  		num=16;
  str_What = new char*[num];
  Gets = new GETTYPE[num];
  for (i=0; i<num; i++)
  {
    sprintf(FrameNames[i],"Frame%d",i);
    str_What[i] = new char[3];
    NList_Entry * tmp_NList_Entry = AddEntry(FrameNames[i], str_What[i], "%3s");
  }

  if (f)
  {
    rewind(f);
    read(f);
  }

  CBuffer *buf = domain()->GetMPP()->GetBuf();

#ifdef V_MPI
  buf->reset();
  pack_nls(buf);
  domain()->BroadCast(buf);
  if (f==NULL)
  	unpack_nls(buf);
#endif

  for (i=0; i<num; i++)
  {
    Gets[i] = &Cell::GetDensG;
    if (strcmp(str_What[i],"ex") == 0)
    	Gets[i] = &Cell::GetExG;
    if (strcmp(str_What[i],"ey") == 0)
    	Gets[i] = &Cell::GetEyG;
    if (strcmp(str_What[i],"ez") == 0)
    	Gets[i] = &Cell::GetEzG;
    if (strcmp(str_What[i],"bx") == 0)
    	Gets[i] = &Cell::GetBxG;
    if (strcmp(str_What[i],"by") == 0)
    	Gets[i] = &Cell::GetByG;
    if (strcmp(str_What[i],"bz") == 0)
    	Gets[i] = &Cell::GetBzG;
    if (strcmp(str_What[i],"jx") == 0)
    	Gets[i] = &Cell::GetJxG;
    if (strcmp(str_What[i],"jy") == 0)
    	Gets[i] = &Cell::GetJyG;
    if (strcmp(str_What[i],"jz") == 0)
    	Gets[i] = &Cell::GetJzG;
    if (strcmp(str_What[i],"I") == 0)
    	Gets[i] = &Cell::GetIntensityG;
    if (strcmp(str_What[i],"ep") == 0)
    	Gets[i] = &Cell::GetEpsilonG;
    if (strcmp(str_What[i],"nb") == 0)
    	Gets[i] = &Cell::GetRhoBeam;
    if (strcmp(str_What[i],"ne") == 0)
    	Gets[i] = &Cell::GetDensG;
    if (strcmp(str_What[i],"nh") == 0)
    	Gets[i] = &Cell::GetPerturbedDensH;
    if (strcmp(str_What[i],"n0") == 0)
    	Gets[i] = &Cell::GetDens0;
    if (strcmp(str_What[i],"n1") == 0)
    	Gets[i] = &Cell::GetDens1;
    if (strcmp(str_What[i],"n2") == 0)
    	Gets[i] = &Cell::GetDens2;
    if (strcmp(str_What[i],"n3") == 0)
    	Gets[i] = &Cell::GetDens3;
    if (strcmp(str_What[i],"n4") == 0)
    	Gets[i] = &Cell::GetDens4;
    if (strcmp(str_What[i],"n5") == 0)
    	Gets[i] = &Cell::GetDens5;
    if (strcmp(str_What[i],"n6") == 0)
    	Gets[i] = &Cell::GetDens6;
    if (strcmp(str_What[i],"n7") == 0)
    	Gets[i] = &Cell::GetDens7;
    if (strcmp(str_What[i],"n8") == 0)
    	Gets[i] = &Cell::GetDens8;
    if (strcmp(str_What[i],"n9") == 0)
    	Gets[i] = &Cell::GetDens9;
    if (strcmp(str_What[i],"n10") == 0)
    	Gets[i] = &Cell::GetDens10;
    if (strcmp(str_What[i],"nb") == 0)
    	Gets[i] = &Cell::GetRhoBeam;
  };
}
