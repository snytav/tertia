
/* HDF5 Library                                                       */ 
#include "myhdfshell.h"

/* System libraries to include                                        */
#include <stdlib.h>
#include <string.h>
#include <assert.h>


using namespace std;

#include "vlpl3d.h"

#define LTBUF 32768
static char tbuf[LTBUF];

static int tbuf_length;

static char *MeshDataName = "meshinfo";

//#include "Logging.h"

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ SAVING PART $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

//--- Mesh::Save_Movie_Frame_H5 ----------------------->
int Mesh::Save_Movie_Frame_H5(int ichunk)
{
   char fname[128];
   long ldump=0;
   int i=0, j=0, k=0, is=0;
   int nsorts = domain()->i_Nsorts;
   int written = 0;
   char sortname[64];
   hid_t       file, fdataset, idataset, pdataset;         /* File and dataset            */
   hid_t       fdataspace, fmemspace, idataspace, imemspace, pdataspace;   /* Dataspace handles           */
   hid_t       fdataspace1;
   hsize_t     dimsf[3], dimsfmem[3], dimsfi[3], dimsp[3];              /* Dataset dimensions          */
   herr_t      status;                /* Error checking              */
   hid_t rank1 = 1;
   hid_t rank2 = 2;
   hid_t rank3 = 3;
   int nPEs = domain()->nPEs();
   int nPE = domain()->nPE();
   int iPE = domain()->iPE();
   int jPE = domain()->jPE();
   int kPE = domain()->kPE();
   int Xpartition = domain()->Xpartition();
   int Ypartition = domain()->Ypartition();
   int Zpartition = domain()->Zpartition();

   hsize_t offset[3];
   hsize_t count[3];

   double *fdata = new double[l_Mx*l_My*l_Mz];

   int isort = ALLSORTS;
   Density(isort);  //seting up density for //Cell::GetDens() for partcles of sort ALLSORTS
   GETTYPE getf;
   long Nfigs = domain()->i_NMovieFramesH5;
   long ifig = 0;

   //=============== saving fields ===============

   double phase = domain()->GetPhase();

   sprintf(fname,"%s//v3d_mframe_%5.5d.h5",domain()->str_MovieDirectory,ichunk);
   domain()->out_Flog << "SAVE MOVIE FRAME: Opening file " << fname << "\n";
   domain()->out_Flog.flush();

   if (nPE == 0) {
      domain()->out_Flog << "SAVE MOVIE FRAME: Opening file " << fname << "\n";
      domain()->out_Flog.flush();
      file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert (file >= 0);

      dimsfi[0] = 16;
      written = WriteHDF5RecordInt(file, MeshDataName, dimsfi[0], &l_Mx);
      assert (written >= 0);

      dimsfi[0] = 1;
      written = WriteHDF5Recorddouble(file, "phase", dimsfi[0], &phase);
      assert (written >= 0);

      dimsfi[0] = 1;
      written = WriteHDF5Record(file, "Nfigs", dimsfi[0], &Nfigs);
      assert (written >= 0);

      //============ X axis ============

      dimsfi[0] = l_Mx*Xpartition;
      double *Xaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Xaxis[i] = (i + domain()->p_Cntrl->GetShift())*Hx();
      }
      written = WriteHDF5Recorddouble(file, "X", dimsfi[0], Xaxis);
      assert (written >= 0);

      delete[] Xaxis;

      //============ Y axis ============

      dimsfi[0] = l_My*Ypartition;
      double *Yaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Yaxis[i] = i*Hy() - domain()->GetYlength()/2.;
      }
      written = WriteHDF5Recorddouble(file, "Y", dimsfi[0], Yaxis);
      assert (written >= 0);
      delete[] Yaxis;

      //============ Z axis ============

      dimsfi[0] = l_Mz*Zpartition;
      double *Zaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Zaxis[i] = i*Hz() - domain()->GetZlength()/2.;
      }
      written = WriteHDF5Recorddouble(file, "Z", dimsfi[0], Zaxis);
      assert (written >= 0);
      delete[] Zaxis;
   }
   
   //------------- Movie Selection -------------->
   for (int ifig=0; ifig<Nfigs; ifig++) {
     getf = domain()->p_MovieFrameH5->Gets[ifig];
     char *DataName = domain()->p_MovieFrameH5->str_What[ifig];
     char FigureName[128];
     
     sprintf(FigureName,"fig%d %s",ifig,DataName);
     
     for (k=0; k<l_Mz; k++) {
       for (j=0; j<l_My; j++) {
	 i = 0;
	 long lccc = GetN(i,j,k);
	 long n = i + l_Mx*(j+l_My*k);
	 for (i=0; i<l_Mx; i++) {
	   Cell &ccc = p_CellArray[lccc++];
	   fdata[n++] = (ccc.*getf)();
	 }
       }
     }
     
     SaveFieldParallel(fdata, file, FigureName,
		       Xpartition, Ypartition, Zpartition,
		       iPE, jPE, kPE, nPE);
     // Writing the data to the dataset                                */
     
   } // loop on frames
     // Close the datefile                                             */
   if (nPE == 0) {
      status = H5Fclose(file);
      assert (status >= 0);
   };

   delete[] fdata;

   return ldump;
}
