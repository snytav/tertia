
/* HDF5 Library                                                       */ 
#include "myhdfshell.h"

/* System libraries to include                                        */
#include <stdlib.h>
#include <assert.h>
#include <iostream>

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

   float *fdata = new float[l_Mx*l_My*l_Mz];

   int isort = ALLSORTS;
   Density(isort);  //seting up density for //Cell::GetDens() for partcles of sort ALLSORTS
   GETTYPE getf;
   long Nfigs = domain()->i_NMovieFramesH5;
   long ifig = 0;

   //=============== saving fields ===============

   float phase = domain()->GetPhase();

   sprintf(fname,"v3d_mframe_%5.5d.h5",ichunk);
   domain()->out_Flog << "SAVE MOVIE FRAME: Opening file " << fname << "\n";
   domain()->out_Flog.flush();

   if (nPE == 0) {
      domain()->out_Flog << "SAVE MOVIE FRAME: Opening file " << fname << "\n";
      domain()->out_Flog.flush();
      file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert (file >= 0);
      dimsfi[0] = 16;
      idataspace = H5Screate_simple(rank1, dimsfi, NULL); 
      assert (idataspace >= 0);
      idataset = H5Dcreate(file, MeshDataName, H5T_STD_I32LE, idataspace, H5P_DEFAULT);
      assert (idataset >= 0);
      status = H5Dwrite(idataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, &l_Mx);
      assert (status >= 0);
      status = H5Dclose(idataset);
      assert (status >= 0);
      status = H5Sclose(idataspace);
      assert (status >= 0);

      dimsfi[0] = 1;
      idataspace = H5Screate_simple(rank1, dimsfi, NULL); 
      assert (idataspace >= 0);
      idataset = H5Dcreate(file, "phase", H5T_STD_FLOAT32LE, idataspace, H5P_DEFAULT);
      assert (idataset >= 0);
      status = H5Dwrite(idataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, &phase);
      assert (status >= 0);
      status = H5Dclose(idataset);
      assert (status >= 0);
      status = H5Sclose(idataspace);
      assert (status >= 0);

      //============ X axis ============

      dimsfi[0] = l_Mx*Xpartition;
      fdataspace = H5Screate_simple(rank1, dimsfi, NULL); 
      assert (fdataspace >= 0);
      fdataset = H5Dcreate(file, "X", H5T_IEEE_F32LE, fdataspace, H5P_DEFAULT);
      assert (fdataset >= 0);
      float *Xaxis = new float[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Xaxis[i] = (i + domain()->p_Cntrl->GetShift())*Hx();
      }

      status = H5Dwrite(fdataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, Xaxis);
      assert (status >= 0);
      status = H5Dclose(fdataset);
      assert (status >= 0);
      status = H5Sclose(fdataspace);
      assert (status >= 0);
      delete[] Xaxis;

      //============ Y axis ============

      dimsfi[0] = l_My*Ypartition;
      fdataspace = H5Screate_simple(rank1, dimsfi, NULL); 
      assert (fdataspace >= 0);
      fdataset = H5Dcreate(file, "Y", H5T_IEEE_F32LE, fdataspace, H5P_DEFAULT);
      assert (fdataset >= 0);
      float *Yaxis = new float[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Yaxis[i] = i*Hy() - domain()->GetYlength()/2.;
      }

      status = H5Dwrite(fdataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, Yaxis);
      assert (status >= 0);
      status = H5Dclose(fdataset);
      assert (status >= 0);
      status = H5Sclose(fdataspace);
      assert (status >= 0);
      delete[] Yaxis;

      //============ Z axis ============

      dimsfi[0] = l_Mz*Zpartition;
      fdataspace = H5Screate_simple(rank1, dimsfi, NULL); 
      assert (fdataspace >= 0);
      fdataset = H5Dcreate(file, "Z", H5T_IEEE_F32LE, fdataspace, H5P_DEFAULT);
      assert (fdataset >= 0);
      float *Zaxis = new float[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Zaxis[i] = i*Hz() - domain()->GetZlength()/2.;
      }

      status = H5Dwrite(fdataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, Zaxis);
      assert (status >= 0);
      status = H5Dclose(fdataset);
      assert (status >= 0);
      status = H5Sclose(fdataspace);
      assert (status >= 0);
      delete[] Zaxis;

      status = H5Fclose(file);
      assert (status >= 0);
   }

   int nprocessor = 0;
   for (nprocessor=0; nprocessor<nPEs; nprocessor++) {
#ifdef V_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (nprocessor != nPE) continue;

      file = H5Fopen (fname, H5F_ACC_RDWR, H5P_DEFAULT);
      if (file <= 0) {
         domain()->out_Flog << "SAVE MOVIE FRAME re-open: no such file " << fname << "\n";
         domain()->out_Flog.flush();
         cout <<  "pe=" << nPE << "SAVE MOVIE FRAME re-open: no such file " << fname << "\n";
      }
      assert (file >= 0);


      dimsf[0] = l_Mz*Zpartition;
      dimsfmem[0] = l_Mz;
      dimsf[1] = l_My*Ypartition;
      dimsfmem[1] = l_My;
      dimsf[2] = l_Mx*Xpartition;
      dimsfmem[2] = l_Mx;

      /* Creating the dataspace                                         */
      fdataspace = H5Screate_simple(rank3, dimsf, NULL); 
      assert (fdataspace >= 0);

      fmemspace = H5Screate_simple(rank3, dimsfmem, NULL); 
      assert (fmemspace >= 0);

      offset[2] = l_Mx*iPE;
      count[2] = l_Mx;
      offset[1] = l_My*jPE;
      count[1] = l_My;
      offset[0] = l_Mz*kPE;
      count[0] = l_Mz;
      status = H5Sselect_hyperslab(fdataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
      assert (status >= 0);

      offset[0] = offset[1] = offset[2] = 0;
      status = H5Sselect_hyperslab(fmemspace, H5S_SELECT_SET, offset, NULL, count, NULL);
      assert (status >= 0);
      hid_t dummy_rank;

      //------------- Movie Selection -------------->
      for (int ifig=0; ifig<Nfigs; ifig++) {
         getf = domain()->p_MovieFrame->Gets[ifig];
         char *DataName = domain()->p_MovieFrame->str_What[ifig];
         char FigureName[128];
         sprintf(FigureName,"fig%d",ifig);
         if (nPE == 0) {
            // Creating the dataset within the dataspace                      */
            fdataset = H5Dcreate(file, FigureName, H5T_IEEE_F32LE, fdataspace, H5P_DEFAULT);
            assert (fdataset >= 0);
         } else { // nPE > 0
            fdataset = H5Dopen(file, FigureName);
            assert (fdataset >= 0);
            fdataspace1 = H5Dget_space(fdataset);    /// Get filespace handle first. 
            dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
            status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

            if (dimsf[2] != l_Mx*Xpartition) {
               domain()->out_Flog << "Mismatch Figure dimsf[2]="<<dimsf[2]<<
                  " and l_Mx*Xpartition="<<l_Mx*Xpartition;
            }
            if (dimsf[1] != l_My*Ypartition) {
               domain()->out_Flog << "Mismatch Figure dimsf[1]="<<dimsf[1]<<
                  " and l_My*Ypartition="<<l_My*Ypartition;
            }
            if (dimsf[0] != l_Mz*Zpartition) {
               domain()->out_Flog << "Mismatch Figure dimsf[0]="<<dimsf[0]<<
                  " and l_Mz*Zpartition="<<l_Mz*Zpartition;
            }
            H5Sclose(fdataspace1);
         }

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

         // Writing the data to the dataset                                */
         status = H5Dwrite(fdataset, H5T_NATIVE_FLOAT, fmemspace, fdataspace,
            H5P_DEFAULT, fdata);
         assert (status >= 0);
         // Close the dataset                                              */
         status = H5Dclose(fdataset);
         assert (status >= 0);
      }


      // Close the datefile                                             */
      status = H5Fclose(file);
      assert (status >= 0);
   }								// end loop on processors
   //---------------------------------------------------------------------------------------
   // Close the dataspace                                              */
   status = H5Sclose(fdataspace);
   assert (status >= 0);
   status = H5Sclose(fmemspace);
   assert (status >= 0);

   delete[] fdata;

   return ldump;
}
