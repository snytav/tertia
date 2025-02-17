

/* HDF5 Library                                                       */ 
#include "myhdfshell.h"

/* System libraries to include                                        */
#include <stdlib.h>
#include <math.h>
#include <assert.h>


using namespace std;

#include "vlpl3d.h"

#define LTBUF 32768
static char tbuf[LTBUF];

static int tbuf_length;

static char *MeshDataName = "meshinfo";
static char *Ex = "ex";
static char *Ey = "ey";
static char *Ez = "ez";
static char *Bx = "bx";
static char *By = "by";
static char *Bz = "bz";
static char *Jx = "jx";
static char *Jy = "jy";
static char *Jz = "jz";
static char *Dens = "dens";
static char *DensSpec = "densspec";

//#include "Logging.h"

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ LOADING PART $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

//--- Domain::Load ----------------------->

long Domain::Load(int rank)
{
   char name[128];
   char sFile[128];
   long ldump=0;
   int i=0;
   int isave = GetCntrl()->GetNwrite();

   sprintf(sFile,"%s//vs%3.3dpe%3.3d_3d.dat",str_DataDirectory,isave,rank);
   out_Flog << "LOAD: Opening file " << sFile << "\n";

   FILE* pFile=fopen(sFile,"rb");
   if (pFile == NULL)
   {
      out_Flog << "ERROR: Domain::Load(int myrank): No such file with the name: " << sFile << endl;
      return -1;
   }

   for (i=0; i<LTBUF; i++) {
      ldump += fread(tbuf+i,1,1,pFile);
      if (tbuf[i]=='$')
         break;
   }

   sread(tbuf); // NameList read from tbuf

   int xpart = p_MPP->GetXpart();
   int ypart = p_MPP->GetYpart();
   int zpart = p_MPP->GetZpart();

   p_MPP->Load(pFile);


   // WARNING!!!   
   if(p_MPP->GetXpart() != xpart ||
      p_MPP->GetYpart() != ypart ||
      p_MPP->GetZpart() != zpart )
   {
      out_Flog << endl;
      out_Flog << "WARNING !!! WARNING !!! WARNING : Check MPP_partition section in ini file!" << endl;
      out_Flog << "Expected partitions: " << p_MPP->GetXpart() << 
         " x " << p_MPP->GetYpart() << 
         " x " << p_MPP->GetZpart() << endl;
   }
   // WARNING!!!

   out_Flog << "p_CGS->Load(pFile)" << "\n";

   p_CGS->Load(pFile);
   p_Cntrl->Load(pFile);

   l_NProcessed = 0;

   if (i_Npulses > 0) {
      pa_Pulses = new Pulse*[i_Npulses];
   } else {
      pa_Pulses = NULL;
   }

   if (i_Nsorts < 1) i_Nsorts = 1;
   pa_Species = new Specie*[i_Nsorts];
   pa_Species[0] = new Specie("Electrons",p_File);

   out_Flog << "Electron specie created"<<endl;

   for (i=1; i<i_Nsorts; i++)
   {
      sprintf(name,"Specie%d",i);
      pa_Species[i] = new IonSpecie(i, name, p_File);
   }

   if (i_NMovieFrames > 0) {
      p_MovieFrame = new MovieFrame("Movie",p_File,i_NMovieFrames);
   } else {
      p_MovieFrame = NULL;
   }

   if (i_NMovieFramesH5 > 0) {
      p_MovieFrameH5 = new MovieFrame("MovieHDF5",p_File,i_NMovieFramesH5);
   } else {
      p_MovieFrameH5 = NULL;
   }

   p_MPP->Init();
   l_NProcessed = 0;

   FILE *ftmp = NULL;
   ftmp = p_File;
   if (GetmyPE()==0 && ftmp==NULL) exit(-20);
   if (GetmyPE()!=0 && ftmp!=NULL) exit(-21);

   p_BndXm = new Boundary("Boundary_Xm",ftmp,XDIR+MDIR,p_MPP->p_XmPE);
   p_BndXp = new Boundary("Boundary_Xp",ftmp,XDIR+PDIR,p_MPP->p_XpPE);
   p_BndYm = new Boundary("Boundary_Ym",ftmp,YDIR+MDIR,p_MPP->p_YmPE);
   p_BndYp = new Boundary("Boundary_Yp",ftmp,YDIR+PDIR,p_MPP->p_YpPE);
   p_BndZm = new Boundary("Boundary_Zm",ftmp,ZDIR+MDIR,p_MPP->p_ZmPE);
   p_BndZp = new Boundary("Boundary_Zp",ftmp,ZDIR+PDIR,p_MPP->p_ZpPE);

   Cell::sf_DimFields = double(-1./(PI*f_Ts));
   Cell::sf_DimDens = 1.;

   long mx = l_Xsize/p_MPP->GetXpart();
   long ofX = mx*iPE();
   long my = l_Ysize/p_MPP->GetYpart();
   long ofY = my*jPE();
   long mz = l_Zsize/p_MPP->GetZpart();
   long ofZ = mz*kPE();

   p_M = new Mesh(mx, ofX, my, ofY, mz, ofZ);

   for (i=0; i<i_Npulses; i++)
   {
      sprintf(name,"Pulse%d",i);
      pa_Pulses[i] = new Pulse(name, p_File);
   }

   pa_Species[0]->Load(pFile);
   for (i=1; i<i_Nsorts; i++)
   {
      sprintf(name,"Specie%d",i);
      pa_Species[i]->Load(pFile);
   }
   p_Synchrotron = new Synchrotron("Synchrotron",p_File);
   p_Synchrotron->Load(pFile);

   ldump += p_M->Load(isave);
   fclose(pFile);
   pFile = NULL;

   Cell::sf_DimFields = double(-1./(PI*f_Ts));
   Cell::sf_DimDens = 1.;

   out_Flog << ldump << " LOAD: bytes read" << "\n";
   return ldump;
}

//--- Partition::Load ----------------------->
long Partition::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL)
      return 0;

   sread(tbuf);

   return ldump;
}

//--- CGS::Load ----------------------->
long UnitsCGS::Load(FILE* pFile)
{
   return 0;
}

//--- Controls::Load ----------------------->
long Controls::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL) return 0;

   ToSave->sread(tbuf);
   return ldump;
}

//--- Specie::Load ----------------------->
long Specie::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL)
      return 0;

   sread(tbuf);
   l_Np = 0;

   if (l_perCell) {
      f_Weight = 1./l_perCell;
      f_WeightCGS = p_CGS->GetCritDensity()*
         p_CGS->GetHx()*p_CGS->GetHy()*p_CGS->GetHz();
      f_dJx = 2.*(PI*domain()->GetTs())*PI*domain()->GetHx();
      f_dJy = 2.*(PI*domain()->GetTs())*PI*domain()->GetHy();
      f_dJz = 2.*(PI*domain()->GetTs())*PI*domain()->GetHz();
   }
   else { 
      f_Weight = 0.;
      f_WeightCGS = 0.;
      f_dJx = f_dJy = f_dJz = 0.;
   }

   return ldump;
}

//--- IonSpecie::Load ----------------------->
long IonSpecie::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL) 
      return 0;

   sread(tbuf);

   return ldump;
}

//--- Boundary::Load ----------------------->
long Boundary::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL)
      return 0;

   sread(tbuf);

   return ldump;
}
//--- Synchrotron::Load ----------------------->
long Synchrotron::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL)
      return 0;

   sread(tbuf);

   return ldump;
}

//--- Pulse::Load ----------------------->
long Pulse::Load(FILE* pFile)
{
   long ldump=0;

   if (pFile==NULL)
      return 0;

   sread(tbuf);

   return ldump;
}
//--- Mesh::Load ----------------------->
long Mesh::Load(int isave)
{
   char fname[128];
   long ldump=0;
   int i=0, j=0, k=0, is=0;
   int nsorts = domain()->i_Nsorts;
   long *npart = new long[nsorts];
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

   double *fdata = NULL;

   if (p_CellArray) {
      delete[] p_CellArray;
      p_CellArray = NULL;
   }

   //=============== loading fields ===============

   double sf_DimFields = Cell::sf_DimFields;
   int nprocessor = 0;
   for (nprocessor=0; nprocessor<nPEs; nprocessor++) {
#ifdef V_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (nprocessor != nPE) continue;

      sprintf(fname,"%s//vs%3.3d_3d_fields.h5",domain()->str_DataDirectory,isave);
      domain()->out_Flog << "LOAD: Opening file " << fname << "\n";

      file = H5Fopen (fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      if (file <= 0) {
         domain()->out_Flog << "LOAD: no such file " << fname << "\n";
      }
      assert (file >= 0);

      idataset = H5Dopen(file, MeshDataName, H5P_DEFAULT);
      assert (idataset >= 0);

      dimsfi[0] = 16;
      int written = ReadHDF5RecordInt(file, MeshDataName, dimsfi[0], &l_Mx);
      assert (written >= 0);

      l_MovieStarted = 0;

      fdata = new double[l_Mx*l_My*l_Mz];

      if (p_CellArray == NULL) {
         p_CellArray = new Cell[l_sizeXYZ];

         long lcell = 0;
         for (lcell=0; lcell<l_sizeXYZ; lcell++) {
            p_CellArray[lcell].l_N = lcell;
            if (p_CellArray[lcell].f_DensArray) {
               delete[] p_CellArray[lcell].f_DensArray;
               p_CellArray[lcell].f_DensArray = NULL;
            }
            if (nsorts>0) {
               p_CellArray[lcell].f_DensArray = new double[nsorts];
            }
         }
      }

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

      //------------- Ex -------------->
      fdataset = H5Dopen(file, Ex, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      hid_t dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Ex dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Ex dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Ex dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Ex = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- Ey -------------->
      fdataset = H5Dopen(file, Ey, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Ey dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Ey dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Ey dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Ey = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);


      //------------- Ez -------------->
      fdataset = H5Dopen(file, Ez, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Ez dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Ez dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Ez dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Ez = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);


      //------------- Bx -------------->
      fdataset = H5Dopen(file, Bx, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Ex dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Bx dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Bx dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Bx = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- By -------------->
      fdataset = H5Dopen(file, By, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch By dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch By dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch By dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_By = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);


      //------------- Bz -------------->
      fdataset = H5Dopen(file, Bz, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Bz dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Bz dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Bz dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Bz = fdata[n++]/sf_DimFields;
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- Jx -------------->
      fdataset = H5Dopen(file, Jx, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Jx dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Jx dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Jx dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Jx = fdata[n++];
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- Jy -------------->
      fdataset = H5Dopen(file, Jy, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Jy dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Jy dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Jy dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Jy = fdata[n++];
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);


      //------------- Jz -------------->
      fdataset = H5Dopen(file, Jz, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Jz dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Jz dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Jz dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Jz = fdata[n++];
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- Dens -------------->

      fdataset = H5Dopen(file, Dens, H5P_DEFAULT);
      assert (fdataset >= 0);
      fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
      dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
      status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

      if (dimsf[2] != l_Mx*Xpartition) {
         domain()->out_Flog << "Mismatch Dens dimsf[2]="<<dimsf[2]<<
            " and l_Mx*Xpartition="<<l_Mx*Xpartition;
      }
      if (dimsf[1] != l_My*Ypartition) {
         domain()->out_Flog << "Mismatch Dens dimsf[1]="<<dimsf[1]<<
            " and l_My*Ypartition="<<l_My*Ypartition;
      }
      if (dimsf[0] != l_Mz*Zpartition) {
         domain()->out_Flog << "Mismatch Dens dimsf[0]="<<dimsf[0]<<
            " and l_Mz*Zpartition="<<l_Mz*Zpartition;
      }
      H5Sclose(fdataspace1);
      status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
         H5P_DEFAULT, fdata);
      assert (status >= 0);

      for (k=0; k<l_Mz; k++) {
         for (j=0; j<l_My; j++) {
            i = 0;
            long lccc = GetN(i,j,k);
            long n = i + l_Mx*(j+l_My*k);
            for (i=0; i<l_Mx; i++) {
               Cell &ccc = p_CellArray[lccc++];
               ccc.f_Dens = fdata[n++];
            }
         }
      }

      /* Writing the data to the dataset                                */
      /* Close the dataset                                              */
      status = H5Dclose(fdataset);
      assert (status >= 0);

      //------------- DensArray -------------->

      for (int isort=0; isort<nsorts; isort++) {
         char DensSort[128];
         sprintf(DensSort,"Dens%d",isort);

         fdataset = H5Dopen(file, DensSort, H5P_DEFAULT);
         assert (fdataset >= 0);
         fdataspace1 = H5Dget_space(fdataset);    //* Get filespace handle first. 
         dummy_rank      = H5Sget_simple_extent_ndims(fdataspace1);
         status  = H5Sget_simple_extent_dims(fdataspace1, dimsf, NULL);

         if (dimsf[2] != l_Mx*Xpartition) {
            domain()->out_Flog << "Mismatch DensSort dimsf[2]="<<dimsf[2]<<
               " and l_Mx*Xpartition="<<l_Mx*Xpartition;
         }
         if (dimsf[1] != l_My*Ypartition) {
            domain()->out_Flog << "Mismatch DensSort dimsf[1]="<<dimsf[1]<<
               " and l_My*Ypartition="<<l_My*Ypartition;
         }
         if (dimsf[0] != l_Mz*Zpartition) {
            domain()->out_Flog << "Mismatch DensSort dimsf[0]="<<dimsf[0]<<
               " and l_Mz*Zpartition="<<l_Mz*Zpartition;
         }
         H5Sclose(fdataspace1);
         status = H5Dread(fdataset, H5T_NATIVE_DOUBLE, fmemspace, fdataspace,
            H5P_DEFAULT, fdata);
         assert (status >= 0);

         for (k=0; k<l_Mz; k++) {
            for (j=0; j<l_My; j++) {
               i = 0;
               long lccc = GetN(i,j,k);
               long n = i + l_Mx*(j+l_My*k);
               for (i=0; i<l_Mx; i++) {
                  Cell &ccc = p_CellArray[lccc++];
                  ccc.f_DensArray[isort] = fdata[n++];
               }
            }
         }
         /* Writing the data to the dataset                                */
         /* Close the dataset                                              */
         status = H5Dclose(fdataset);
         assert (status >= 0);
      }

      delete[] fdata;
      /* Close the datefile                                             */
      status = H5Fclose(file);
      assert (status >= 0);
   }								// end loop on processors

   //---------------------------------------------------------------------------------------
   /* Close the dataspace                                              */
   status = H5Sclose(fdataspace);
   assert (status >= 0);
   status = H5Sclose(fmemspace);
   assert (status >= 0);


   //=============== loading particles ===============

   hid_t* particles_dataset = new hid_t[nsorts];
   hid_t* particles_dataspace = new hid_t[nsorts];
   hid_t* particles_mem_dataspace = new hid_t[nsorts];

   sprintf(fname,"%s//vs%3.3d_3d_particles.h5",domain()->str_DataDirectory,isave);
   domain()->out_Flog << "SAVE: Opening file " << fname << "\n";

   dimsfi[0] = nsorts;

   idataspace = H5Screate_simple(rank1, dimsfi, NULL); 
   assert (idataspace >= 0);

   hid_t electron_tid = H5Tcreate (H5T_COMPOUND, sizeof(SavedParticle));
   assert (electron_tid >= 0);
   hid_t electron_mem_tid = H5Tcreate (H5T_COMPOUND, sizeof(SavedParticle));
   assert (electron_mem_tid >= 0);

   H5Tinsert(electron_tid, "cell", HOFFSET(SavedParticle, l_Cell), H5T_NATIVE_INT);
   H5Tinsert(electron_tid, "x", HOFFSET(SavedParticle, f_X), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "y", HOFFSET(SavedParticle, f_Y), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "z", HOFFSET(SavedParticle, f_Z), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "px", HOFFSET(SavedParticle, f_Px), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "py", HOFFSET(SavedParticle, f_Py), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "pz", HOFFSET(SavedParticle, f_Pz), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "weight", HOFFSET(SavedParticle, f_Weight), H5T_IEEE_F64LE);
   H5Tinsert(electron_tid, "q2m", HOFFSET(SavedParticle, f_Q2m), H5T_IEEE_F64LE);

   H5Tinsert(electron_mem_tid, "cell", HOFFSET(SavedParticle, l_Cell), H5T_NATIVE_INT);
   H5Tinsert(electron_mem_tid, "x", HOFFSET(SavedParticle, f_X), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "y", HOFFSET(SavedParticle, f_Y), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "z", HOFFSET(SavedParticle, f_Z), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "px", HOFFSET(SavedParticle, f_Px), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "py", HOFFSET(SavedParticle, f_Py), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "pz", HOFFSET(SavedParticle, f_Pz), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "weight", HOFFSET(SavedParticle, f_Weight), H5T_IEEE_F64LE);
   H5Tinsert(electron_mem_tid, "q2m", HOFFSET(SavedParticle, f_Q2m), H5T_IEEE_F64LE);

   hid_t ion_tid = H5Tcreate (H5T_COMPOUND, sizeof(SavedIon));
   assert (ion_tid >= 0);
   hid_t ion_mem_tid = H5Tcreate (H5T_COMPOUND, sizeof(SavedIon));
   assert (ion_mem_tid >= 0);

   H5Tinsert(ion_tid, "cell", HOFFSET(SavedIon, l_Cell), H5T_NATIVE_INT);
   H5Tinsert(ion_tid, "x", HOFFSET(SavedIon, f_X), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "y", HOFFSET(SavedIon, f_Y), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "z", HOFFSET(SavedIon, f_Z), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "px", HOFFSET(SavedIon, f_Px), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "py", HOFFSET(SavedIon, f_Py), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "pz", HOFFSET(SavedIon, f_Pz), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "weight", HOFFSET(SavedIon, f_Weight), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "q2m", HOFFSET(SavedIon, f_Q2m), H5T_IEEE_F64LE);
   H5Tinsert(ion_tid, "i_z", HOFFSET(SavedIon, i_Z), H5T_NATIVE_INT);
   H5Tinsert(ion_tid, "sort", HOFFSET(SavedIon, i_Sort), H5T_NATIVE_INT);

   H5Tinsert(ion_mem_tid, "cell", HOFFSET(SavedIon, l_Cell), H5T_NATIVE_INT);
   H5Tinsert(ion_mem_tid, "x", HOFFSET(SavedIon, f_X), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "y", HOFFSET(SavedIon, f_Y), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "z", HOFFSET(SavedIon, f_Z), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "px", HOFFSET(SavedIon, f_Px), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "py", HOFFSET(SavedIon, f_Py), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "pz", HOFFSET(SavedIon, f_Pz), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "weight", HOFFSET(SavedIon, f_Weight), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "q2m", HOFFSET(SavedIon, f_Q2m), H5T_IEEE_F64LE);
   H5Tinsert(ion_mem_tid, "i_z", HOFFSET(SavedIon, i_Z), H5T_NATIVE_INT);
   H5Tinsert(ion_mem_tid, "sort", HOFFSET(SavedIon, i_Sort), H5T_NATIVE_INT);

   long* pcount = new long[nsorts];

   for (nprocessor=0; nprocessor<nPEs; nprocessor++) {
#ifdef V_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      if (nprocessor != nPE) continue;

      file = H5Fopen (fname, H5F_ACC_RDONLY, H5P_DEFAULT);
      if (file <= 0) {
         domain()->out_Flog << "LOAD: no such file " << fname << "\n";
      }
      assert (file >= 0);

      char idataname[128];
      sprintf(idataname,"Nparticles_at_PE_%d",nPE);
      ReadHDF5Record(file, idataname, nsorts, pcount);

      for (is=0; is<nsorts; is++) {
         domain()->out_Flog << "Looking for "<<pcount[is]<<" particles of sort "<<is<<endl;
         cout << "Looking for "<<pcount[is]<<" particles of sort "<<is<<endl;
         npart[is] = pcount[is];
         dimsp[0] = npart[is];
         pcount[is] = 0;
         if (npart[is] < 1) continue;
         particles_dataspace[is] = H5Screate_simple(rank1, dimsp, NULL);
         assert(particles_dataspace[is] >= 0);
         particles_mem_dataspace[is] = H5Screate_simple(rank1, dimsp, NULL);
         assert(particles_mem_dataspace[is] >= 0);
      }

      status = H5Dclose(idataset);
      assert (status >= 0);

      char pdataname[128];
      if (npart[0] > 0) {
         sprintf(pdataname,"Electrons_at_PE_%d",nPE);
         particles_dataset[0] = H5Dopen(file, pdataname, H5P_DEFAULT);
         assert (particles_dataset[0] >= 0);
      }

      for (is=1; is<nsorts; is++) {
         if (npart[is] < 1) continue;
         sprintf(sortname,"Ions_of_sort_%2.2d_at_PE_%d",is,nPE);
         particles_dataset[is] = H5Dopen(file, sortname, H5P_DEFAULT);
         assert(particles_dataspace[is] >= 0);
      }

      for (i=0; i<3; i++) {
         offset[i] = 0;
         count[i] = 1;
      }

      double hx = domain()->GetHx();
      double hy = domain()->GetHy();
      double hz = domain()->GetHz();
      for (is=0; is<nsorts; is++) {
         if (npart[is] < 1) continue;
         for (long np=0; np<npart[is]; np++) {
            offset[0] = np;
            count[0] = 1;
            status = H5Sselect_hyperslab(particles_dataspace[is], H5S_SELECT_SET, offset, 
               NULL, count, NULL);
            assert (status >= 0);
            offset[0] = 0;
            status = H5Sselect_hyperslab(particles_mem_dataspace[is], H5S_SELECT_SET, offset, 
               NULL, count, NULL);
            assert (status >= 0);
            Particle *p = NULL;
            Electron *etmp = NULL;
            Ion *iontmp = NULL;
            if (npart[is] < 1) {
               domain()->out_Flog << "LOAD: trying to read not existing particles of sort "<<is <<endl;
               cout << "LOAD: trying to read not existing particles of sort "<<is <<endl;
            }
            if (is==0) {
               p = etmp = new Electron();
               SavedParticle el;
               status = H5Dread(particles_dataset[is], electron_tid, particles_mem_dataspace[is], 
                  particles_dataspace[is], H5P_DEFAULT, &el);
               assert (status >= 0);
               *p = el;
            } else {
               p = iontmp = new Ion(NULL,is);
               SavedIon ion;
               status = H5Dread(particles_dataset[is], ion_tid, particles_mem_dataspace[is], 
                  particles_dataspace[is], H5P_DEFAULT, &ion);
               if (status <0) {
                  domain()->out_Flog << "LOAD: was tgrying to read ion of sort=" << is << " np="<<np<<" of " << npart[is] <<"\n";
                  cout << "LOAD: was tgrying to read ion of sort=" << is << " np="<<np<<" of " << npart[is] <<"\n";
               }
               assert (status >= 0);
               *iontmp = ion;
            }
            //				p->p_Next = NULL;
            assert (p->p_Next == NULL);
            long lcell = p->GetCellNumber();
            if (lcell<0 || lcell > l_sizeXYZ) {
               domain()->out_Flog << "LOAD: error lcell=" << lcell<<" sort " << is << " np="<<np<< "\n";
               cout << "LOAD: error lcell=" << lcell<<" sort " << is << " np="<<np<< "\n";
            }
            assert (lcell>=0 && lcell < l_sizeXYZ);
            int isort_test = p->GetSort();
            assert (isort_test == is);

            Cell &ccc = p_CellArray[lcell];
            int itmp = GetI_from_CellNumber(lcell) + l_Mx*domain()->GetMPP()->GetiPE();
            int jtmp = GetJ_from_CellNumber(lcell) + l_My*domain()->GetMPP()->GetjPE();
            int ktmp = GetK_from_CellNumber(lcell) + l_Mz*domain()->GetMPP()->GetjPE();
            p->f_X -= itmp*hx;
            p->f_Y -= jtmp*hy;
            p->f_Z -= ktmp*hz;
            double energy = p->f_Px*p->f_Px + p->f_Py*p->f_Py + p->f_Pz*p->f_Pz;
            energy = energy*energy/(1.+sqrt(1.+energy*energy));
            if (energy > 1e9) {
               domain()->out_Flog << "Load: Too high energy="<<energy<<" sort="<<is
                  <<" px="<<p->f_Px<<" py="<<p->f_Py<<" pz="<<p->f_Pz << endl;
               cout << "Load: Too high energy="<<energy<<" sort="<<is
                  <<" px="<<p->f_Px<<" py="<<p->f_Py<<" pz="<<p->f_Pz << endl;
            }
            if (p->f_X<0 || p->f_X>1 ||p->f_Y<0 || p->f_Y>1 || p->f_Z<0 || p->f_Z>1) {
               domain()->out_Flog << "Load: Error position! sort="<<is
                  <<" x="<<p->f_X<<" y="<<p->f_Y<<" z="<<p->f_Z << endl;
               cout << "Load: Error position! sort="<<is
                  <<" x="<<p->f_X<<" y="<<p->f_Y<<" z="<<p->f_Z << endl;
            }
            ccc.AddParticle(p);
         }
         status = H5Dclose(particles_dataset[is]);
         assert (status >= 0);
         status = H5Sclose(particles_dataspace[is]);
         assert (status >= 0);
         status = H5Sclose(particles_mem_dataspace[is]);
         assert (status >= 0);
      }
      status = H5Fclose(file);
      assert (status >= 0);
   }

   status = H5Tclose(electron_tid);
   assert (status >= 0);
   status = H5Tclose(ion_tid);
   assert (status >= 0);
   status = H5Tclose(electron_mem_tid);
   assert (status >= 0);
   status = H5Tclose(ion_mem_tid);
   assert (status >= 0);

   delete[] npart;
   delete[] particles_dataset;
   delete[] pcount;
   delete[] particles_dataspace;
   delete[] particles_mem_dataspace;

   return ldump;
}
