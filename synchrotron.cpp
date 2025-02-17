/* HDF5 Library                                                       */ 
#include "myhdfshell.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include "vlpl3d.h"

//---------------------------- UnitsCGS::UnitsCGS -----------------------
Synchrotron::Synchrotron(char *nm, FILE *f) : NList (nm)
{
   AddEntry("SynMin", &f_SynMin, 1e2);
   AddEntry("nEbins", &i_nEbins, 1e2);
   AddEntry("nThetabins", &i_nThetabins, 1e2);
   AddEntry("nPhibins", &i_nPhibins, 1e2);
   AddEntry("Emin", &f_Emin, 1e2);
   AddEntry("Emax", &f_Emax, 1e6);
   AddEntry("f_LogEmin", &f_LogEmin);
   AddEntry("f_LogEmax", &f_LogEmax);
   AddEntry("f_EStepFactor", &f_EStepFactor);
   AddEntry("f_PhiStep", &f_PhiStep);
   AddEntry("f_ThetaStep", &f_ThetaStep);
   AddEntry("i_Save", &i_Save, 0);

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

   if (f_Emax <= 0.) f_Emax = 1e6;
   if (f_Emin <= 0. || f_Emin >= f_Emax) f_Emin = f_Emax*1e-4; 
   f_LogEmin = log(f_Emin);
   f_LogEmax = log(f_Emax);

   if (i_nEbins < 1) i_nEbins = 100;
   if (i_nThetabins < 1) i_nThetabins = 100;
   if (i_nPhibins < 1) i_nPhibins = 100;

   f_EStepFactor = (log(f_Emax) - log(f_Emin))/i_nEbins;
   f_ThetaStep = PI/i_nThetabins;
   f_PhiStep = 2*PI/i_nPhibins;

   p_PhotonsArray3D = new double[i_nPhibins*i_nThetabins*i_nEbins];

}

void Synchrotron::AddParticle(Particle* p, double nph, double theta, double phi, double Ephoton) {
   if (Ephoton < f_Emin || Ephoton > f_Emax) return; 
   int iE = (log(Ephoton)-f_LogEmin)/f_EStepFactor;
   if (iE < 0) iE = 0;
   if (iE >= i_nEbins) iE = i_nEbins - 1; 

   if (theta < 0 || theta > PI) return; 
   int iTheta = theta/f_ThetaStep;
   if (iTheta < 0) iTheta = 0;
   if (iTheta >= i_nThetabins) iTheta = i_nThetabins - 1; 

   if (phi < -PI || phi > PI) return; 
   int iPhi = (phi + PI)/f_PhiStep;
   if (iPhi < 0) iPhi = 0;
   if (iPhi >= i_nPhibins) iPhi = i_nPhibins - 1; 

   double weight = p->GetSpecie()->GetWeightCGS()*p->GetWeight();
   p_PhotonsArray3D[iE + i_nEbins*(iTheta + iPhi*i_nThetabins)] += weight*nph;
}

double Synchrotron::GetRadiatedEnergy_eV() {
   double eV = 0.;
   for (int i=0; i<i_nEbins; i++) {
      double energy = exp(i*f_EStepFactor + f_LogEmin);
      for (int k=0; k<i_nPhibins; k++) {
         for (int j=0; j<i_nThetabins; j++) {
            eV += energy*p_PhotonsArray3D[i + i_nEbins*(j + k*i_nThetabins)];
         }
      }
   }
   return eV;
}

double Synchrotron::GetRadiatedEnergy_J() {
   return GetRadiatedEnergy_eV()*1.6e-19;
}

int Synchrotron::StoreDistributionHDF5() {
   char fname[128];
   long ldump=i_nPhibins*i_nThetabins*i_nEbins;
   int i=0, j=0, k=0, is=0;
   int written = 0;
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

   //=============== saving distribution ===============

   double phase = domain()->GetPhase();

   sprintf(fname,"v3d_synchrotron_%5.5d.h5",i_Save++);
   domain()->out_Flog << "SAVE SYNCHROTRON: Opening file " << fname << "\n";
   domain()->out_Flog.flush();

   if (nPE == 0) {
      double *fdata = new double[i_nEbins*i_nThetabins*i_nPhibins];
      double *fdataAccumulated = new double[i_nEbins*i_nThetabins*i_nPhibins];
      for (i=0; i<i_nPhibins*i_nThetabins*i_nEbins; i++) {
         fdataAccumulated[i] = fdata[i] = p_PhotonsArray3D[i];
      }

      domain()->out_Flog << "SAVE SYNCHROTRON: Opening file " << fname << "\n";
      domain()->out_Flog.flush();
      file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert (file >= 0);

      dimsfi[0] = 1;
      written = WriteHDF5Recorddouble(file, "phase", dimsfi[0], &phase);
      assert (written >= 0);
      written = WriteHDF5Recorddouble(file, "f_Emin", dimsfi[0], &f_Emin);
      assert (written >= 0);
      written = WriteHDF5Recorddouble(file, "f_Emax", dimsfi[0], &f_Emax);
      assert (written >= 0);
      written = WriteHDF5Recorddouble(file, "f_EStepFactor", dimsfi[0], &f_EStepFactor);
      assert (written >= 0);
      written = WriteHDF5Recorddouble(file, "f_PhiStep", dimsfi[0], &f_PhiStep);
      assert (written >= 0);
      written = WriteHDF5Recorddouble(file, "f_ThetaStep", dimsfi[0], &f_ThetaStep);
      assert (written >= 0);

      written = WriteHDF5Record(file, "i_nEbins", dimsfi[0], &i_nEbins);
      assert (written >= 0);
      written = WriteHDF5Record(file, "i_nThetabins", dimsfi[0], &i_nEbins);
      assert (written >= 0);
      written = WriteHDF5Record(file, "i_nPhibins", dimsfi[0], &i_nEbins);
      assert (written >= 0);

      //============ E axis ============

      dimsfi[0] = i_nEbins;
      double *Xaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Xaxis[i] = exp(i*f_EStepFactor + f_LogEmin);
      }
      written = WriteHDF5Recorddouble(file, "Energy", dimsfi[0], Xaxis);
      assert (written >= 0);

      delete[] Xaxis;

      //============ Theta axis ============

      dimsfi[0] = i_nThetabins;
      double *Yaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Yaxis[i] = i*f_ThetaStep;
      }
      written = WriteHDF5Recorddouble(file, "Theta", dimsfi[0], Yaxis);
      assert (written >= 0);
      delete[] Yaxis;

      //============ Z axis ============

      dimsfi[0] = i_nPhibins;
      double *Zaxis = new double[dimsfi[0]];
      for (i=0; i<dimsfi[0]; i++) {
         Zaxis[i] = f_PhiStep*i - PI;
      }
      written = WriteHDF5Recorddouble(file, "Z", dimsfi[0], Zaxis);
      assert (written >= 0);
      delete[] Zaxis;

      int nprocessor = 0;
#ifdef V_MPI
      for (nprocessor=1; nprocessor<nPEs; nprocessor++) {
         int nsend = i_nEbins*i_nThetabins*i_nPhibins;
         int fromN = nprocessor;
         int msgtag = 10000 + nprocessor;
         MPI_Status status;
         int ierr = MPI_Recv(fdata, nsend*sizeof(double), MPI_BYTE, fromN, msgtag, MPI_COMM_WORLD, &status);
         for (i=0; i<i_nPhibins*i_nThetabins*i_nEbins; i++) {
            fdataAccumulated[i] += fdata[i];
         }
      }
#endif
      assert (file >= 0);

      dimsf[0] = i_nPhibins;
      dimsf[1] = i_nThetabins;
      dimsf[2] = i_nEbins;

      /* Creating the dataspace                                         */
      fdataspace = H5Screate_simple(rank3, dimsf, NULL); 
      assert (fdataspace >= 0);

      /* Creating the dataset within the dataspace                      */
      fdataset = H5Dcreate(file, "Synchrotron3D", H5T_IEEE_F64LE, fdataspace,
			   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert (fdataset >= 0);

      /* Writing the data to the dataset                                */
      status = H5Dwrite(fdataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
         H5P_DEFAULT, fdata);
      assert (status >= 0);
      status = H5Dclose(fdataset);
      assert (status >= 0);

      // Close the datefile                                             */
      status = H5Fclose(file);
      assert (status >= 0);

      // Close the dataspace                                              */
      status = H5Sclose(fdataspace);
      assert (status >= 0);

      delete[] fdata;
      delete[] fdataAccumulated;

   } else {
#ifdef V_MPI
      int nsend = i_nEbins*i_nThetabins*i_nPhibins;
      int toN = 0;
      int msgtag = 10000 + nPE;
      MPI_Status status;
      int ierr = MPI_Bsend(p_PhotonsArray3D, nsend*sizeof(double), MPI_BYTE, toN, msgtag, MPI_COMM_WORLD);
#endif
   }

   return ldump;
}
