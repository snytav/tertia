
/* HDF5 Library                                                       */ 
#include "myhdfshell.h"

/* System libraries to include                                        */
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

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

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ SAVING PART $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

//--- Domain::Save ----------------------->
long Domain::Save()
{
  char sFile[128];
  long ldump=0;
  int i=0;
  int rank = GetmyPE();
  int isave = p_Cntrl->i_Nwrite;

  sprintf(sFile,"%s//vs%3.3dpe%3.3d_3d.dat",str_DataDirectory,isave,rank);
  out_Flog << "SAVE: Opening file " << sFile << "\n";

  FILE* pFile=fopen(sFile,"wb");

  swrite(tbuf);

  ldump += p_MPP->Save(pFile);      //
  ldump += p_CGS->Save(pFile);
  ldump += p_Cntrl->Save(pFile);

  for (i=0; i<i_Npulses; i++)
    ldump += pa_Pulses[i]->Save(pFile);

  ldump += pa_Species[0]->Save(pFile);

  for (i=1; i<i_Nsorts; i++)
    ldump += pa_Species[i]->Save(pFile);

  ldump += p_BndXm->Save(pFile);
  ldump += p_BndXp->Save(pFile);
  ldump += p_BndYm->Save(pFile);
  ldump += p_BndYp->Save(pFile);
  ldump += p_BndZm->Save(pFile);
  ldump += p_BndZp->Save(pFile);

  ldump += p_Synchrotron->Save(pFile);

  if (p_MovieFrame) ldump += p_MovieFrame->Save(pFile);
  if (p_MovieFrameH5) ldump += p_MovieFrameH5->Save(pFile);

  tbuf_length = strlen(tbuf);
  sprintf(tbuf+tbuf_length,"$");

  tbuf_length = strlen(tbuf);
  ldump += fwrite(tbuf,1,tbuf_length,pFile);
  fclose(pFile);
  pFile = NULL;
  out_Flog << ldump << " bytes written" << "\n";

  ldump += p_M->Save(isave);
  ldump += p_Synchrotron->StoreDistributionHDF5();
  return ldump;
}

//--- Partition::Save ----------------------->
long Partition::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}

//--- CGS::Save ----------------------->
long UnitsCGS::Save(FILE* pFile)
{
  return 0;
}

//--- Controls::Save ----------------------->
long Controls::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  ToSave->swrite(tbuf+tbuf_length);
  return ldump;
}

//--- Specie::Save ----------------------->
long Specie::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}

//--- IonSpecie::Save ----------------------->
long IonSpecie::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);

  return ldump;
}

//--- Boundary::Save ----------------------->
long Boundary::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}

//--- Pulse::Save ----------------------->
long Pulse::Save(FILE* pFile)
{
  long ldump=0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}

//---------------------------- MovieFrame::Save -----------------------
long MovieFrame::Save(FILE* pFile)
{
  long ldump = 0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}
//---------------------------- Synchrotron::Save -----------------------
long Synchrotron::Save(FILE* pFile)
{
  long ldump = 0;

  tbuf_length = strlen(tbuf);
  swrite(tbuf+tbuf_length);
  return ldump;
}

//--- Mesh::Save ----------------------->
long Mesh::Save(int isave)
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



  //=============== saving particles ===============

  hid_t* particles_dataset = new hid_t[nsorts];
  hid_t* particles_dataspace = new hid_t[nsorts];
  hid_t* particles_mem_dataspace = new hid_t[nsorts];

  for (is=0; is<nsorts; is++) {
    npart[is]=domain()->GetSpecie(is)->GetNp();
    domain()->out_Flog << "SAVE: " << npart[is] << " particles of sort "
		       <<is<< "\n";
  }

  sprintf(fname,"%s//vs%3.3d_3d_particles.h5",domain()->str_DataDirectory,isave);
  domain()->out_Flog << "SAVE: Opening file " << fname << "\n";

  dimsfi[0] = nsorts;

  char idataname[128];
  sprintf(idataname,"Nparticles_at_PE_%d",nPE);

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
  for (is=0; is<nsorts; is++) {
    dimsp[0] = npart[is];
    pcount[is] = 0;
    if (dimsp[0] > 0) {
      particles_dataspace[is] = H5Screate_simple(rank1, dimsp, NULL);
      assert(particles_dataspace[is] >= 0);
      particles_mem_dataspace[is] = H5Screate_simple(rank1, dimsp, NULL);
      assert(particles_mem_dataspace[is] >= 0);
    }
  }


  if (nPE == 0) {
    file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert (file >= 0);
    status = H5Fclose(file);
    assert (status >= 0);
  }

  int nParallelStreams = 1;
  int nprocessor = 0;

  for (nprocessor=0; nprocessor<nPEs; nprocessor+=nParallelStreams) {
#ifdef V_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (nprocessor < nPE || nprocessor > nPE+nParallelStreams-1) continue;

    char fnameZ[128];
    sprintf(fnameZ,"%s//vs%3.3d_3d_zombies.dat",
	    domain()->str_DataDirectory,isave);
    domain()->out_Flog << "SAVE zombies in file " << fnameZ << "\n";
    FILE *fZombies = fopen(fnameZ,"ab");
    domain()->p_BndXm->SaveZombies(fZombies);
    domain()->p_BndXp->SaveZombies(fZombies);
    domain()->p_BndYm->SaveZombies(fZombies);
    domain()->p_BndYp->SaveZombies(fZombies);
    domain()->p_BndZm->SaveZombies(fZombies);
    domain()->p_BndZp->SaveZombies(fZombies);
    fclose(fZombies);

    domain()->out_Flog << "SAVE tries to re-open file " << fname << "\n";
    domain()->out_Flog.flush();
    file = H5Fopen (fname, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file <= 0) {
      domain()->out_Flog << "SAVE re-open: no such file " << fname << "\n";
      domain()->out_Flog.flush();
    }
    assert (file >= 0);
   
    for (is=0; is<nsorts; is++) {
      domain()->out_Flog << "SAVE: we have to save "<<npart[is]<<
	" particles of sort "<<is << endl;
    };

    SavedIon **pi_array = new SavedIon*[nsorts];
    SavedParticle *e_array = NULL;

    char pdataname[128];
    sprintf(pdataname,"Electrons_at_PE_%d",nPE);
    if (npart[0] > 0) {
      e_array = new SavedParticle[npart[0]];
      particles_dataset[0] = H5Dcreate(file, pdataname, electron_tid, 
				       particles_dataspace[0], 
				       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert (particles_dataset[0] >= 0);
    };

    for (is=1; is<nsorts; is++) {
      pi_array[is] = new SavedIon[npart[is]];
      sprintf(sortname,"Ions_of_sort_%2.2d_at_PE_%d",is,nPE);
      dimsp[0] = npart[is];
      if (npart[is] < 1) continue;
      particles_dataset[is] = H5Dcreate(file, sortname, ion_tid, 
					particles_dataspace[is], 
					H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(particles_dataspace[is] >= 0);
    }

    for (i=0; i<3; i++) {
      offset[i] = 0;
      count[i] = 1;
    }

    double hx = domain()->GetHx();
    double hy = domain()->GetHy();
    double hz = domain()->GetHz();

    for (k=0; k<l_Mz; k++) {
      for (j=0; j<l_My; j++) {
	i = 0;
	long lccc = GetN(i,j,k);
	for (i=0; i<l_Mx; i++) {
	  Cell &ccc = p_CellArray[lccc++];
	  Particle *p = ccc.p_Particles;
	  while(p) {
	    p->l_Cell = lccc-1;
	    is = p->GetSort();
	    assert(is<nsorts);
	    offset[0] = pcount[is];
	    assert(pcount[is] < npart[is]);
	    pcount[is]++;

	    status = H5Sselect_hyperslab(particles_dataspace[is], H5S_SELECT_SET, 
					 offset, NULL, count, NULL);
	    assert (status >= 0);
	    hsize_t mem_offset[3];
	    mem_offset[0] = mem_offset[1] = mem_offset[2] = 0;
	    status = H5Sselect_hyperslab(particles_mem_dataspace[is], H5S_SELECT_SET, 
					 mem_offset, NULL, count, NULL);
	    assert (status >= 0);

	    if (is==0) {
	      SavedParticle *e = &e_array[offset[0]];
	      *e = *p;
	      e->f_X += FullI(i);
	      e->f_Y += FullJ(j);
	      e->f_Z += FullK(k);
	      e->f_X *= hx;
	      e->f_Y *= hy;
	      e->f_Z *= hz;
	    } else {
	      SavedIon *ion = &pi_array[is][offset[0]];;
	      *ion = *p;
	      ion->f_X += FullI(i);
	      ion->f_Y += FullJ(j);
	      ion->f_Z += FullK(k);
	      ion->f_X *= hx;
	      ion->f_Y *= hy;
	      ion->f_Z *= hz;
	    }
	    //			p->f_X -= addX;

	    p = p->p_Next;
	  }
	}
      }
    }

    if (npart[0]>0) {
      status = H5Dwrite(particles_dataset[0], electron_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
			e_array);
      assert (status >= 0);
      status = H5Dclose(particles_dataset[0]);
      assert (status >= 0);
      domain()->out_Flog << "SAVE: we have saved "<<npart[0]<<
	" electrons"<< endl;
      delete[] e_array;
    };



    for (is=1; is<nsorts; is++) {
      if (npart[is] < 1) continue;
      status = H5Dwrite(particles_dataset[is], ion_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
			pi_array[is]);
      assert (status >= 0);
      delete[] pi_array[is];
      status = H5Dclose(particles_dataset[is]);
      assert (status >= 0);
      domain()->out_Flog << "SAVE: we have saved "<<npart[is]<<
	" ions of sort "<< is << endl;
    }

    for (is=0; is<nsorts; is++) {
      if (npart[is] < 1) continue;
      status = H5Sclose(particles_dataspace[is]);
      assert (status >= 0);
      status = H5Sclose(particles_mem_dataspace[is]);
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

    WriteHDF5RecordLong(file, idataname, nsorts, npart);
    status = H5Fclose(file);
    assert (status >= 0);
  };

  delete[] npart;
  delete[] particles_dataset;
  delete[] pcount;
  delete[] particles_dataspace;
  delete[] particles_mem_dataspace;

  double *fdata = new double[l_Mx*l_My*l_Mz];

  Density(ALLSORTS);

  //=============== saving fields ===============
  double sf_DimFields = Cell::sf_DimFields;
  hid_t dummy_rank;

  sprintf(fname,"%s//vs%3.3d_3d_fields.h5",domain()->str_DataDirectory,isave);
  domain()->out_Flog << "SAVE: Opening file " << fname << "\n";
  domain()->out_Flog.flush();
  double phase = domain()->GetPhase();

  if (nPE == 0) {
    domain()->out_Flog << "SAVE: Opening file " << fname << "\n";
    domain()->out_Flog.flush();
    file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert (file >= 0);

    dimsfi[0] = 16;
    int written = WriteHDF5RecordInt(file, MeshDataName, dimsfi[0], &l_Mx);
    assert (written >= 0);

    dimsfi[0] = 1;
    written = WriteHDF5Recorddouble(file, "phase", dimsfi[0], &phase);
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

  //------------- Ex -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Ex*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Ex,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Ex);
#endif // V_MPI


  //------------- Ey -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Ey*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Ey,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Ey);
#endif // V_MPI


  //------------- Ez -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Ez*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Ez,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Ez);
#endif // V_MPI

  //------------- Bx -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Bx*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Bx,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Bx);
#endif // V_MPI


  //------------- By -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_By*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, By,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, By);
#endif // V_MPI


  //------------- Bz -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Bz*sf_DimFields;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Bz,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Bz);
#endif // V_MPI


  //------------- Jx -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Jx;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Jx,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Jx);
#endif // V_MPI


  //------------- Jy -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Jy;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Jy,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Jy);
#endif // V_MPI


  //------------- Jz -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Jz;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Jz,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Jz);
#endif // V_MPI



  //------------- Dens -------------->

  for (k=0; k<l_Mz; k++) {
    for (j=0; j<l_My; j++) {
      i = 0;
      long lccc = GetN(i,j,k);
      long n = i + l_Mx*(j+l_My*k);
      for (i=0; i<l_Mx; i++) {
	Cell &ccc = p_CellArray[lccc++];
	fdata[n++] = ccc.f_Dens;
      }
    }
  }

#ifdef V_MPI
  SaveFieldParallel(fdata, file, Dens,
		    Xpartition, Ypartition, Zpartition,
		    iPE, jPE, kPE, nPE);
#else // V_MPI
  SaveFieldSequential(fdata, file, Dens);
#endif // V_MPI


  //------------- DensArray -------------->
  for (int isort=0; isort<nsorts; isort++) {
    char DensSort[128];
    sprintf(DensSort,"Dens%d",isort);

    for (k=0; k<l_Mz; k++) {
      for (j=0; j<l_My; j++) {
	i = 0;
	long lccc = GetN(i,j,k);
	long n = i + l_Mx*(j+l_My*k);
	for (i=0; i<l_Mx; i++) {
	  Cell &ccc = p_CellArray[lccc++];
	  fdata[n++] = ccc.f_DensArray[isort];
	}
      }
    }
      
#ifdef V_MPI
    SaveFieldParallel(fdata, file, DensSort,
		      Xpartition, Ypartition, Zpartition,
		      iPE, jPE, kPE, nPE);
#else // V_MPI
    SaveFieldSequential(fdata, file, DensSort);
#endif // V_MPI
  }
 
  if (nPE == 0) {
    status = H5Fclose(file);
    assert (status >= 0);
  };
  delete[] fdata;

  return ldump;
}

///---------------------- single process save -------------------
/*
int Mesh::SaveFieldSequential(double* fdata, hid_t file, char* SetName)
{
  hid_t rank3 = 3;
  hsize_t dimsf[3];
  hsize_t dimsfmem[3];
  hid_t       fdataset;         // File and dataset 
  hid_t       fmemspace;   
  hid_t       fdataspace;   // Dataspace handles
  herr_t      status;                // Error checking
  hid_t dummy_rank;

  hsize_t offset[3];
  hsize_t count[3];

  dimsf[0] = l_Mz;
  dimsf[1] = l_My;
  dimsf[2] = l_Mx;

  if (file <= 0) {
    domain()->out_Flog << "SAVE wrong file " << file << "\n";
    domain()->out_Flog.flush();
  }
  assert (file >= 0);
  fdataspace = H5Screate_simple(rank3, dimsf, NULL); 
  assert (fdataspace >= 0);
  fdataset = H5Dcreate(file, SetName, H5T_IEEE_F64LE, 
		       fdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert (fdataset >= 0);

  status = H5Dwrite(fdataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
		    H5P_DEFAULT, fdata);
  assert (status >= 0);
  status = H5Dclose(fdataset);
  assert (status >= 0);
  status = H5Sclose(fdataspace);
  assert (status >= 0);
  int istatus = status;
  return istatus;
}

*/
///////////-------- sequential save ----------------/////////////

int Mesh::SaveFieldSequential(double* fdata, hid_t file, char* SetName)
{

   hid_t rank1 = 1;
   hid_t rank2 = 2;
   hid_t rank3 = 3;
   hsize_t dimsf[3];
   hsize_t dimsfmem[3];
   hid_t       fdataset;         /* File and dataset            */
   hid_t       fmemspace;   
   hid_t       fdataspace;   /* Dataspace handles           */
   herr_t      status = 0;                /* Error checking              */
   hid_t dummy_rank;

   hsize_t offset[3];
   hsize_t count[3];

   dimsf[0] = l_Mz;
   dimsf[1] = l_My;
   dimsf[2] = l_Mx;

   if (file <= 0) {
      domain()->out_Flog << "SAVE no such file " << file << "\n";
      domain()->out_Flog.flush();
      cout <<  "SAVE re-open: no such file " << file << "\n";
   }
   assert (file >= 0);
   fdataspace = H5Screate_simple(rank3, dimsf, NULL); 
   assert (fdataspace >= 0);
   fdataset = H5Dcreate(file, SetName, H5T_IEEE_F64LE, 
      fdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert (fdataset >= 0);

   dimsfmem[0] = l_Mx;

   fmemspace = H5Screate_simple(rank1, dimsfmem, NULL); 
   assert (fmemspace >= 0);

   domain()->out_Flog << "SAVE field " << SetName << " ";
   domain()->out_Flog.flush();

   double *field1d = new double[l_Mx];

   for (int k=0; k<l_Mz; k++) {
      for (int j=0; j<l_My; j++) { 
         int fromN = 0;
         long atN1d = 0;

         if (fromN == 0) {
            for (int i=0; i<l_Mx; i++) {
               field1d[i] = fdata[i+l_Mx*(j+l_My*k)];
            }
         } else {
         }
         offset[2] = 0;
         offset[1] = j;
         offset[0] = k;

         count[2] = l_Mx;
         count[1] = 1;
         count[0] = 1;
         status = H5Sselect_hyperslab(fdataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
         assert (status >= 0);

         offset[0] = offset[1] = offset[2] = 0;
         count[0] = l_Mx;
         status = H5Sselect_hyperslab(fmemspace, H5S_SELECT_SET, offset, NULL, count, NULL);
         assert (status >= 0);
         // Writing the data to the dataset   

         status = H5Dwrite(fdataset, H5T_NATIVE_DOUBLE, fmemspace, 
            fdataspace,
            H5P_DEFAULT, field1d);
         assert (status >= 0);
         // Close the dataset                                


      }
   }
   status = H5Dclose(fdataset);
   assert (status >= 0);
   status = H5Sclose(fdataspace);
   assert (status >= 0);

   delete[] field1d;

   int istatus = status;

   domain()->out_Flog << " done \n";
   domain()->out_Flog.flush();

   return istatus;
}


///////////-------- parallel save ----------------/////////////

int Mesh::SaveFieldParallel(double* field, hid_t file, char* SetName,
			    int Xpartition, int Ypartition, int Zpartition,
			    int iPE, int jPE, int kPE, int nPE)
{

#ifdef V_MPI
  hid_t rank1 = 1;
  hid_t rank2 = 2;
  hid_t rank3 = 3;
  hsize_t dimsf[3];
  hsize_t dimsfmem[3];
  hid_t       fdataset;         /* File and dataset            */
  hid_t       fmemspace;   
  hid_t       fdataspace;   /* Dataspace handles           */
  herr_t      status = 0;                /* Error checking              */
  hid_t dummy_rank;

  hsize_t offset[3];
  hsize_t count[3];

  dimsf[0] = l_Mz*Zpartition;
  dimsf[1] = l_My*Ypartition;
  dimsf[2] = l_Mx*Xpartition;

  if (nPE == 0) {
    if (file <= 0) {
      domain()->out_Flog << "SAVE no such file " << file << "\n";
      domain()->out_Flog.flush();
      cout <<  "pe=" << nPE << "SAVE re-open: no such file " << file << "\n";
    }
    assert (file >= 0);
    fdataspace = H5Screate_simple(rank3, dimsf, NULL); 
    assert (fdataspace >= 0);
    fdataset = H5Dcreate(file, SetName, H5T_IEEE_F64LE, 
			 fdataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert (fdataset >= 0);

    dimsfmem[0] = l_Mx*Xpartition;

    fmemspace = H5Screate_simple(rank1, dimsfmem, NULL); 
    assert (fmemspace >= 0);
  }
  domain()->out_Flog << "SAVE field " << SetName << " ";
  domain()->out_Flog.flush();
  MPI_Barrier(MPI_COMM_WORLD);
   
  double *field1d = new double[2*l_Mx*Xpartition];
  MPI_Status mstatus;

  if (nPE == 0) {
    for (int kp=0; kp<Zpartition; kp++) {
      for (int jp=0; jp<Ypartition; jp++) {
	for (int k=0; k<l_Mz; k++) {
	  for (int j=0; j<l_My; j++) { 
	    long atNlarge = l_Mx*Xpartition*(j+jp*l_My + l_My*Ypartition*(k+kp*l_My)); 
	    for (int ip=0; ip<Xpartition; ip++) {
	      int fromN = ip + Xpartition*(jp + Ypartition*kp);
	      long atN1d = l_Mx*ip;

	      if (fromN == 0) {
		for (int i=0; i<l_Mx; i++) {
		  field1d[i] = field[i+l_Mx*(j+l_My*k)];
		}
	      } else {
		int msgtag = j + l_My*k;
		/*		domain()->out_Flog << "SAVE field " << SetName 
				   << " waiting for "<< msgtag 
				   <<" from "<< fromN;
		domain()->out_Flog.flush();
		*/
		int ierr = MPI_Recv(&field1d[atN1d], l_Mx*sizeof(double), 
				    MPI_CHAR,
				    fromN, msgtag, MPI_COMM_WORLD, &mstatus);
		/*		domain()->out_Flog << " received \n";
		  domain()->out_Flog.flush();*/
	      }
	       
	    }
	    offset[2] = 0;
	    offset[1] = j + l_My*jp;
	    offset[0] = k + l_Mz*kp;

	    count[2] = l_Mx*Xpartition;
	    count[1] = 1;
	    count[0] = 1;
	    status = H5Sselect_hyperslab(fdataspace, H5S_SELECT_SET, offset, NULL, count, NULL);
	    assert (status >= 0);

	    offset[0] = offset[1] = offset[2] = 0;
	    count[0] = l_Mx*Xpartition;
	    status = H5Sselect_hyperslab(fmemspace, H5S_SELECT_SET, offset, NULL, count, NULL);
	    assert (status >= 0);
	    // Writing the data to the dataset   

	    status = H5Dwrite(fdataset, H5T_NATIVE_DOUBLE, fmemspace, 
			      fdataspace,
			      H5P_DEFAULT, field1d);
	    assert (status >= 0);
	    // Close the dataset                                

	  }
	}
      }
    }
    status = H5Dclose(fdataset);
    assert (status >= 0);
    status = H5Sclose(fdataspace);
    assert (status >= 0);
  } else {
    for (int kp=0; kp<Zpartition; kp++) {
      for (int jp=0; jp<Ypartition; jp++) {
	for (int k=0; k<l_Mz; k++) {
	  for (int j=0; j<l_My; j++) { 
	    long atNlarge = l_Mx*Xpartition*(j+jp*l_My + l_My*Ypartition*(k+kp*l_My)); 
	    for (int ip=0; ip<Xpartition; ip++) {
	      int fromN = ip + Xpartition*(jp + Ypartition*kp);
	      if (fromN == nPE) {
		long atN1d = l_Mx*ip;
		int msgtag = j + l_My*k;
		/*	domain()->out_Flog << "SAVE field " << SetName 
		  << " sending "<< msgtag 
		  <<" to zero \n";
		  domain()->out_Flog.flush();*/
		int ierr = MPI_Ssend(&field[l_Mx*(j+l_My*k)], l_Mx*sizeof(double), 
				     MPI_CHAR, 0, msgtag, MPI_COMM_WORLD);
	      }
	    }
	  }
	}
      }
    }
  }

  delete[] field1d;

  int istatus = status;

  domain()->out_Flog << " done \n";
  domain()->out_Flog.flush();
  MPI_Barrier(MPI_COMM_WORLD);

  return istatus;
#endif
  return -1;
}
