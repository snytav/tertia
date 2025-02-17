#include <stdio.h>
#include <stdlib.h>


#include "vlpl3d.h"

//---Domain:: --------------------->

static double fCPU;

int Domain::Diagnose(void)
{
  long i, j, k, n;
  int diagNe = 0;
  int tmp_n;
  double Eem = 0.;
  double Epart = 0.;
  
  out_Flog << "-----------------------"<<GetmyPE()<<"-----------\n";
  out_Flog << "Time = "<<p_Cntrl->GetPhase()<<"\n";
  out_Flog << "We have " << pa_Species[0]->GetNp() << " electrons \n";
  if (i_Nsorts)
  {
    for (i=1; i<i_Nsorts; i++)
    {
      out_Flog << "We have " << pa_Species[i]->GetNp() << " ions of sort "
	       <<i<<"\n";
    }
  }
  Particle *p = new Particle();
  out_Flog << "Number of ionization events is " << p->GetNionized() << endl;
  delete p;

  out_Flog << "Nel processed = "<<p_M->l_Processed<<"\n";

  if (p_Cntrl->GetPhase() == 0.)
  {
    out_Flog << "hx="<<f_Hx<<"\n";
    out_Flog << "hy="<<f_Hy<<"\n";
    out_Flog << "hz="<<f_Hz<<"\n";
    out_Flog << "ts="<<f_Ts<<"\n";
    out_Flog << "xlength="<<f_Xlength<<"\n";
    out_Flog << "ylength="<<f_Ylength<<"\n";
    out_Flog << "zlength="<<f_Zlength<<"\n";
    out_Flog << "Nsorts="<<i_Nsorts<<"\n";

    out_Flog << "p_Cntrl->start="<<p_Cntrl->i_Reload<<"\n";
    out_Flog << "p_Cntrl->n_diagnose="<<p_Cntrl->i_Ndiagnose<<"\n";
    out_Flog << "p_Cntrl->write_N="<<p_Cntrl->i_Nwrite<<"\n";
    out_Flog << "p_Cntrl->cpu_stop="<<p_Cntrl->f_CPUstop<<"\n";
    out_Flog << "p_Cntrl->phase_stop="<<p_Cntrl->f_PhaseStop<<"\n";
    out_Flog << "p_Cntrl->save_period="<<p_Cntrl->f_SavePeriod<<"\n";
    out_Flog << "p_Cntrl->write_period="<<p_Cntrl->f_WritePeriod<<"\n";
    out_Flog << "p_Cntrl->shift_period="<<p_Cntrl->f_ShiftPeriod<<"\n";
    out_Flog << "p_Cntrl->first_shift_time="<<p_Cntrl->f_FirstShiftTime<<"\n";

    out_Flog << "pa_Pulses[0]->a="<<pa_Pulses[0]->f_A<<"\n";
    out_Flog << "pa_Pulses[0]->polY="<<pa_Pulses[0]->f_Ypol<<"\n";
    out_Flog << "pa_Pulses[0]->polZ="<<pa_Pulses[0]->f_Zpol<<"\n";
    out_Flog << "pa_Pulses[0]->length="<<pa_Pulses[0]->f_Length<<"\n";
    out_Flog << "pa_Pulses[0]->widthY="<<pa_Pulses[0]->f_Ywidth<<"\n";
    out_Flog << "pa_Pulses[0]->widthZ="<<pa_Pulses[0]->f_Zwidth<<"\n";
    out_Flog << "pa_Pulses[0]->rise="<<pa_Pulses[0]->f_Rise<<"\n";
    out_Flog << "pa_Pulses[0]->drop="<<pa_Pulses[0]->f_Drop<<"\n";
    out_Flog << "pa_Pulses[0]->centerX="<<pa_Pulses[0]->f_Xcenter<<"\n";
    out_Flog << "pa_Pulses[0]->centerY="<<pa_Pulses[0]->f_Ycenter<<"\n";
    out_Flog << "pa_Pulses[0]->centerZ="<<pa_Pulses[0]->f_Zcenter<<"\n";

    out_Flog << "pa_Species[0]->density="<<pa_Species[0]->f_Narb<<"\n";
    out_Flog << "pa_Species[0]->begin="<<pa_Species[0]->f_Begin<<"\n";
    out_Flog << "pa_Species[0]->end="<<pa_Species[0]->f_End<<"\n";
    out_Flog << "pa_Species[0]->plateau_begin="<<pa_Species[0]->f_PlateauBegin<<"\n";
    out_Flog << "pa_Species[0]->plateau_end="<<pa_Species[0]->f_PlateauEnd<<"\n";
    out_Flog << "pa_Species[0]->p_perCell="<<pa_Species[0]->l_perCell<<"\n";
    out_Flog << "pa_Species[0]->px="<<pa_Species[0]->f_Px0<<"\n";
    out_Flog << "pa_Species[0]->px="<<pa_Species[0]->f_Py0<<"\n";
    out_Flog << "pa_Species[0]->px="<<pa_Species[0]->f_Pz0<<"\n";

    long  mx,  my,  mz,  dx,  dy,  dz;
    p_M->GetSizes(mx, my, mz, dx, dy, dz);
    out_Flog << "We have a mesh of " << mx << "x" << my << "x" << mz 
	 << " size \n";
    out_Flog << "dx=" << dx << " dy=" << dy << " dz=" << dz <<" \n";

    p_M->GetOffsets(mx, my, mz);
    out_Flog << "offset x,y,z=" << mx << my << mz <<" \n";
  }

#ifdef X_ACCESS 
  if (plot->nview) plot->show();
#endif

  if (i_Nsorts>1)
  	out_Flog<<"We have "<<pa_Species[1]->GetNp()<<" ions \n";
  double jx=0., jy=0., jz=0.;

  for ( k=0; k<p_M->l_Mz; k++)
  {
    for ( j=0; j<p_M->l_My; j++)
    {
      i = 2;
      Cell &ca = p_M->GetCell(i,j,k);
      /*
      out_Flog << "Ex= " << ca.GetEx() << " Ey= " << ca.GetEy() 
	       << " Ez= " << ca.GetEz() << " Bx= " << ca.GetBx() 
	       << " By= " << ca.GetBy() << " Bz= " << ca.GetBz() 
	       << " i="<<i<<" j="<<j<<" k="<<k<<"\n";
      */
      for (i=0; i<p_M->l_Mx; i++)
      {
      	Cell *c = &p_M->GetCell(i,j,k);
				tmp_n = c->PCount();
				diagNe = diagNe + tmp_n;
	/*
	if (tmp_n) {
	  Particle *p = c->GetParticles();
	  while (p) {
	  double x, y, z, px, py, pz;
	  p->GetX(x,y,z);
	  p->GetP(px,py,pz);
	  out_Flog << "Electron: x="<<x<<" y="<< y <<" z="<<z<<
	    " px="<<px<<" py="<<py << " pz="<<pz <<"\n";
	  out_Flog << "In cell i="<<i<<" j="<<j<<" k="<<k<<"\n";;
	  p = p->p_Next;
	  }
	}
	*/
			Eem += c->GetIntensityNorm();
			Epart += c->GetTemperature();
      }
    }
  }
  out_Flog<<"Eem = " << Eem << " Epart =" <<Epart <<" Etot="<<Eem+Epart<<"\n";
  p_M->TotalCurrents();
  out_Flog<<"Jx="<<p_M->f_Jx<<" Jy="<<p_M->f_Jy<<" Jz="<<p_M->f_Jz
	  <<" Density="<<p_M->f_Dens<<"\n";
  
  out_Flog << "Actual Ne = " << diagNe << " electrons \n";
  out_Flog<<"GammaMax = " << p_M->f_GammaMax <<"\n";

  out_Flog << "CPU time is: " << p_Cntrl->GetCPU() << "\n";
  double fCPUold = fCPU;
  fCPU = p_Cntrl->GetCPU();
  if (p_Cntrl->l_Nstep && diagNe) {
  	out_Flog << "Performance is: "
       << (fCPU-fCPUold)/(diagNe*p_Cntrl->i_Ndiagnose)*1e6
       << "us per electron \n";
  }

  double wc = p_Cntrl->GetWallClockElapsed();
  out_Flog<<"Processor load efficiency: " << fCPU/wc*100. << "%"<<endl;
  int mins = wc/60.;
  int hours = wc/3600.;
  int days = wc/24./3600;
  while (hours > 23) {
     hours -= 24;
  }
  while (mins > 59) {
     mins -= 60;
  }

  double wcETA = p_Cntrl->GetWallClockETA();
  int minsETA = wcETA/60.;
  int hoursETA = wcETA/3600.;
  int daysETA = wcETA/24./3600;
  while (hoursETA > 23) {
     hoursETA -= 24;
  }
  while (minsETA > 59) {
     minsETA -= 60;
  }

  double wcLEFT = wcETA - wc;
  int minsLEFT = wcLEFT/60.;
  int hoursLEFT = wcLEFT/3600.;
  int daysLEFT = wcLEFT/24./3600;
  while (hoursLEFT > 23) {
     hoursLEFT -= 24;
  }
  while (minsLEFT > 59) {
     minsLEFT -= 60;
  }
  out_Flog<<"Elapsed wall clock time:  " 
     << days << " days " << hours << " hours " << mins << " mins" <<endl;
  out_Flog<<"Estimated total simulation time: " 
     << daysETA << " days " << hoursETA << " hours " << minsETA << " mins" <<endl;

	if ( GetMPP()->GetnPE() == 0 )	{
	  cout <<"\rt="<<GetPhase() <<" Remaining time: " 
     << daysLEFT << " days " << hoursLEFT << " hours " << minsLEFT << " mins  ";
     cout.flush();
   };

  out_Flog<<"========================================================="<<endl;
  out_Flog<<"================End of Domain::Diagnose=================="<<endl;
  out_Flog.flush();
  return -1;
/*
  k=p_M->l_Mz/2;
  j=p_M->l_My/2;
  i = 2;
  for (i=0; i<p_M->l_Mx; i++) {
    Cell c = p_M->GetCell(i,j,k);
    tmp_n = c.PCount();
    //   out_Fig8<<"i= " << i <<" ne= " << tmp_n << " dens=" <<c.GetDens();
    Particle *p = c.p_Particles;
    while (p) {
      out_Fig8<<" x=" <<p->f_X<<" px="<<p->f_Px;
      p = p->p_Next;
    }
    out_Fig8<<endl;
  }
  return -1;
*/
}
/*
	void Domain::WholeDistribution(void)
	{
		  for ( int k=0; k<p_M->l_Mz; k++)
		  {
		    for ( int j=0; j<p_M->l_My; j++)
    		{
		      for (int i=0; i<p_M->l_Mx; i++)
    		  {
		      	Cell *c = &p_M->GetCell(i,j,k);
						for(int l = 0; l < 100; l++)
							DistributionArray[l] += c->Distribution[l];
		      }
    		}
		  }
	};

	bool Domain::WriteDistribution(void)
	{
		short dims[3];
		dims[0] = 100;
		dims[1] = 10;
		dims[3] = 0;
		int veclen = 1;
		double x0 = 0;
		double y0 = 0;
		double z0 = 0;
		double dx = 3;
		double dy = 1;
		double dz = 0;
		FILE* iris_file;
		iris_file=fopen("iris3db.dat","wb");
	  if (iris_file == NULL) return 1;
	
	  if (fwrite(dims, sizeof(short), 1, iris_file)==NULL) return -1;
	  if (fwrite(dims+1, sizeof(short), 1, iris_file)==NULL) return -2;
	  if (fwrite(dims+2, sizeof(short), 1, iris_file)==NULL) return -3;
	
	  if (fwrite(&veclen, sizeof(short), 1, iris_file)==NULL) return -4;
	  if (fwrite(&x0, sizeof(double), 1, iris_file)==NULL) return -5;
	  if (fwrite(&y0, sizeof(double), 1, iris_file)==NULL) return -6;
	  if (fwrite(&z0, sizeof(double), 1, iris_file)==NULL) return -7;

	  if (fwrite(&dx, sizeof(double), 1, iris_file)==NULL) return -8;
	  if (fwrite(&dy, sizeof(double), 1, iris_file)==NULL) return -9;
		if (fwrite(&dz, sizeof(double), 1, iris_file)==NULL) return -10;

	  if (fwrite(DistributionArray, sizeof(double), dims[0]*veclen, iris_file)==NULL) return -11;
		fclose(iris_file);
		return true;
	};

	void Domain::GetDistribution(void)
	{
		for(int i = 0; i < 100; i++)
			DistributionArray[i] = 0;
			
	  for ( int k=0; k<p_M->l_Mz; k++)
	  {
	    for ( int j=0; j<p_M->l_My; j++)
   		{
	      for (int i=0; i<p_M->l_Mx; i++)
   		  {
	      	Cell *c = &p_M->GetCell(i,j,k);
	      	c->GetDistribution(0);
	      }
   		}
	  }

	};
*/
