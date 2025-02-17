#include <math.h>
#include <iostream>
#include <stdlib.h>
#include "vlpl3d.h"
#include "cell3d.h"

//---------------------------- Mesh::Mesh --------------------
Mesh::Mesh(long mx, long ofX, long my, long ofY, long mz, long ofZ)
{
	SetSizes(mx,my,mz);
	l_offsetX = ofX; 
	l_offsetY = ofY; 
	l_offsetZ = ofZ; 
	l_MovieStarted = 0;
	l_Processed = 0;

	p_CellArray = new Cell[l_sizeXYZ];

	for (long i=0; i<l_sizeXYZ; i++) {
		p_CellArray[i].l_N = i;
	}

	f_aJx = new float[l_Mx+l_My+l_Mz+2*l_dMx];
}

//---------------------------- Mesh::makeit --------------------
void Mesh::MakeIt(void)
{
	long i = 4;
	long j = l_My/2;
	long k = l_Mz/2;
	long n=0;
	//  SeedParticles(i,j,k);
	//  return;
	for (k=0; k<l_Mz; k++) 
		for (j=0; j<l_My; j++) 
			for (i=0; i<l_Mx; i++) {
				SeedParticles(i,j,k);
			}
			for (n=0; n<l_sizeXYZ; n++)
				for (k=0; k<3; k++) {
					p_CellArray[n].f_J0[k]= -p_CellArray[n].f_Currents[k];
					p_CellArray[n].f_Currents[k]=0.;
				}
				return;
				// Debugging 
				for (k=0; k<l_Mz; k++) 
					for (j=0; j<l_My; j++) 
						for (i=0; i<l_Mx; i++) {
							n = GetN(i,j,k);
							Particle *p = p_CellArray[n].p_Particles;
							while (p) {
								float x = X(i);
								p->f_Px = .1;
								p->f_Py = 0;
								p->f_Pz = 0;
								p = p->p_Next;
							}
						}
}

//---------------------------- Mesh::SeedParticles --------------------
void Mesh::SeedParticles(long i, long j, long k)
{
	float xtmp=0., ytmp=0., ztmp=0.;
	long nseed=0, iseed=0;
	float xco = X(i)+domain()->p_Cntrl->GetShift()*Hx();
	float yco = Y(j) - domain()->GetYlength()/2.;;
	float zco = Z(k) - domain()->GetZlength()/2.;;
	float dens = -1.;
	Particle *p=NULL;
	float ts2hx = domain()->GetTs()/(domain()->GetHx());
	float ts2hy = domain()->GetTs()/(domain()->GetHy());
	float ts2hz = domain()->GetTs()/(domain()->GetHz());

	long n = GetN(i,j,k);
	if (n<0 || n> l_sizeXYZ-1) {
		cout << "We have problems with n="<<n<<
			" i=" << i <<" j=" << j <<" k=" << k << endl;
		exit(-15);
	}
	Cell &c = p_CellArray[n];

	int isort = domain()->GetNsorts();
	while (isort--) {
		Specie *spec = domain()->GetSpecie(isort);
		dens = spec->Density(xco,yco,zco);
		float djx0, djy0, djz0;
		spec->GetdJ( djx0, djy0, djz0 );

		if (dens > 0.) {
			nseed = iseed = spec->GetPperCell();
			if (nseed > 0) {
				double fiside=pow(double(nseed-1),1./3.)+1.;
				long iside = (long)(fiside);
				long iside2 = iside*iside;
				while(iseed--) {
					if (nseed == 1) {
						xtmp = 0.5;
						ytmp = ztmp = 0.5;
					} else if (nseed == 2 || nseed == 3) {
						xtmp = (iseed+1.)/(nseed+1.);
						ytmp = ztmp = 0.5;
					} else if (nseed == 4) {
						xtmp = 0.5;
						iside2 = 2;
						ytmp = 1./3.*(1.+iseed%iside2);
						ztmp = 1./3.*(1.+iseed/iside2);
					} else { 
						long iz = iseed/iside2;
						long iy = (iseed-iside2*iz)/iside;
						long ix = iseed-iside*(iy+iside*iz);
						xtmp = 1./(iside+1)*(ix+1.);
						ytmp = 1./(iside+1)*(iy+1.);
						ztmp = 1./(iside+1)*(iz+1.);
					}

					float weight = spec->GetWeight()*dens;
					float q2m = spec->GetQ2M();
					float px0 = spec->GetPx();
					float py0 = spec->GetPy();
					float pz0 = spec->GetPz();

					float spread = spec->GetPspread();
					double random;
					random = spread*(rand()-0.5*RAND_MAX)/RAND_MAX;
					px0 += random;
					random = spread*(rand()-0.5*RAND_MAX)/RAND_MAX;
					py0 += random;
					random = spread*(rand()-0.5*RAND_MAX)/RAND_MAX;
					pz0 += random;


					int type = spec->GetType();
					if (type)  {
						int state=spec->GetState0();

						//	    printf("Mesh::SeedParticles %d\n", isort);

						p = new Ion(p_CellArray+n, isort, state, weight, q2m,
							xtmp, ytmp, ztmp, px0, py0, pz0 );
					} else { 
						p = new Electron(p_CellArray+n, weight, 
							xtmp, ytmp, ztmp, px0, py0, pz0);
						p->GetX(xtmp,ytmp,ztmp);
						p->GetP(px0,py0,pz0);

						float gamma = 1./sqrt(1. + px0*px0 + py0*py0 + pz0*pz0);
						float dx = px0*gamma*ts2hx;
						float dy = py0*gamma*ts2hy;
						float dz = pz0*gamma*ts2hz;
						float djx = weight*djx0;
						float djy = weight*djy0;
						float djz = weight*djz0;
						//	    MoveSimple(i,j,k,xtmp,dx,ytmp,dy,ztmp,dz,djx,djy,djz);
					}
				}
			}
		} 
	}
	return;
}

//---------------------------- Mesh::Shift --------------------
void Mesh::Shift(void)
{
	long i, j, k;
	float f[FLD_DIM+CURR_DIM];
	for (i=0; i<FLD_DIM+CURR_DIM; i++) f[i]=0.;

	char what = SPACK_F;
	Send(domain()->GetBndXm(), what);
	Receive(domain()->GetBndXp(),what);

	for (k=-1; k<l_Mz+1; k++) 
		for (j=-1; j<l_My+1; j++) {
			i = -1;
			long n = GetN(i,j,k);
			long nm = n;
			for (i=0; i<l_Mx+1; i++) {
				nm = n;
				n++;
				p_CellArray[nm] = p_CellArray[n];
			}
		}

		what = SPACK_P+SPACK_F;
		Send(domain()->GetBndXm(), what);
		Receive(domain()->GetBndXp(),what);

		if (domain()->XpEdge()) {
			for (k=0; k<l_Mz; k++) 
				for (j=0; j<l_My; j++) {
					i=l_Mx-1;
					SeedParticles(i,j,k);
				}
		}
}

//---------------------------- Mesh::ClearDensity --------------------
void Mesh::ClearDensity(void) {
	for (long n=0; n<l_sizeXYZ; n++) p_CellArray[n].f_Dens=0.;
}

//---------------------------- Mesh::ClearCurrents --------------------
void Mesh::ClearCurrents(void) {
	for (long n=0; n<l_sizeXYZ; n++) 
		for (long k=0; k<3; k++) {
			p_CellArray[n].f_Currents[k]=p_CellArray[n].f_J0[k];
			//      p_CellArray[n].f_Dens=0.;
		}
}

//---------------------------- Mesh::Density --------------------
void Mesh::Density(int isort)
{
	ClearDensity();
	for (long k=0; k<l_Mz; k++) 
		for (long j=0; j<l_My; j++)
		{
			long i = 0;
			Cell3D c(this,i,j,k);
			for (i=0; i < l_Mx; i++)
			{
				Particle *p = c.XYZ->p_Particles;
				while (p)
				{
					if (isort == ALLSORTS)
						c.AddDensity(p, p->GetSpecie()->GetPolarity());
					else
						if (isort == p->GetSort())
							c.AddDensity(p, 1.);
					p = p->Next();
				}
				c.Next();
			}
		}
		domain()->Exchange(SPACK_J);
}

//---------------------------- Mesh::TotalCurrents --------------------
void Mesh::TotalCurrents()
{
	int ic=0;
	long i, j, k;
	for (ic=0; ic<CURR_DIM; ic++) f_Currents[ic]=0.;

	for (k=0; k<l_Mz; k++) 
		for (j=0; j<l_My; j++) {
			for (i=0; i < l_Mx; i++) {
				Cell &c = GetCell(i,j,k);
				for (ic=0; ic<CURR_DIM; ic++) f_Currents[ic]+=c.f_Currents[ic];
				//	if (c.f_Jx != 0.) domain()->out_Flog << i << j << k << 
				//		    " Jz=" << c.f_Jx << " total=" << f_Jx << "\n";
			}
		}
		i = l_Mx/2+1;
		j = l_My/2;
		for (k=-l_dMz; k < l_Mz+l_dMz; k++) {
			Cell &c = GetCell(i,j,k);
			f_aJx[k+l_dMz] = c.f_Jz;
		}
}

//---------------------------- int Mesh::GetI_from_CellNumber(long n); --------------------
int Mesh::GetI_from_CellNumber(long n) {
	while (n >= l_sizeX) {
		n -= l_sizeX;
	}
	n -= l_dMx;
	return n;
}

//---------------------------- int Mesh::GetJ_from_CellNumber(long n); --------------------
int Mesh::GetJ_from_CellNumber(long n) {
	n /= l_sizeX;
	while (n >= l_sizeY) {
		n -= l_sizeY;
	}
	n -= l_dMy;
	return n;
}

//---------------------------- int Mesh::GetK_from_CellNumber(long n); --------------------
int Mesh::GetK_from_CellNumber(long n) {
	n /= l_sizeXY;
	n -= l_dMz;
	return n;
}


//---------------------------- Mesh::SaveCadrMovie --------------------
void Mesh::SaveCadrMovie(FILE* fout)
{
	long istat=-1;
	GETTYPE getf;
	long i_Hfig = domain()->i_NMovieFrames;
	long i_Vfig = domain()->i_NMovieFrames;
	long i_Nfig = i_Vfig + i_Hfig;
	long ifig = 0;

	float hxw = domain()->GetHx();
	float hyw = domain()->GetHy();
	float hzw = domain()->GetHz();

	if (l_MovieStarted==0)
	{
		l_MovieStarted++;
		fwrite(&l_Mx,sizeof(long),1,fout);
		fwrite(&l_My,sizeof(long),1,fout);
		fwrite(&l_Mz,sizeof(long),1,fout);

		fwrite(&i_Vfig,sizeof(long),1,fout);
		fwrite(&i_Hfig,sizeof(long),1,fout);
		fwrite(&i_Nfig,sizeof(long),1,fout);
	}

	fwrite(&istat,sizeof(long),1,fout);

	float phase=domain()->GetPhase();
	fwrite(&phase,sizeof(float),1,fout);
	float shift = domain()->p_Cntrl->GetShift()*hxw;
	float ShiftPeriod = 0;
	long ShiftPad=0;
	long ShiftN=domain()->p_Cntrl->GetShift();
	fwrite(&shift,sizeof(float),1,fout);
	fwrite(&ShiftPeriod,sizeof(float),1,fout);
	fwrite(&ShiftPad,sizeof(long),1,fout);
	fwrite(&ShiftN,sizeof(long),1,fout);

	for (ifig=0; ifig<i_Hfig; ifig++)
	{
		getf = domain()->p_MovieFrame->Gets[ifig];
		fwrite(&hxw,sizeof(float),1,fout);
		fwrite(&hyw,sizeof(float),1,fout);
		fwrite(&l_Mx,sizeof(long),1,fout);
		fwrite(&l_My,sizeof(long),1,fout);
		long k = l_Mz/2;

		for (long j=0; j<l_My; j++)
			for (long i=0; i<l_Mx; i++)
			{
				Cell &c = GetCell(i,j,k);
				float dum = (c.*getf)();
				fwrite(&dum,sizeof(float),1,fout);
			}
	}

	for (ifig=0; ifig<i_Vfig; ifig++)
	{
		getf = domain()->p_MovieFrame->Gets[ifig];
		fwrite(&hxw,sizeof(float),1,fout);
		fwrite(&hzw,sizeof(float),1,fout);
		fwrite(&l_Mx,sizeof(long),1,fout);
		fwrite(&l_Mz,sizeof(long),1,fout);
		long j = l_My/2;
		for (long k=0; k<l_Mz; k++)
			for (long i=0; i<l_Mx; i++)
			{
				Cell &c = GetCell(i,j,k);
				float dum = (c.*getf)();
				fwrite(&dum,sizeof(float),1,fout);
			}
	}
}

//---------------------------- Mesh::MovieWriteEnabled --------------------
bool Mesh::MovieWriteEnabled(void)
{

	//  int Xpart = domain()->p_MPP->GetXpart();
	int Ypart = domain()->p_MPP->GetYpart();
	int Zpart = domain()->p_MPP->GetZpart();

	//  int iPE = domain()->p_MPP->GetiPE();
	int jPE = domain()->p_MPP->GetjPE();
	int kPE = domain()->p_MPP->GetkPE();

	int odd_k = Zpart % 2;
	int odd_j = Ypart % 2;
	//  float my_i = fmod(Xpart, 2.0);

	int k_p = 1;
	int j_p = 1;
	//  int i_p = 1;

	//long k = 0;

	if(!odd_k)
	{
		// even number
		k_p = (Zpart / 2) - 1;
		//  	k     = l_Mz-1;
	}
	else
	{
		k_p = (Zpart - 1)/2;
		//k     = l_Mz/2;
	};

	//long j = 0;
	/*
	if(!odd_j)
	{
	// even number
	j_p = (Ypart / 2);
	j = 0;
	}
	else

	{
	j_p = (Ypart - 1)/2;
	j = l_My/2;
	};
	*/
	//*
	if(!odd_j)
	{
		// even number
		j_p = (Ypart / 2) - 1;
		//j = l_My-1;
	}
	else

	{
		j_p = (Ypart - 1)/2;
		//j = l_My/2;
	};
	//*/

	/*
	if(!odd_i)
	{
	i_p = (Xpart / 2) - 1;
	}
	else
	{
	i_p = (Xpart - 1)/2;
	}; 	
	*/

	if(kPE != k_p && jPE != j_p)
		return false;

	return true;
}

//---------------------------- Mesh::SaveCadrMovie2 --------------------
int Mesh::SaveCadrMovie2(FILE* fout)
{
	long istat=-1;
	GETTYPE getf;
	long i_Hfig = domain()->i_NMovieFrames;
	long i_Vfig = domain()->i_NMovieFrames;
	long i_Nfig = i_Vfig + i_Hfig;
	long ifig = 0;


	float hxw = domain()->GetHx();
	float hyw = domain()->GetHy();
	float hzw = domain()->GetHz();


	//  int Xpart = domain()->p_MPP->GetXpart();
	int Ypart = domain()->p_MPP->GetYpart();
	int Zpart = domain()->p_MPP->GetZpart();

	//  int iPE = domain()->p_MPP->GetiPE();
	int jPE = domain()->p_MPP->GetjPE();
	int kPE = domain()->p_MPP->GetkPE();

	int odd_k = Zpart % 2;
	int odd_j = Ypart % 2;
	//  float my_i = fmod(Xpart, 2.0);

	int k_p = 1;
	int j_p = 1;
	//  int i_p = 1;

	long k = 0;

	if(!odd_k)
	{
		// even number
		k_p = (Zpart / 2) - 1;
		k     = l_Mz-1;
	}
	else
	{
		k_p = (Zpart - 1)/2;
		k     = l_Mz/2;
	};

	long j = 0;

	if(!odd_j)
	{
		// even number
		j_p = (Ypart / 2) - 1;
		j = l_My-1;
	}
	else

	{
		j_p = (Ypart - 1)/2;
		j = l_My/2;
	};



	if(kPE != k_p && jPE != j_p)
		return 3;

	/*
	if(iPE != i_p)
	break;
	*/

	if (l_MovieStarted==0)
	{
		l_MovieStarted++;
		fwrite(&l_Mx,sizeof(long),1,fout);
		fwrite(&l_My,sizeof(long),1,fout);
		fwrite(&l_Mz,sizeof(long),1,fout);

		fwrite(&i_Vfig,sizeof(long),1,fout);
		fwrite(&i_Hfig,sizeof(long),1,fout);
		fwrite(&i_Nfig,sizeof(long),1,fout);
	}

	fwrite(&istat,sizeof(long),1,fout);

	float phase=domain()->GetPhase();
	fwrite(&phase,sizeof(float),1,fout);
	float shift = domain()->p_Cntrl->GetShift()*hxw;
	float ShiftPeriod = 0;
	long ShiftPad=0;
	long ShiftN=domain()->p_Cntrl->GetShift();
	fwrite(&shift,sizeof(float),1,fout);
	fwrite(&ShiftPeriod,sizeof(float),1,fout);
	fwrite(&ShiftPad,sizeof(long),1,fout);
	fwrite(&ShiftN,sizeof(long),1,fout);

	if(kPE == k_p)
		for (ifig=0; ifig<i_Hfig; ifig++)
		{
			getf = domain()->p_MovieFrame->Gets[ifig];
			fwrite(&hxw,sizeof(float),1,fout);
			fwrite(&hyw,sizeof(float),1,fout);
			fwrite(&l_Mx,sizeof(long),1,fout);
			fwrite(&l_My,sizeof(long),1,fout);

			for (long j=0; j<l_My; j++)
				for (long i=0; i<l_Mx; i++)
				{
					Cell &c = GetCell(i,j,k);
					float dum = (c.*getf)();

					fwrite(&dum,sizeof(float),1,fout);
				}
		}

		if(jPE == j_p)
			for (ifig=0; ifig<i_Vfig; ifig++)
			{
				getf = domain()->p_MovieFrame->Gets[ifig];
				fwrite(&hxw,sizeof(float),1,fout);
				fwrite(&hzw,sizeof(float),1,fout);
				fwrite(&l_Mx,sizeof(long),1,fout);
				fwrite(&l_Mz,sizeof(long),1,fout);

				for (long k=0; k<l_Mz; k++)
					for (long i=0; i<l_Mx; i++)
					{
						Cell &c = GetCell(i,j,k);
						float dum = (c.*getf)();
						fwrite(&dum,sizeof(float),1,fout);
					}
			}

			return 1;
}
