#ifndef H_CELLS
#define H_CELLS

#define SNOPACK  0
#define SPACK_E  1
#define SPACK_B  2
#define SPACK_H  128
#define SPACK_F  3
#define SPACK_J  8
#define SPACK_P  16
#define SPACK_JB  32
#define SPACK_PB  64
//#define SPACK_D  32

#define E_DIM 3
#define B_DIM 3
#define H_DIM 5
#define FLD_DIM (E_DIM+B_DIM)
#define CURR_DIM 4
#define BAV_DIM 9

#define ALLSORTS -1
#define FINISH -127
#define PKILL 1

#include "vlpl3d.h"

//---------------------------- Cell class -----------------------

class Cell
{

   friend class Mesh;
   friend class Cell3D;
   friend class Cell3Dm;
   friend class Domain;

private:
   long l_N;
   double f_Epsilon; // permittivity

   union
   {
      double f_Fields[1];
      double f_Ex;
   };

   double f_Ey, f_Ez, f_Bx, f_By, f_Bz;

   union
   {
      double f_Currents[1];
      double f_Jx;
   };

   double f_Jy, f_Jz, f_Dens;
   double *f_DensArray;

   union
   {
      double f_JBeam[1];
      double f_JxBeam;
   };

   double f_JyBeam, f_JzBeam, f_RhoBeam;

   union
   {
      double f_hData[1];
      double f_PxH;
   };
   double f_PyH;
   double f_PzH;
   double f_DensH;
   double f_DeltaDensH;

//   VComplex vc_A;
//   VComplex vc_A1;

   Particle *p_Particles;
   Particle *p_BeamParticles;
   Particle *p_Hook;
   Particle *p_BeamHook;

public:

   void operator=(Cell &ctmp);

   //	bool Cell::WriteDistribution(void);
   //	void Cell::GetDistribution(int sort);
   //	int Distribution[100];

   static double sf_DimFields, sf_DimCurr, sf_DimDens;
   static int si_Sort;

   Domain *GetDomain(void)
   {
      return Domain::p_D;
   };
   Domain *domain(void)
   {
      return Domain::p_D;
   };

   Mesh *GetMesh(void)
   {
      return GetDomain()->GetMesh();
   };

   Partition *GetMPP(void)
   {
      return GetDomain()->GetMPP();
   };

   Particle* GetParticles(void)
   {
      return p_Particles;
   };
   
   Particle* GetBeamParticles(void)
   {
      return p_BeamParticles;
   };
   

   void AddParticle(Particle* p);
   void AddBeamParticle(Particle* p);

   Particle* RemoveParticle(Particle* p, Particle* prev);
   Particle* RemoveBeamParticle(Particle* p, Particle* prev);
   // pointer to the next particle is returned

   Particle* KillParticle(Particle* p, Particle* prev);
   // pointer to the next particle is returned (not previous!!)
   Particle* KillBeamParticle(Particle* p, Particle* prev);
   // pointer to the next particle is returned (not previous!!)

   long PCount(int psort = ALLSORTS);
   long PKill(int psort = ALLSORTS);

   void RelayParticles(void);

   void PackF(CBuffer *j, int cond, int what);
   void UnPackF(CBuffer *j, int cond, int what);

   long PackP(Boundary *bnd, CBuffer *j, double x = 0., double y = 0., double z = 0., int cond = 0, int sort = ALLSORTS);
   long UnPackP(CBuffer *j, int cond = 0, int sort = ALLSORTS);
   long PackPB(Boundary *bnd, CBuffer *j, double x = 0., double y = 0., double z = 0., int cond = 0, int sort = ALLSORTS);
   long UnPackPB(CBuffer *j, int cond = 0, int sort = ALLSORTS);

   long SaveP(FILE* pFile, double x, double y, double z, int sort);

   //  long N(void);
   //  long I(void){ return i_myI;};
   //  long J(void){ return i_myJ;};
   //  long K(void){ return i_myK;};
   double x(void);
   double y(void);
   double z(void);

   void SetFields(double ex, double ey, double ez, double bx, double by, double bz)
   {
      f_Ex=ex;
      f_Ey=ey;
      f_Ez=ez;
      f_Bx=bx;
      f_By=by;
      f_Bz=bz;
   }
   
   void SetParticlesToNULL() {p_Particles = NULL;}

   void SetFields(double *fields)
   {
      for (int i=0; i<FLD_DIM; i++)
         f_Fields[i]=fields[i];
   };

   void AddFields(double *fields)
   {
      for (int i=0; i<FLD_DIM; i++)
         f_Fields[i]+=fields[i];
   };

   void SetAll(double *fields)
   {
      for (int i=0; i<FLD_DIM+CURR_DIM; i++)
         f_Fields[i]=fields[i];
   };

   long GetN(void) {return l_N;};

   double* GetFields()
   {
      return f_Fields;
   };

   double GetEx(void)
   {
      return f_Ex;
   };

   double GetEy(void)
   {
      return f_Ey;
   };

   double GetEz(void)
   {
      return f_Ez;
   };

   double GetBx(void)
   {
      return f_Bx;
   };

   double GetBy(void)
   {
      return f_By;
   };

   double GetBz(void)
   {
      return f_Bz;
   };


   double GetJx(void)
   {
      return f_Jx;
   };

   double GetJy(void)
   {
      return f_Jy;
   };

   double GetJz(void)
   {
      return f_Jz;
   };

   double GetIntensityNorm(void)
   {
      return	double(0.5) * (f_Ex*f_Ex + f_Ey*f_Ey + f_Ez*f_Ez + f_Bx*f_Bx + f_By*f_By + f_Bz*f_Bz);
   };

   // Graphic Information

   double GetExG(void)

   {
      return f_Ex*sf_DimFields;
   };

   double GetEyG(void)
   {
      return f_Ey*sf_DimFields;
   };

   double GetEzG(void)
   {
      return f_Ez*sf_DimFields;
   };

   double GetBxG(void)
   {
      return f_Bx*sf_DimFields;
   };

   double GetByG(void)
   {
      return f_By*sf_DimFields;
   };

   double GetBzG(void)

   {
      return f_Bz*sf_DimFields;
   };

   double GetIntensityG(void)
   {
      return double(0.5)*(f_Ex*f_Ex + f_Ey*f_Ey + f_Ez*f_Ez + f_Bx*f_Bx + f_By*f_By + f_Bz*f_Bz)*sf_DimFields*sf_DimFields;
   };

   double GetJxG(void)
   {
      return f_Jx*sf_DimCurr;
   };

   double GetJyG(void)
   {
      return f_Jy*sf_DimCurr;
   };

   double GetJzG(void)
   {
      return f_Jz*sf_DimCurr;
   };

   double GetDimFields(void)
   {
      return sf_DimFields;
   };

   float ConvertFieldNum2Rel(float field)
   {
      return field*sf_DimFields;
   }; 
   double GetTemperature(int sort=-1);

   double GetDens(int isort=-1);
   double GetDensG(void);
   double GetEpsilonG(void){ return f_Epsilon;};
   double GetPerturbedDensH(void) { return f_DensH + f_DeltaDensH; };
   double GetDensH(void) { return f_DensH; };
   double GetDeltaDensH(void) { return f_DeltaDensH; };
   double GetRhoBeam(void) { return f_RhoBeam;};
   double GetJxBeam(void)  { return f_JxBeam;};

   double GetDens0(void) { return GetDens(0); };
   double GetDens1(void) { return GetDens(1); };
   double GetDens2(void) { return GetDens(2); };
   double GetDens3(void) { return GetDens(3); };
   double GetDens4(void) { return GetDens(4); };
   double GetDens5(void) { return GetDens(5); };
   double GetDens6(void) { return GetDens(6); };
   double GetDens7(void) { return GetDens(7); };
   double GetDens8(void) { return GetDens(8); };
   double GetDens9(void) { return GetDens(9); };
   double GetDens10(void) { return GetDens(10); };

   void Save(CBuffer *j);
   void Load(CBuffer *j);

   Cell();
   ~Cell();
};

#endif
