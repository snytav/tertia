#ifndef H_MESH
#define H_MESH

#include "cells.h"
#include "domain.h"
/* HDF5 Library                                                       */ 
#include "myhdfshell.h"

//---------------------------- Mesh class -----------------------
class Mesh{
   friend class Domain;
private:
   Domain *domain() {return Domain::p_D;};

   char c_Begin[4]; // begin of variables
   int l_Mx, l_dMx, l_sizeX, l_offsetX;
   int l_My, l_dMy, l_sizeY, l_offsetY;
   int l_Mz, l_dMz, l_sizeZ, l_offsetZ;
   long l_sizeXY, l_sizeXZ, l_sizeYZ, l_sizeXYZ;
   long l_Processed;
   long l_MovieStarted;
   double f_GammaMax;
   int i_OptimalWakeRecorded;
   double f_WakeZeroPosition;
   double f_RecentWakeZeroPosition;
   double f_WakeCorrection;
   char c_End[4]; // end of variables

   Cell* p_CellArray;
   Cell* p_CellLayerP;
   Cell* p_CellLayerC;
   Cell* p_CellLayerM;
   FieldStencil m_Stencil;
   Particle *p_PrevPart;
   FILE *pf_FileWakeControl;

public:
   union
   {
      double f_Currents[1];
      double f_Jx;
   };

   double f_Jy, f_Jz, f_Dens;

   double *f_aJx;


   bool MovieWriteEnabled(void);

   void MoveFields(void);
   void MoveFieldsLayer(int iLayer, int iInitStep, double part=1.);
   void MoveFieldsHydroLinLayer(int iLayer, int iInitStep=0, double part=1.);
   void MoveLayer(int iLayer);
   void MoveSplitLayer(int iLayer,int iSplit);
   void MoveAllLayers(void);
   void MoveAllSplitLayers(void);
   void MoveFieldsHydro(void);
   void GuessFieldsHydroLinLayer(int iLayer);
   void GuessFieldsHydroLinLayerSplit(int iLayer,int iSplit);
   int  getLayerParticles(int iLayer);
   double IterateFieldsHydroLinLayer(int iLayer);
   double IterateFieldsHydroLinLayerSplit(int iLayer,int iSplit,int N_iter);
   void ExchangeCurrents(int iLayer);
   void ExchangeFields(int iLayer);
   void ExchangeCurrentsSplit(int iLayer);
   void ExchangeFieldsSplit(int iLayer);
   void ExchangeRho(int iLayer);
   void MoveHydroLayer(int iLayer);
   void MoveAllHydroLayers(void);
   double EM_EnergyLayer(int iLayer);
   void MoveParticlesLayer(int iLayer, int iFullStep, double part=1.);
   void MoveParticlesLayerSplit(int iLayer,int iSplit, int iFullStep, double part);


   void MoveBfield(void);
   void MoveParticles(void);
   void MoveBeamParticles(void);
   void AdjustHybridDensity(void);
   void FilterFieldX(void);
   double WakeControl(void);
   void AddWakeCorrection(double dDensityChange);

   Particle* Hook(Particle* p, Particle* prev, Cell *c, Cell* cn);
   Particle* UnHook(Particle* p, Cell &c);
   
   Domain *GetControlDomain(){return domain();}

   double Hx(void)
   {
      return domain()->GetHx();
   };

   double HxSplit(void)
   {
      return domain()->GetHxSplit();
   };
   int GetNxSplit(void)
   {
      return domain()->GetNxSplit();
   };
   int nIter(void)
   {
      return domain()->nIter();
   };

   double Hy(void)
   {
      return domain()->GetHy();
   };

   double Hz(void)
   {
      return domain()->GetHz();
   };

   double Ts(void)
   {
      return domain()->GetTs();
   };

   int MaxwellSolver(void)
   {
      return domain()->GetMaxwellSolver();
   };
   int HybridIncluded(void)
   {
      return domain()->HybridIncluded();
   };

   Cell *get_p_CellLayerP(){return  p_CellLayerP;}
   Cell *get_p_CellLayerC(){return  p_CellLayerC;}

   void InitPulse(Pulse*);
   void SetCellNumbers(void);

   void Send(Boundary *bnd, int what);
   void Receive(Boundary *bnd, int what);
   long UnPackP(CBuffer* buf, int cond=0);
   long UnPackPB(CBuffer* buf, int cond=0);

   void ClearDensity(void);
   void ClearRho(int layer);
   void ClearRhoSplit(void);
   void ClearBeamCurrents(void);
   void ClearCurrents(int layer);
   void ClearCurrentsSplit(void);
   void ClearFields(void);
   void Density(int isort=0);

   template <class C> inline void ExchangeEndian(C& value) const;

   inline long GetN(long i, long j, long k)
   {
      return i + l_dMx + l_sizeX * ( j + l_dMy + l_sizeY * ( k + l_dMz ) );
   };

   inline long GetNyz(long j, long k)
   {
      return ( j + l_dMy + l_sizeY * ( k + l_dMz ) );
   };

   inline long GetMx()
   {
      return l_Mx;
   };

   inline long GetMy()
   {
      return l_My;
   };

   inline long GetMz()
   {
      return l_Mz;
   };

   inline Cell& GetCell(long i, long j, long k)
   {
      return p_CellArray[GetN(i,j,k)];
   };

   inline Cell& GetCell(long n)
   {
      return p_CellArray[n];
   };

   inline Cell& operator[] (long n)
   {
      return p_CellArray[n];
   };

   inline long Xp (long n)
   {
      return n+1;
   };

   inline long Yp (long n)
   {
      return n+l_sizeX;
   };

   inline long Zp (long n)
   {
      return n+l_sizeXY;
   };

   inline long Xm (long n)
   {
      return n-1;
   };

   inline long Ym (long n)
   {
      return n-l_sizeX;
   };

   inline long Zm (long n)
   {
      return n-l_sizeXY;
   };

   void GetSizes(long &x, long &y, long &z, long &dx, long &dy, long &dz)
   {
      x = l_Mx; dx = l_dMx;
      y = l_My; dy = l_dMy;
      z = l_Mz; dz = l_dMz;
   };

   void GetOffsets(long &x, long &y, long &z)
   {
      x = l_offsetX;
      y = l_offsetY;
      z = l_offsetZ;
   };

   void SetSizes(long x, long y, long z, long dx=2, long dy=2, long dz=2)
   {
      l_Mx = x; l_dMx = dx;
      l_My = y; l_dMy = dy;
      l_Mz = z; l_dMz = dz;
      l_sizeX = x + 2*dx;
      l_sizeY = y + 2*dy;
      l_sizeZ = z + 2*dz;
      l_sizeXY = l_sizeX*l_sizeY;
      l_sizeXZ = l_sizeX*l_sizeZ;
      l_sizeYZ = l_sizeY*l_sizeZ;
      l_sizeXYZ = l_sizeX*l_sizeY*l_sizeZ;
   };

   double X(long ntmp)
   {
      return Hx()*double(ntmp+l_offsetX);
   };

   double FullI(long ntmp)
   {
      return double(ntmp+l_offsetX);
   };

   long GetI(double xcoord)
   {
      return long(xcoord/Hx()-l_offsetX);
   };

   double Y(long ntmp)
   {
      return Hy()*double(ntmp+l_offsetY);
   };

   double FullJ(long ntmp)
   {
      return double(ntmp+l_offsetY);
   };

   long GetJ(double ycoord)
   {
      return long(ycoord/Hy()-l_offsetY);
   };

   double Z(long ntmp)
   {
      return Hz()*double(ntmp+l_offsetZ);
   };

   double FullK(long ntmp)
   {
      return double(ntmp+l_offsetZ);
   };

   long GetK(double zcoord)
   {
      return long(zcoord/Hz()-l_offsetZ);
   };

   int GetI_from_CellNumber(long n);
   int GetJ_from_CellNumber(long n);
   int GetK_from_CellNumber(long n);

   void MakeIt(void);

   void DepositCurrentsInCell(   Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho);

   void DepositCurrentsInCellSplit(   Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho,int np);

   void DepositRhoInCell(   Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho);
   void MoveInCell(   Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho);
   void MoveBeamInCell(   Particle *p, int isort,
                      int i, int j, int k, 
                      double Vx, double Vy, double Vz, 
                      double x, double y, double z, 
                      double djx, double djy, double djz, double drho,int np);

   void SeedFrontParticles();
   void SeedBeamParticles(long i, long j, long k);
   void SeedParticles(long i, long j, long k);
   void AverageBfield(void);
   void AverageEfield(void);
   void AverageJfield(void);
   void Radiate(Pulse **pulse, int i);
   void TotalCurrents(void);
   void Shift();

   void SaveCadrMovie(FILE* fmovie);
   int SaveCadrMovie2(FILE* fmovie);
   int Save_Movie_Frame_H5(int ichunk);

   long Save(int isave);
   long Load(int isave);
   int SaveFieldSequential(double* fdata, hid_t file, char* SetName);
   int SaveFieldParallel(double* fdata, hid_t file, char* SetName,
			  int Xpartition, int Ypartition, int Zpartition,
		     int iPE, int jPE, int kPE, int nPE);
   Mesh(long mx, long ofX, long my, long ofY, long mz, long ofZ);
   ~Mesh() { delete[] p_CellArray;};
};

#endif
