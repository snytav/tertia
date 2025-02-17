#ifndef H_CONTROLS
#define H_CONTROLS

#include <stdio.h>



#include "vlpl3dclass.h"

//---------------------------- Controls class -----------------------

class Controls : public NList {
   friend class Domain;
private:
   Domain *domain() {return Domain::p_D;};

   NList* ToSave;

   char c_Begin[4]; // begin of variables
   double f_Phase;
   double f_CPU;
   double f_CPU_start;
   double f_CPU_finish;
   time_t t_CPU_start;
   time_t t_CPU_finish;
   time_t t_WallClockStart;
   time_t t_WallClock;
   double f_WallClockETA;
   double f_WallClockElapsed;

   long l_Nstep;

   int i_Reload;
   int i_Movie;
   int i_MovieH5;
   int i_Shift;

   int i_Nwrite;
   double f_CPUstop;
   double f_PhaseStop;

   double f_SavePeriod;
   double f_SaveTime;

   double f_WritePeriod;
   double f_WriteTime;

   double f_RefreshPeriod;
   double f_RefreshTime;

   double f_MoviePeriod;
   double f_MovieTime;

   double f_MoviePeriodH5;
   double f_MovieTimeH5;
   int i_MovieFrameH5;

   double f_FieldFilterPeriod;
   double f_FieldFilterTime;
   int i_FieldFilterFlag;

   double f_WakeControlPeriod;
   double f_WakeControlTime;
   int i_WakeControlFlag;

   int i_ShiftPad;
   long l_Nshifted;
   double f_ShiftedDistance;
   double f_ShiftPeriod;
   double f_ShiftTime;
   double f_FirstShiftTime;
   double f_LastShiftTime;

   int i_Ndiagnose;
   char c_End[4]; // end of variables

   //<SergK>
   int i_PostProcessing;
   //</SergK>


public:


   int Reload() {return i_Reload;};
   int PostProcessing() {return i_PostProcessing;};

   double GetCPU(void);
   double GetWallClockElapsed(void) { return f_WallClockElapsed;};
   double GetWallClockETA(void) { return f_WallClockETA;};
   double GetPhase(void) { return f_Phase;};
   void Step();
   long Save(FILE* pFile);
   long Load(FILE* pFile);
   long LoadDummy(FILE* pFile);
   int PhaseStop(void);
   int CPUStop(void);
   int PhaseSave(void);
   int PhaseMovie(void);
   int PhaseMovieH5(void);
   int PhaseFieldFilter(void);
   int PhaseWakeControl(void);
   int IfShift(void) {return i_Shift && f_Phase > f_ShiftTime;};
   long Shift(void);
   long GetShift() {return l_Nshifted;};
   int GetNwrite(void) {return i_Nwrite;};
   int GetMovieFrameH5(void) {return i_MovieFrameH5;};
   double GetShiftPeriod() const {return f_ShiftPeriod;};
   int GetShiftPad() const {return i_ShiftPad;};
   double GetSavePeriod() const {return f_SavePeriod;};
   double GetFilterPeriod() {return f_FieldFilterPeriod;}
   int GetFilterFlag() {return i_FieldFilterFlag;}
   double GetWakeControlPeriod() {return f_WakeControlPeriod;}
   int GetWakeControlFlag() {return i_WakeControlFlag;}
   long GetNstep(){return l_Nstep;}

   void Seti_Nwrite(const int& new_value){i_Nwrite = new_value;};  // See vlpl3d.C

   Controls (char *nm, FILE *f);
};

#endif

