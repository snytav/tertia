#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include <time.h>
clock_t clock(void);

#include "vlpl3d.h"

//---------------------------- Controls::Controls -----------------------
Controls::Controls (char *nm, FILE *f) : NList (nm)
{
   AddEntry("Reload", &i_Reload);
   AddEntry("Ndiagnose", &i_Ndiagnose);
   AddEntry("PostProcessing", &i_PostProcessing, 0);
   AddEntry("Nwrite", &i_Nwrite);
   AddEntry("CPUstop", &f_CPUstop);
   AddEntry("PhaseStop", &f_PhaseStop);
   AddEntry("SavePeriod", &f_SavePeriod);
   AddEntry("WritePeriod", &f_WritePeriod);
   AddEntry("MovieFlag", &i_Movie);
   AddEntry("MovieFlagH5", &i_MovieH5, 0);
   AddEntry("MoviePeriod", &f_MoviePeriod);
   AddEntry("MoviePeriodH5", &f_MoviePeriodH5, 1e20);
   AddEntry("ShiftFlag", &i_Shift);
   AddEntry("ShiftPad", &i_ShiftPad, 1);
   AddEntry("ShiftPeriod", &f_ShiftPeriod, 0.);
   AddEntry("FirstShiftTime", &f_FirstShiftTime, 0.);
   AddEntry("LastShiftTime", &f_LastShiftTime, 0.);
   AddEntry("FieldFilterFlag", &i_FieldFilterFlag);
   AddEntry("FieldFilterTime", &f_FieldFilterTime, 1.);
   AddEntry("FieldFilterPeriod", &f_FieldFilterPeriod, 1.);
   AddEntry("WakeControlFlag", &i_WakeControlFlag);
   AddEntry("WakeControlTime", &f_WakeControlTime, 1.);
   AddEntry("WakeControlPeriod", &f_WakeControlPeriod, 1.);

   ToSave = new NList(nm);
   ToSave->AddEntry("f_Phase", &f_Phase);
   ToSave->AddEntry("l_Nshifted", &l_Nshifted);
   ToSave->AddEntry("f_ShiftedDistance", &f_ShiftedDistance);
   ToSave->AddEntry("f_ShiftTime", &f_ShiftTime);
   ToSave->AddEntry("FirstShiftTime", &f_FirstShiftTime);
   ToSave->AddEntry("f_MovieTime", &f_MovieTime);
   ToSave->AddEntry("f_MovieTimeH5", &f_MovieTimeH5);
   ToSave->AddEntry("i_MovieFrameH5", &i_MovieFrameH5);
   ToSave->AddEntry("f_SaveTime", &f_SaveTime);


   if (f)
   {
      rewind(f);
      read(f);
   }

#ifdef V_MPI
   domain()->GetBufMPP()->reset();
   pack_nls(domain()->GetBufMPP());
   domain()->BroadCast(domain()->GetBufMPP());
   if (f==NULL)
      unpack_nls(domain()->GetBufMPP());
#endif

   f_Phase = 0.;
   l_Nstep = 0;
   f_CPU = 0.;
   f_CPU_start = 0.;
   f_CPU_finish = 0.;

   f_CPU_start = clock();
   time( &t_WallClockStart );
   time( &t_WallClock );

   f_SaveTime = f_SavePeriod;
   f_MovieTime = f_MoviePeriod;
   l_Nshifted = 0;
   if (f_ShiftPeriod == 0.)
      f_ShiftPeriod = domain()->GetHx();
   f_ShiftTime = f_FirstShiftTime;
   if (f_LastShiftTime == 0) {
     f_LastShiftTime = f_PhaseStop;
   };
}

//--- Controls:: ----------------------->
double Controls::GetCPU(void)
{
   clock_t cl;
   double ftime;

   f_CPU_finish = clock();
   /* printf("\n cl = %d, Divider = %d \n",cl,CLOCKS_PER_SEC);*/
   /*time = (double)cl;*/

   ftime = (f_CPU_finish - f_CPU_start);

   if (ftime < 0.) {
      ftime += 2147483647;
   }

   f_CPU_start = clock();
   f_CPU += ftime/CLOCKS_PER_SEC;

   /* printf("\n time = %g \n",time);*/

   time( &t_WallClock );

   f_WallClockElapsed = difftime( t_WallClock, t_WallClockStart );
   if (f_Phase > 0.) {
      f_WallClockETA = f_WallClockElapsed/f_Phase*f_PhaseStop;
   } else {
      f_WallClockETA = 0.;
   }

   return f_CPU;
}

//--- Controls:: ----------------------->
void Controls::Step(void) {
   l_Nstep++;
}

//--- Controls:: ----------------------->
int Controls::PhaseSave(void) {
   if (f_Phase > f_SaveTime) {
      i_Nwrite++;
      f_SaveTime += f_SavePeriod;
      domain()->out_Flog << "Controls::PhaseSave i_Nwrite="<<i_Nwrite<<endl;
      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->

int Controls::PhaseMovie(void) {
   printf("i_Movie %d f_Phase %e f_MovieTime %e f_MoviePeriod %e\n",i_Movie,f_Phase,f_MovieTime,f_MoviePeriod); 
   if (i_Movie == 0) return 0;
   if (f_Phase > f_MovieTime) {
      f_MovieTime += f_MoviePeriod;
      domain()->out_Flog << "Controls::PhaseMovie"<<endl;
      return -1;
   }
   return 0;
}
//--- Controls:: ----------------------->

int Controls::PhaseFieldFilter(void) {
   if (i_FieldFilterFlag == 0) return 0;
   if (f_Phase > f_FieldFilterTime) {
      f_FieldFilterTime += f_FieldFilterPeriod;
      domain()->out_Flog << "Controls::PhaseFieldFilter"<<endl;
      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->

int Controls::PhaseWakeControl(void) {
   if (i_WakeControlFlag == 0) return 0;
   if (f_Phase > f_WakeControlTime) {
      f_WakeControlTime += f_WakeControlPeriod;
      domain()->out_Flog << "Controls::PhaseWakeControl"<<endl;
      double dChange = domain()->GetMesh()->WakeControl();

#ifdef V_MPI
      int root = 0;
      int ierr = MPI_Bcast(&dChange, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
#endif
      dChange = 0.;
      domain()->GetMesh()->AddWakeCorrection(dChange);

      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->

int Controls::PhaseMovieH5(void) {
   if (i_MovieH5 == 0) return 0;
   if (f_Phase > f_MovieTimeH5) {
      f_MovieTimeH5 += f_MoviePeriodH5;
      i_MovieFrameH5++;
      domain()->out_Flog << "Controls::PhaseMovieH5"<<endl;
      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->
int Controls::PhaseStop(void) {
   if (f_Phase > f_PhaseStop) {
      domain()->out_Flog << "Controls::PhaseStop f_Phase="<<f_Phase<<endl;
      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->
int Controls::CPUStop(void) {
   if (f_CPU > f_CPUstop && f_CPUstop > 0. ) {
      domain()->out_Flog << "Controls::CPUStop f_CPU="<<f_CPU<<endl;
      return -1;
   }
   return 0;
}

//--- Controls:: ----------------------->
long Controls::Shift(void) {

  if (i_Shift && f_Phase > f_ShiftTime && f_Phase < f_PhaseStop && f_Phase < f_LastShiftTime) {
      f_ShiftTime += f_ShiftPeriod;
      l_Nshifted++;
      return l_Nshifted;
   }
   return 0;
}
