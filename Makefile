                       #
##########################################################################################################

#========================================== Program name ================================================#
PN = v3d.e
#========================================================================================================#

#===================================== Postprocessor program name =======================================#
PPN = v3dpp.e
#========================================================================================================#

#=============================== Synchrotron Postprocessor program name =================================#
PPSYNCHN = vlpl3dppsynch.e
#========================================================================================================#

#=========================================== Define V_MPI ===============================================#
#DEF_V_MPI=-DV_MPI
#========================================================================================================#

#======================================== Define SYNCHROTRON ============================================#
DEF_SYNCHROTRON=-DSYNCHROTRON
#========================================================================================================#


#========================================= mpich directory ==============================================#
MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi/
#MPI_DIR=/opt/mpich2/
#MPI_DIR=/opt/mvapich2/
#========================================================================================================#

#============================== Message Passing Interface include files =================================#
#MPI_INCLUDE = -I/.netmount/storage_giga/$(MPICH)/ch_p4/include/
#MPI_INCLUDE = -I/.netmount/storage_giga/$(MPICH)/ch_p4mpd/include/
MPI_INCLUDE = -I$(MPI_DIR)/include
#========================================================================================================#

#========================================== MPI lib files ===============================================#
#MPI_LIB_DIR = -L/.netmount/storage_giga/$(MPICH)/ch_p4/lib/
#MPI_LIB_DIR = -L/.netmount/storage_giga/$(MPICH)/ch_p4mpd/lib/
MPI_LIB_DIR = -L$(MPI_DIR)/lib/ -L/opt/slurm/lib

#MPI_LIB = -lmpi
#MPI_LIB = -lmpich  -lmpichcxx -lrtkaio -lpmi
MPI_LIB = -lmpi  -lmpi_cxx -pg -gstabs -lcublas -lcudart \
             -L/opt/cuda4/lib64/ 

#========================================= HDF5 directory ==============================================#
HDF5=/usr/include/hdf5/
#========================================= FFTW directory ==============================================#
FFTW=/usr/lib64/
#============================== Message Passing Interface include files =================================#
HDF5_INCLUDE = -I$(HDF5)/serial/
#========================================================================================================#

#========================================== HDF lib files ===============================================#
HDF5_LIB_DIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
HDF5_LIB = -lhdf5 -lz
#========================================================================================================#
#========================================== FFTW lib files ===============================================#
FFTW_LIB_DIR = -L/usr/lib64/
FFTW_LIB = -lfftw3 -lfftw3_threads
#========================================================================================================#

#========================================== CUDA ========================================================#
CUDACC = /usr/local/cuda-12.8/bin/nvcc
CUDAFLAGS = -c  -g 
CUDA_INC = -I/usr/local/cuda/include  $(HDF5_INCLUDE) \
            $(MPI_INCLUDE)
CUDA_LIB =  -lcudart -lcufft -L/usr/local/cuda/lib64  

#======================================== PulseProfiles lib =============================================#
PULSE_PROFILE_LIB_DIR = PulseProfile
PULSE_PROFILE_LIB = -lPulseProfiles
#========================================================================================================#

#======================================== PulseProfiles lib =============================================#
SCALES_LIB_DIR = Scales
SCALES_LIB = -lScales
#========================================================================================================#

#=============================== GNU Scientific library ============= ===================================#
GSL_DIR = /usr/
#=============================== GNU Scientific library include files ===================================#
#GSL_INCLUDE = -I/.netmount/storage_vlpl/usr/include/gsl/
#GSL_INCLUDE = -I/.netmount/storage_giga/usr/include/gsl/ -I/.netmount/storage_giga/usr/include/
GSL_INCLUDE = -I$(GSL_DIR)/include/gsl
#========================================================================================================#

#================================= GNU Scientific library lib files =====================================#
#GSL_LIB_DIR = -L/.netmount/storage_vlpl/usr/lib/
#GSL_LIB_DIR = -L/.netmount/storage_giga/usr/lib/
GSL_LIB_DIR = -L$(GSL_DIR)/lib64
GSL_LIB = -lgsl -lgslcblas
#========================================================================================================#

#======================================= KTrace include files ===========================================#
#KTRACE_INCLUDE = -I/usr/include/kde/
#========================================================================================================#

#======================================== OLDVERSION Define =============================================#
# It is define for compatibility with old version of saving files.
#DEFINE_OLDVERSION=-DOLDVERSION
#========================================================================================================#

##########################################################################################################
#============================================== Make ====================================================#
M_MAKE = $(MAKE)
#M_MAKE = $(MAKE)
#========================================================================================================#

#============================================ Compiler ==================================================#
#CC = xlC
#mpCCFlags = -qrtti
CC = g++ -std=c++03
#CC = icpc # for debug version with Intel compiler
#CC = icc # Intel compiler
#FS = -ffloat-store
#NO-DEPRECATED = -Wno-deprecated
#WARNING = -W -Wall
#========================================================================================================#

#====================================== Intel Compiler flags ============================================#
# If you use Intel compiler please specify additional flags
ICC_FLAGS=-static-libgcc $(WARNING) 
GCC_FLAGS= 
#========================================================================================================#

#========================================= Compiler flags ===============================================#
Optimized = -O0 
Debug = -O0 
CFLAGS = -pg -g  $(mpCCFlags) $(OO) $(GG) $(ICC_FLAGS) -c $(MPI_INCLUDE) $(GSL_INCLUDE) $(HDF5_INCLUDE)  \
	 $(KTRACE_INCLUDE) $(FS) $(DEFINES) \
         $(MPI) $(NO-DEPRECATED) $(WARNING) \
         $(CUDA_INC)
STATIC = -static
#========================================================================================================#

#=========================================== Link flags =================================================#
LINK = $(CC) $(ICC_FLAGS) -o # if you use icc then specify "ICC_FLAGS"
LINK_FLAGS = $(LF)
#========================================================================================================#

#======================================= vlpl3d project libs ============================================#
PROJECT_LIB_DIR = lib
PROJECT_LIB_DIR_FLAG = -Llib ${MPI_LIB_DIR} ${HDF5_LIB_DIR} ${GSL_LIB_DIR} $(CUDA_LIB)
PROJECT_LIB_DIR_FLAGS =${PULSE_PROFILE_LIB_DIR} ${SCALES_LIB_DIR} ${MPI_LIB_DIR} ${HDF5_LIB_DIR} ${GSL_LIB_DIR}
#PROJECT_LIBS_FLAGS =${PULSE_PROFILE_LIB} ${SCALES_LIB} ${GSL_LIB} ${HDF5_LIB}
PROJECT_LIBS_FLAGS =${HDF5_LIB} ${FFTW_LIB}
#========================================================================================================#

#========================================== lib utilities ===============================================#
AR_LIB = /usr/bin/ar
AR_FLAGS = rv
RAN_LIB = /usr/bin/ranlib
#========================================================================================================#

#===================================== vlpl3d project libs define =======================================#
P_LIBS = vlpl3d_lib
#========================================================================================================#

#======================================== vlpl3d project files ==========================================#
objects =  mesh.o movepart.o movefields.o movieframe.o \
		movebeam.o movefieldshydro.o movefieldshydrolinlayer.o \
		movefieldshydrolinlayersplit.o \
		movepart_layer.o movepart_splitlayer.o \
		boundary.o cells.o cell3d.o Cgs.o controls.o \
		diagnose.o domain.o adk.o \
		elements.o exchange.o save.o load.o save_movie_frame.o \
		mkdirLinux.o myhdfshell.o \
		particles.o partition.o plasma.o pulse.o \
		step.o synchrotron.o \
		vlpl3d.o \
		namelist.o buffers.o  para.o\
		half_integer1D.o  CUDA_WRAP/half_integer2D.o \
		CUDA_WRAP/transpose.o CUDA_WRAP/turn.o CUDA_WRAP/1d_batch.o CUDA_WRAP/phase.o \
		CUDA_WRAP/mult.o CUDA_WRAP/diagnostic_print.o \
		CUDA_WRAP/cuda_wrap_vector_list.o \
		CUDA_WRAP/variables.o CUDA_WRAP/linearized.o \
		CUDA_WRAP/cuda_grid.o CUDA_WRAP/cuda_wrap_control.o \
		CUDA_WRAP/normalization.o  CUDA_WRAP/surfaceFFT.o \
		CUDA_WRAP/beam_copy.o CUDA_WRAP/profile.o CUDA_WRAP/cuBeam.o \
		CUDA_WRAP/cuParticles.o CUDA_WRAP/cuBeamValues.o CUDA_WRAP/copy_hydro.o \
		CUDA_WRAP/plasma_particles.o CUDA_WRAP/cuLayers.o CUDA_WRAP/paraLayers.o \
		CUDA_WRAP/paraCPUlayers.o CUDA_WRAP/surf.o
		
		
#========================================================================================================#

#====================================== vlpl3d project objects ==========================================#
objects_vlpl3d = $(objects)
objects_vlpl3d_link = $(objects)
#========================================================================================================#
##########################################################################################################

#========================================== vlpl3d project ==============================================#
#vl : vlpl3d strip

all :
	$(M_MAKE) OO=$(Optimized) LF="${PROJECT_LIBS_FLAGS} ${MPI_LIB} ${PROJECT_LIB_DIR_FLAG}" $(PN) \
   MPI=$(DEF_V_MPI) DEFINES="${DEFINE4ALL}"

$(PN) : $(objects_vlpl3d)
	${LINK} $(PN) $(objects_vlpl3d_link) $(LINK_FLAGS)
#========================================================================================================#

#====================================== vlpl3d project DEBUG ============================================#
vld : makevld

makevld :
	$(M_MAKE) OO=$(Debug) GG=-g LF="${PROJECT_LIBS_FLAGS} ${MPI_LIB} ${PROJECT_LIB_DIR_FLAG} ${STATIC}" \
   MPI=$(DEF_V_MPI) DEFINES="${DEFINE4ALL} -D_DEBUG" $(PN)
#========================================================================================================#

#=================================== vlpl3d project SINGLE VERSION ======================================#
single : _single strip

_single :
	$(M_MAKE) OO=$(Optimized) LF="${PROJECT_LIBS_FLAGS} ${PROJECT_LIB_DIR_FLAG} ${STATIC}" DEFINES="${DEFINE4ALL}" \
   $(PN)
#========================================================================================================#

#================================ vlpl3d project SINGLE VERSION DEBUG====================================#
singled : _singled

_singled :
	$(M_MAKE) OO=$(Debug) GG=-g LF="${PROJECT_LIBS_FLAGS} ${PROJECT_LIB_DIR_FLAG} ${STATIC}" \
   DEFINES="${DEFINE4ALL} -D_DEBUG" $(PN)
#========================================================================================================#

#=========================== remove debug information from release file =================================#           

#========================================================================================================#
   
#===================================== vlpl3d project CLEAN =============================================#           
allclean : exec_clean clean libsclean

exec_clean :
	-rm -f *.e

clear : clean

clean :     
	-rm -f *.o CUDA_WRAP/*.o v3d.e

libsclean : pulse_profile_libs_clean scales_libs_clean projects_libs_clean   
      
pulse_profile_libs_clean :
	$(MAKE) -C ${PULSE_PROFILE_LIB_DIR} clean   

scales_libs_clean :
	$(MAKE) -C ${SCALES_LIB_DIR} clean
      
projects_libs_clean :
	-rm -f $(PROJECT_LIB_DIR)/*.a

.PHONY : clean
#========================================================================================================#

#======================================= objects compile ================================================#              
.C.o	  :
		$(CC) $(CFLAGS) $<
		
%.o  : %.cu
		$(CUDACC) $(CUDA_INC) $(CUDAFLAGS) $<
		cp *.o CUDA_WRAP

.cpp.o	  :
		$(CC)  $(CFLAGS) $<

.c.o	  :
		$(CC) -Kc++ $(CFLAGS) $<
#========================================================================================================#      


