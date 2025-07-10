obj =  mesh.o movepart.o movefields.o movieframe.o \
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
		half_integer1D.o labels.o control_point.o beam_control_point.o



cuobj = CUDA_WRAP/transpose.o CUDA_WRAP/turn.o CUDA_WRAP/1d_batch.o \
        CUDA_WRAP/phase.o \
                CUDA_WRAP/mult.o CUDA_WRAP/diagnostic_print.o \
                CUDA_WRAP/cuda_wrap_vector_list.o \
                CUDA_WRAP/variables.o CUDA_WRAP/linearized.o \
                CUDA_WRAP/cuda_grid.o CUDA_WRAP/cuda_wrap_control.o \
                CUDA_WRAP/normalization.o  CUDA_WRAP/surfaceFFT.o \
                CUDA_WRAP/beam_copy.o CUDA_WRAP/profile.o CUDA_WRAP/cuBeam.o \
                CUDA_WRAP/cuParticles.o CUDA_WRAP/cuBeamValues.o CUDA_WRAP/copy_hydro.o \
                CUDA_WRAP/plasma_particles.o CUDA_WRAP/cuLayers.o CUDA_WRAP/paraLayers.o \
                CUDA_WRAP/paraCPUlayers.o CUDA_WRAP/half_integer2D.o CUDA_WRAP/copy_layer.o

HDF5=/usr/include/hdf5/
HDF5_INCLUDE = -I$(HDF5)/serial/

MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi/
MPI_INCLUDE = -I$(MPI_DIR)/include


CUDAFLAGS = -dc -G -g -O0
CUDA_INC = -I/usr/local/cuda/include  $(HDF5_INCLUDE) \
            $(MPI_INCLUDE)

CUDA_LIB =  -lcudart -lcufft -L/usr/local/cuda/lib64

CC = g++ -std=c++03 -O0
MPI_INCLUDE = -I$(MPI_DIR)/include
GSL_INCLUDE = -I$(GSL_DIR)/include/gsl
CFLAGS = -pg -g $(OO) $(GG) $(ICC_FLAGS) -c $(MPI_INCLUDE) $(GSL_INCLUDE) $(HDF5_INCLUDE)  \
	 $(KTRACE_INCLUDE) $(FS) $(DEFINES) \
         $(MPI) $(NO-DEPRECATED) $(WARNING) \
         $(CUDA_INC)

HDF5_LIB_DIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
HDF5_LIB = -lhdf5 -lz

MPI_LIB = -lmpi  -lmpi_cxx -pg -lcublas -lcudart 
             

FFTW=/usr/lib64/
FFTW_LIB_DIR = -L/usr/lib64/
FFTW_LIB = -lfftw3 -lfftw3_threads


all:   $(cuobj) $(obj)
	nvcc -rdc=true *.o  -o test $(HDF5_LIB_DIR) $(HDF5_LIB) $(MPI_LIB) \
	                            $(FFTW_LIB) $(FFTW_LIB_DIR) $(CUDA_LIB)
%.o:    %.cu
	@echo $<
	nvcc $(CUDA_INC) $(CUDAFLAGS) $<
	cp *.o ..

%.o:    %.cpp
	@echo $<
	$(CC) $(CFLAGS) $<

clean:
	rm *.o test CUDA_WRAP/*.o
