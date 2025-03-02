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
                CUDA_WRAP/paraCPUlayers.o

HDF5=/usr/include/hdf5/
HDF5_INCLUDE = -I$(HDF5)/serial/

#MPI_DIR=/opt/mvapich2/
MPI_INCLUDE = -I$(MPI_DIR)/include


CUDAFLAGS = -dc  -g
CUDA_INC = -I/usr/local/cuda/include  $(HDF5_INCLUDE) \
            $(MPI_INCLUDE)


test:   $(cuobj)
	nvcc -rdc=true *.o -o test
%.o:    %.cu
	@echo $<
	nvcc $(CUDA_INC) $(CUDAFLAGS) $<

clean:
	rm *.o test CUDA_WRAP/*.o
