#ifndef CUDA_WRAP_SEND_BEAM_H
#define CUDA_WRAP_SEND_BEAM_H


int CUDA_WRAP_getFlyList(int *Np,double x_min,double x_max,beamParticle *fly_list_min,int *size_fly_list_min,
			                                                              beamParticle *fly_list_max,int *size_fly_list_max);
int addBeamParticles(int *Np,beamParticle *d_fly_list,int size_fly_list);

int copyParticlesToDevice(beamParticle *d_p,beamParticle *h_p,int Np);

int copyParticlesToHost(beamParticle *h_p,beamParticle *d_p,int Np);

int CUDA_WRAP_allocFly(int Np,beamParticle **d_send_down,beamParticle **d_send_up,beamParticle **d_recv_down,beamParticle **d_recv_up);

#endif