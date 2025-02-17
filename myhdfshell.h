#ifndef MYHDFSHELL
/* HDF5 Library                                                       */ 
#include <hdf5.h>
#include <assert.h>

#include <iostream>
#include <fstream>

#ifdef hz
#undef hz
#endif

int WriteHDF5Recorddouble(hid_t file, char* dataname, int nfields, double *source);
int WriteHDF5RecordInt(hid_t file, char* dataname, int nfields, int *source);
int WriteHDF5RecordLong(hid_t file, char* dataname, int nfields, long *source);
int WriteHDF5RecordString(hid_t file, char* dataname, int nfields, char *source);

int WriteHDF5Record(hid_t file, char* dataname, int nfields, double *source);
int WriteHDF5Record(hid_t file, char* dataname, int nfields, int *source);
int WriteHDF5Record(hid_t file, char* dataname, int nfields, long *source);
int WriteHDF5Record(hid_t file, char* dataname, int nfields, char *source);

int ReadHDF5Recorddouble(hid_t file, char* dataname, int nfields, double *source);
int ReadHDF5RecordInt(hid_t file, char* dataname, int nfields, int *source);
int ReadHDF5RecordLong(hid_t file, char* dataname, int nfields, long *source);
int ReadHDF5RecordString(hid_t file, char* dataname, int nfields, char *source);

int ReadHDF5Record(hid_t file, char* dataname, int nfields, double *source);
int ReadHDF5Record(hid_t file, char* dataname, int nfields, int *source);
int ReadHDF5Record(hid_t file, char* dataname, int nfields, long *source);
int ReadHDF5Record(hid_t file, char* dataname, int nfields, char *source);

#define MYHDFSHELL
#endif
