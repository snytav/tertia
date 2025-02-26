#ifndef H_BUFFERS
#define H_BUFFERS

#include <iostream>
#include <fstream>

#ifdef hz
#undef hz
#endif

#include <memory>
#include <memory.h>
#include <stdexcept>
#include "defines.h"
#include "vcomplex.h"
#include <stdlib.h>

using namespace std;

class CBufferError
{
   const string str_what;
public:
   CBufferError(const string str) : str_what(str){}
   virtual string what() const { return str_what; }
};

//---------------------------- CBuffer class -----------------------
class CBuffer
{
protected:
   char *buf;
   long b_len;
   long b_pos;
   long b_out;
   long b_CRC;

public:
   long GetCRC() { return b_CRC;}
   long FindCRC(long upto=0);
   virtual char *getbuf(void) {return buf;};
   virtual long getlen(void) const {return b_len;};
   virtual long getpos(void) const {return b_pos;};
   virtual long getout(void) const {return b_out;};
   virtual void setpos(int value) {b_pos = value;};
   virtual long fcnd(void) {return 0;};
   virtual long pcnd(void) {return 0;};
   virtual long kill(void) {return 0;};
   virtual long Adjust(long size);
   virtual void dump(void);
   virtual void push(char rc)   {npush( &rc, sizeof(char), 1);};
   virtual void push(int ri)    {npush( &ri, sizeof(int), 1);};
   virtual void push(long rl)   {npush( &rl, sizeof(long), 1);};
   virtual void push(float rf)  {npush( &rf, sizeof(float), 1);};
   virtual void push(double rd) {npush( &rd, sizeof(double), 1);};
   virtual void push(VComplex cc) {npush( &cc.re, sizeof(double), 2);};

   virtual void npush(void *ptr, long size, long nitems=1); //throw(bad_alloc);
   virtual void npop (void *ptr, long size, long nitems=1); //throw(CBufferError);

   virtual void pop(char &rc)   {npop(&rc, sizeof(char), 1);};
   virtual void pop(int &ri)    {npop(&ri, sizeof(int), 1);};
   virtual void pop(long &rl)   {npop(&rl, sizeof(long), 1);};
   virtual void pop(float &rf)  {npop(&rf, sizeof(float), 1);};
   virtual void pop(double &rd) {npop(&rd, sizeof(double), 1);};
   virtual void pop(VComplex &cc) {npop( &cc.re, sizeof(double), 2);};
   virtual void print_state(void);
   virtual void reset(void) {b_pos = b_out = 0;};
   virtual void pack(int what) { return ;};
   virtual void unpack(int what) { return ;};
   virtual void receive(CBuffer* rbuf)  { return ;};
   virtual void copy(CBuffer* rbuf);
   virtual CBuffer* operator<<(CBuffer* rbuf)
   {
      copy(rbuf); return this;
   };
   virtual void send(void) { return ;};

   virtual CBuffer& operator<<(char rc)
   {
      npush( &rc, sizeof(char), 1);
      return *this;
   };
   virtual CBuffer& operator<<(int ri)
   {
      npush( &ri, sizeof(int), 1);
      return *this;
   };
   virtual CBuffer& operator<<(long rl)
   {
      npush( &rl, sizeof(long), 1);
      return *this;
   };
   virtual CBuffer& operator<<(float rf)
   {
      npush( &rf, sizeof(float), 1);
      return *this;
   };
   virtual CBuffer& operator<<(double rd)
   {
      npush( &rd, sizeof(double), 1);
      return *this;
   };
   virtual CBuffer& operator<<(VComplex cc)
   {
      npush( &cc.re, sizeof(double), 2);
      return *this;
   };
   virtual CBuffer& operator>>(char &rc)
   {
      npop( &rc, sizeof(char), 1);
      return *this;
   };
   virtual CBuffer& operator>>(int &ri)
   {
      npop( &ri, sizeof(int), 1);
      return *this;
   };
   virtual CBuffer& operator>>(long &rl)
   {
      npop( &rl, sizeof(long), 1);
      return *this;
   };
   virtual CBuffer& operator>>(float &rf)
   {
      npop( &rf, sizeof(float), 1);
      return *this;
   };
   virtual CBuffer& operator>>(double &rd)
   {
      npop( &rd, sizeof(double), 1);
      return *this;
   };
   virtual CBuffer& operator>>(VComplex cc)
   {
      npop( &cc.re, sizeof(double), 2);
      return *this;
   };
   // For compatibility with Junction
   virtual long GetFcnd(void) { return 0;};
   virtual long GetPcnd(void) { return 0;};
   virtual long Kill(void) { return 0;};
   //

   CBuffer(long size=BUF_SIZE);
   virtual ~CBuffer()
   {
      b_len = 0;
      b_pos = 0;
      b_out = 0;
      b_CRC = 0;
      delete[] buf;
      buf = NULL;
   }
};

//---------------------------- CBuffer class -----------------------
class FCBuffer : public CBuffer
{
protected:
   fstream file;
   char *filename;
public:
   virtual long fcnd(void) {return 0;};
   virtual long pcnd(void) {return 0;};
   virtual long kill(void) {return 0;};
   virtual void dump(void) { return;};

   virtual void reset(void);
   virtual void print_state(void);
   FCBuffer(char *fname);
   virtual ~FCBuffer() { file.close(); };
};

#endif
