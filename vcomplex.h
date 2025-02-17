#ifndef H_VCOMPLEX
#define H_VCOMPLEX

#include <stdlib.h>
#include <math.h>

class VComplex {
 public:
  double re;
  double im;

  inline VComplex operator+(VComplex s) {
    VComplex tmp(re,im); 
    tmp.re+=s.re; 
    tmp.im+=s.im;
    return tmp;
  };

  inline VComplex operator+(double s) {
    VComplex tmp(re,im); 
    tmp.re+=s; 
    return tmp;
  };

  inline VComplex operator-(VComplex s) {
    VComplex tmp(re,im); 
    tmp.re-=s.re; 
    tmp.im-=s.im; 
    return tmp;
  };

  inline VComplex operator-(double s) {
    VComplex tmp(re,im); 
    tmp.re-=s; 
    return tmp;
  };

  VComplex operator-(void) {VComplex tmp(-re,-im); return tmp;};

  inline VComplex operator*(VComplex s) {
    VComplex tmp(re,im); 
    tmp.im = re*s.im + im*s.re;
    tmp.re = re*s.re - im*s.im;
    return tmp;
  };

  inline VComplex operator*(double s) {
    VComplex tmp(re,im); 
    tmp.im *= s;
    tmp.re *= s;
    return tmp;
  };

  inline double abs(void) {
    return sqrt(re*re + im*im);
  }

  inline double abs2(void) {
    return re*re + im*im;
  }

  inline VComplex operator/(VComplex s) {
    VComplex tmp(re,im);
    double a = s.abs2();
    tmp.im = (-re*s.im + im*s.re)/a;
    tmp.re = (re*s.re + im*s.im)/a;
    return tmp;
  };

  inline VComplex operator/(double s) {
    VComplex tmp(re,im); 
    tmp.im /= s;
    tmp.re /= s;
    return tmp;
  };

  inline VComplex mult(VComplex s) {
    VComplex tmp(re,im); 
    tmp.im = re*s.im + im*s.re;
    tmp.re = re*s.re - im*s.im;
    return tmp;
  };

  double imag(){return im;}
  double real(){return re;}

  VComplex(double r=0., double i=0.) {re=r; im=i;};
};

//double abs(VComplex t) {return sqrt(t.re*t.re + t.im*t.im);};

typedef VComplex Complex;

inline double abs(VComplex c){return sqrt(c.im*c.im+c.re*c.re);};

inline VComplex operator*(double s, VComplex vc) {
    VComplex tmp = vc; 
    tmp.im *= s;
    tmp.re *= s;
    return tmp;
  };

inline VComplex operator+(double s, VComplex vc) {
    VComplex tmp = vc; 
    tmp.re += s;
    return tmp;
  };


inline VComplex operator/(double s, VComplex vc) {
   VComplex tmp = vc;
   double a = vc.abs2();
   tmp.im = -s*vc.im/a;
   tmp.re =  s*vc.re/a;
   return tmp;
};

/*

#include <complex>

using namespace std;
typedef complex<double> Complex;

*/

#endif


