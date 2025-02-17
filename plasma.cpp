#include <math.h>
#include "vlpl3d.h"

//---Plasma:: ----------------------->

double Specie::Density (double xco, double yco, double zco)
{
   // double Specie::Density (double xco, double yco, double zco) returns
   // density of a Specie at position (xco,yco,zco)
   // the returned value is the local density normalized on the critical density
   // corresponding to the fundamental laser wavelength.
   // The characteristic Specie density "Density" read from the .ini file 
   // is stored in the variable f_Narb
   // The coordinates are measured in the fundamental laser wavelengths:
   // xco=0 at the left X boundary
   // yco=0 and zco=0 at the optical axis in the middle of the simulation box
   //
  double modulation = 1.;
  double Ry = 1.;
  double Rz = 1.;
  double tmp = 0.;
  double density = f_Narb;
  double density1 = f_Narb1;
  double arg = 0.;
  double r2 = 0.;
  double x0, y0, z0;
  double xtmp = xco;
  double arg_front = 0.;
  double front = 1.;
  switch (i_Distribution) {
	  case 12:	// Periodic
		  if (xco<f_Begin || xco>f_End) return tmp;
		  if (xco<f_PlateauBegin) {
			  //tmp = density*(xco - f_Begin)/(f_PlateauBegin - f_Begin);
			  if (f_Scale <= 0) f_Scale = 1;
			  tmp = density*exp((xco - f_PlateauBegin)/f_Scale);
			  return tmp;
		  }
		  if (xco>f_PlateauEnd) {
			  tmp = density*(f_End - xco)/(f_End - f_PlateauEnd);
			  return tmp;
		  }
		  density1 = f_Narb1;
        if (density1 < 0) density1 = 0.;
		  xtmp = xco;
		  while (xtmp > f_Xperiod) {
			  xtmp -= f_Xperiod;
		  }
		  while (xtmp < 0.) {
			  xtmp += f_Xperiod;
		  }
        xco = xtmp;
		  if (xtmp>f_Xperiod - f_Delta) {
			  tmp = density1;
		  } else {
			  tmp = density;
		  }
		  return tmp;
		  break;
  case 11:     //  gradient
    if (xco < 0.) return 0.;
    if (xco<f_Begin || xco>f_End) return 0.;
    if (xco<f_PlateauBegin) {
      return density*(xco - f_Begin)/(f_PlateauBegin - f_Begin);
    }
    if (xco<f_PlateauEnd) {
      return density*exp((xco - f_PlateauBegin)/f_Delta);
    }
    if (xco>f_PlateauEnd) {
      return density*(f_End - xco)/(f_End - f_PlateauEnd);
    }
    return 0.;
    break;
  case 10:     // profile with linear step
    if (xco < 0.) return 0.;
    if (xco<f_Begin || xco>f_End) return tmp;
    if (xco<f_PlateauBegin) {
      return f_Narb;
    }
    if (xco<f_PlateauEnd) {
      tmp = density + 
	density*f_Delta*(xco - f_PlateauBegin)/(f_PlateauEnd - f_PlateauBegin);
      return tmp;
    }
    if (xco>f_PlateauEnd) {
      tmp = density*(1. + f_Delta);
      return tmp;
    }
    return 0.;
    break;
   case 9:     // round rod and gophred rod

    if (xco<f_Begin || xco>f_End) return 0.;

    yco = yco + f_AngleY*(xco-f_Begin);

    if (f_Yperiod > 0.) {
      while (yco > f_Yperiod/2.) {
	yco -= f_Yperiod;
      }
      while (yco <= -f_Yperiod/2.) {
	yco += f_Yperiod;
      };
    };

    if (f_Zperiod > 0.) {
      while (zco > f_Zperiod/2.) {
	zco -= f_Zperiod;
      }
      while (zco <= -f_Zperiod/2.) {
	zco += f_Zperiod;
      };
    };

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }

    if (f_RadiusX <= 0.) {
      f_RadiusX = f_Radius;
    }
    if (f_RadiusY == 0.) {
      f_RadiusY = f_Radius;
    }
    if (f_RadiusZ == 0.) {
      f_RadiusZ = f_Radius;
    }
    arg = (yco-f_y0)*(yco-f_y0)/f_RadiusY/f_RadiusY
      + (zco-f_z0)*(zco-f_z0)/f_RadiusZ/f_RadiusZ;

    if (f_Xperiod > 0.) {
      arg *= 1 + f_Delta*sin(2.*PI*xco/f_Xperiod);
    }

    if ((f_RadiusY>0 || f_RadiusZ>0) && arg > 1) {
      return 0.;
    }
    if ((f_RadiusY<0 || f_RadiusZ<0) && arg < 1) {
      return 0.;
    }
    return f_Narb;
    break;
  case 8:     // Kumar's tanh-step beam
    tmp = 0.;
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;

    r2 = yco*yco + zco*zco;

    if (r2 > f_Radius*f_Radius) {
      return tmp;
    }

    if (f_RadiusX <= 0.) {
      f_RadiusX = 1.;
    }

    density = 0.5*(1.-tanh((xco-f_x0)/f_RadiusX));

    return f_Narb*density;
    break;
  case 7:     // COS-GAUSS beam
    tmp = 0.;
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;

    front = (f_End-f_x0);
    if (front<f_RadiusX) {
       front = 1;
    } else {
       front = f_RadiusX*100.;
    };

    arg_front = 0.;
    if (xco>f_x0) arg_front = (xco-f_x0)/front;
    arg_front *= arg_front/2.;

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }

    if (f_RadiusX <= 0.) {
      f_RadiusX = f_Radius;
    }
    if (f_RadiusY <= 0.) {
      f_RadiusY = f_Radius;
    }
    if (f_RadiusZ <= 0.) {
      f_RadiusZ = f_Radius;
    }

    Ry = f_RadiusY;
    Rz = f_RadiusZ;

    arg = (xco-f_x0)/f_RadiusX;
    if (fabs(arg) > sqrt(2.*PI)) {
      return tmp;
    };
    density = 0.5*(1 + cos(sqrt(PI/2.)*arg));

    if (f_Xperiod != 0.) {
       modulation = 1.+f_Delta*cos(2*PI*xco/f_Xperiod);
    } else {
       modulation = 1.;
    }
    if (modulation <= 0.) modulation = 1.;

    Ry = Ry*modulation;
    Rz = Rz*modulation;
    density = density/(modulation*modulation);

    arg = 0.5*(yco-f_y0)*(yco-f_y0)/Ry/Ry 
      + 0.5*(zco-f_z0)*(zco-f_z0)/Rz/Rz;

    if (arg > 4) {
      return tmp;
    };
    density *= exp(-arg-arg_front);

    if (xco>f_PlateauEnd && xco < f_End) {
       double xmid = (f_End + f_PlateauEnd)/2.;
       double width = (f_End - f_PlateauEnd);
       density *= 0.5*(1.-tanh(4.*(xco - xmid)/width)); 
    }

    return f_Narb*density;
    break;
  case 6:     // Empty channel
    tmp = 0.;
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;

    r2 = (yco-f_y0)*(yco-f_y0) + (zco-f_z0)*(zco-f_z0);

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }

    if (r2 < f_Radius*f_Radius) {
      density *= f_Delta;
    }

    if (xco<f_PlateauBegin) {
      density *= (xco - f_Begin)/(f_PlateauBegin - f_Begin);
      return density;
    }
    if (xco>f_PlateauEnd) {
      density *= (f_End - xco)/(f_End - f_PlateauEnd);
      return density;
    }
      
    return density;
    break;
  case 5:     // Exponential profile
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;
    if (xco<f_PlateauBegin) {
      tmp = 0.5*density*(1.+tanh(5.*(xco - f_Begin)/(f_PlateauBegin - f_Begin)));
      return tmp;
    }
    if (xco>f_PlateauEnd) {
      tmp = density*(1. + f_Delta)*(f_End - xco)/(f_End - f_PlateauEnd);
      return tmp;
    }
    return f_Narb*(1. + f_Delta*(xco - f_PlateauBegin)
		   /(f_PlateauEnd - f_PlateauBegin));
    break;
  case 4:     // parallelepiped
    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }

    if (f_RadiusX <= 0.) {
      f_RadiusX = f_Radius;
    }
    if (f_RadiusY <= 0.) {
      f_RadiusY = f_Radius;
    }
    if (f_RadiusZ <= 0.) {
      f_RadiusZ = f_Radius;
    }
    if (fabs(xco-f_x0) > f_RadiusX) return 0.;
    if (fabs(yco-f_y0) > f_RadiusY) return 0.;
    if (fabs(zco-f_z0) > f_RadiusZ) return 0.;
    return density;
    break;
  case 3:     // curved Gaussian channel
    tmp = 0.;
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;

    y0 = f_y0;
    z0 = f_z0;
    if (f_Xperiod > 0. && xco > f_CurvatureBegin && xco < f_CurvatureEnd) {
      y0 = f_y0 + f_Ycurvature*sin(2.*PI*xco/f_Xperiod);
      z0 = f_z0 + f_Zcurvature*cos(2.*PI*xco/f_Xperiod);
    }

    r2 = (yco-y0)*(yco-y0) + (zco-z0)*(zco-z0);

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }
    arg = r2/f_Radius/f_Radius;
    if (arg < 2) {
      tmp = f_Delta*exp(-arg);
    }
    if (tmp < 0) tmp = 0.;
    if (tmp > 1) tmp = 1.;
    density *= (1. - tmp); 
    if (xco<f_PlateauBegin) {
      density *= (xco - f_Begin)/(f_PlateauBegin - f_Begin);
      return density;
    }
    if (xco>f_PlateauEnd) {
      density *= (f_End - xco)/(f_End - f_PlateauEnd);
      return density;
    }
      
    return density;
    break;
  case 2:     // Gaussian channel
    tmp = 0.;
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;

    r2 = (yco-f_y0)*(yco-f_y0) + (zco-f_z0)*(zco-f_z0);

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }
    arg = r2/f_Radius/f_Radius;
    if (arg < 10) {
      tmp = f_Delta*exp(-arg);
    }
    if (tmp < 0) tmp = 0.;
    if (tmp > 1) tmp = 1.;
    if (xco<f_PlateauBegin) {
      return density*(xco - f_Begin)/(f_PlateauBegin - f_Begin)*(1.-tmp);
    }
    if (xco>f_PlateauEnd) {
      return density1*(f_End - xco)/(f_End - f_PlateauEnd)*(1.-tmp);
    }
    density = (1. - tmp)
      *(f_Narb*(f_PlateauEnd - xco) + f_Narb1*(xco - f_PlateauBegin))
	/(f_PlateauEnd - f_PlateauBegin) ; 
      
    return density;
    break;
  case 1:     // Gaussian ellipsoid

    if (xco<f_Begin || xco>f_End) return 0.;

    if (f_Radius <= 0.) {
      f_Radius = 1.;
    }

    if (f_RadiusX <= 0.) {
      f_RadiusX = f_Radius;
    }
    if (f_RadiusY <= 0.) {
      f_RadiusY = f_Radius;
    }
    if (f_RadiusZ <= 0.) {
      f_RadiusZ = f_Radius;
    }
    arg = (xco-f_x0)*(xco-f_x0)/f_RadiusX/f_RadiusX
      + (yco-f_y0)*(yco-f_y0)/f_RadiusY/f_RadiusY
      + (zco-f_z0)*(zco-f_z0)/f_RadiusZ/f_RadiusZ;

    if (arg > 16) {
      return tmp;
    }
    if (f_Xperiod <= 0.) {
      f_Xperiod = 1.;
    };
    return f_Narb*exp(-arg)*(1.+f_Delta*cos(2*PI*(xco-f_x0)/f_Xperiod));
    return 0.;
    break;
  default:
  case 0:     // Default trapezoidal profile
    if (xco < 0.) return tmp;
    if (xco<f_Begin || xco>f_End) return tmp;
    if (xco<f_PlateauBegin) {
      tmp = density*(xco - f_Begin)/(f_PlateauBegin - f_Begin);
      return tmp;
    }
    if (xco>f_PlateauEnd) {
      tmp = density*(f_End - xco)/(f_End - f_PlateauEnd);
      return tmp;
    }
    return f_Narb;
    break;
  }

}

/*
//---Plasma:: ----------------------->

double Specie::Density (double xco, double yco, double zco)
{
double tmp = 0.;
double density = f_Narb;
if (xco < 0.) return tmp;
//  if (xco >= domain()->GetXlength()) return tmp;
if (xco<f_Begin || xco>f_End) return tmp;
//      if (yco<plateau_begin || yco>plateau_end) return tmp;
//      if (zco<plateau_begin || zco>plateau_end) return tmp;
if (xco<f_PlateauBegin) {
tmp = density*(xco - f_Begin)/(f_PlateauBegin - f_Begin);
return tmp;
}
if (xco>f_PlateauEnd) {
tmp = density*(f_End - xco)/(f_End - f_PlateauEnd);
return tmp;
}
return f_Narb;
}
*/
