
#include "vlpl3d.h"
#include "cell3d.h"

//---Cell3Dm::--------------------------------------------->
void Cell3Dm::AddDensity(Particle *p, double polarity)
{
   double one = double(1.);
   double x, y, z;
   p->GetX(x,y,z);
   double abs_weight = p->GetWeight();
   double weight = abs_weight*p->GetZ()*polarity;
   double xmymzm = (one-x)*(one-y)*(one-z);
   double xymzm = x*(one-y)*(one-z);
   double xmyzm = (one-x)*y*(one-z);
   double xyzm = x*y*(one-z);
   double xmymz = (one-x)*(one-y)*z;
   double xymz = x*(one-y)*z;
   double xmyz = (one-x)*y*z;
   double xyz = x*y*z;

   XYZ->f_Dens += xyz*weight;
   XmYZ->f_Dens += xmyz*weight;
   XYmZ->f_Dens += xymz*weight;
   XmYmZ->f_Dens += xmymz*weight;
   XYZm->f_Dens += xyzm*weight;
   XmYZm->f_Dens += xmyzm*weight;
   XYmZm->f_Dens += xymzm*weight;
   XmYmZm->f_Dens += xmymzm*weight;

   int isort = p->GetSort();
   if (isort>=0 && isort < Domain::p_D->GetNsorts()) {
      weight = abs(weight);
      XYZ->f_DensArray[isort] += xyz*weight;
      XmYZ->f_DensArray[isort] += xmyz*weight;
      XYmZ->f_DensArray[isort] += xymz*weight;
      XmYmZ->f_DensArray[isort] += xmymz*weight;
      XYZm->f_DensArray[isort] += xyzm*weight;
      XmYZm->f_DensArray[isort] += xmyzm*weight;
      XYmZm->f_DensArray[isort] += xymzm*weight;
      XmYmZm->f_DensArray[isort] += xmymzm*weight;
   }
}

//---Cell3Dm::--------------------------------------------->
Cell3Dm::Cell3Dm(Mesh* mesh, long i, long j, long k)
{
   m = mesh;
   long n = m->GetN(i,j,k);
   XYZ = &(m->GetCell(n));
   XmYZ = &(m->GetCell(m->Xm(n)));
   XYmZ = &(m->GetCell(m->Ym(n)));
   XmYmZ = &(m->GetCell(m->Xm(m->Ym(n))));
   XYZm = &(m->GetCell(m->Zm(n)));
   XmYZm = &(m->GetCell(m->Xm(m->Zm(n))));
   XYmZm = &(m->GetCell(m->Ym(m->Zm(n))));
   XmYmZm = &(m->GetCell(m->Xm(m->Ym(m->Zm(n)))));
}

//---Cell3D::--------------------------------------------->
void Cell3D::AddDensity(Particle *p, double polarity)
{
   double one = double(1.);
   double x, y, z;
   p->GetX(x,y,z);
   double weight = p->GetWeight()*p->GetZ()*polarity;
   double abs_weight = p->GetWeight();
   double xyz = (one-x)*(one-y)*(one-z);
   double xpyz = x*(one-y)*(one-z);
   double xypz = (one-x)*y*(one-z);
   double xpypz = x*y*(one-z);
   double xyzp = (one-x)*(one-y)*z;
   double xpyzp = x*(one-y)*z;
   double xypzp = (one-x)*y*z;
   double xpypzp = x*y*z;

   XYZ->f_Dens += xyz*weight;
   XpYZ->f_Dens += xpyz*weight;
   XYpZ->f_Dens += xypz*weight;
   XpYpZ->f_Dens += xpypz*weight;
   XYZp->f_Dens += xyzp*weight;
   XpYZp->f_Dens += xpyzp*weight;
   XYpZp->f_Dens += xypzp*weight;
   XpYpZp->f_Dens += xpypzp*weight;

   int isort = p->GetSort();
   if (isort>=0 && isort < Domain::p_D->GetNsorts()) {
      weight = abs_weight;
      XYZ->f_DensArray[isort] += xyz*weight;
      XpYZ->f_DensArray[isort] += xpyz*weight;
      XYpZ->f_DensArray[isort] += xypz*weight;
      XpYpZ->f_DensArray[isort] += xpypz*weight;
      XYZp->f_DensArray[isort] += xyzp*weight;
      XpYZp->f_DensArray[isort] += xpyzp*weight;
      XYpZp->f_DensArray[isort] += xypzp*weight;
      XpYpZp->f_DensArray[isort] += xpypzp*weight;
   }
}

//---Cell3D::--------------------------------------------->
Cell3D::Cell3D(Mesh* mesh, long i, long j, long k)
{
   m = mesh;
   long n = m->GetN(i,j,k);
   XYZ = &m->GetCell(n);
   XpYZ = &m->GetCell(m->Xp(n));
   XYpZ = &m->GetCell(m->Yp(n));
   XpYpZ = &m->GetCell(m->Xp(m->Yp(n)));
   XYZp = &m->GetCell(m->Zp(n));
   XpYZp = &m->GetCell(m->Xp(m->Zp(n)));
   XYpZp = &m->GetCell(m->Yp(m->Zp(n)));
   XpYpZp = &m->GetCell(m->Xp(m->Yp(m->Zp(n))));
}
