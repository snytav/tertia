#ifndef H_CELL3D
#define H_CELL3D

#include "vlpl3d.h"

//---------------------------- Cell3D class -----------------------
class Cell3Dm {
  friend class Mesh;
 private:
  Mesh *m;
  Cell *XYZ;
  Cell *XYmZ;
  Cell *XYZm;
  Cell *XYmZm;
  Cell *XmYZ;
  Cell *XmYmZ;
  Cell *XmYZm;
  Cell *XmYmZm;
 public:
  void Next(void) {
    XYZ++;
    XYmZ++;
    XYZm++;
    XYmZm++;
    XmYZ++;
    XmYmZ++;
    XmYZm++;
    XmYmZm++;
  }

  void AddDensity(Particle *p, double polarity);
  Cell3Dm(Mesh* m, long i, long j, long k);
};
//---------------------------- Cell3D class -----------------------
class Cell3D {
  friend class Mesh;
 private:
  Mesh *m;
  Cell *XYZ;
  Cell *XYpZ;
  Cell *XYZp;
  Cell *XYpZp;
  Cell *XpYZ;
  Cell *XpYpZ;
  Cell *XpYZp;
  Cell *XpYpZp;
 public:
  void Next(void) {
    XYZ++;
    XYpZ++;
    XYZp++;
    XYpZp++;
    XpYZ++;
    XpYpZ++;
    XpYZp++;
    XpYpZp++;
  }

  void AddDensity(Particle *p, double polarity);
  Cell3D(Mesh* m, long i, long j, long k);
};

#endif // H_CELL3D
