#include <stdlib.h>

#include <math.h>

static long move3d_count;

#include "vlpl3d.h"

//---Mesh::Move3d ---------------------------------------------->
void Mesh::Move3d(Particle* p, Cell* cSource, long i, long j, long k,
		   double &xtmp, double &ytmp, double &ztmp,
		   double &x,    double &y,    double &z,
		   double &dx,   double &dy,   double &dz,
		   double djx,   double djy,   double djz)
{
  move3d_count++;

  if (move3d_count == 3455)
  {
		domain()->out_Flog << "Strange from move3: unexpected xtmp="<<xtmp
			   << " ytmp="<<ytmp << " ztmp="<<ztmp
			   <<" dx="<<dx << " dy="<<dy << " dz="<<dz<<"\n";
  }

  double part1, part2, part3;
  if (xtmp<-1 || xtmp>2 || ytmp<-1 || ytmp>2 || ztmp<-1 || ztmp>2)
  {
		domain()->out_Flog << "Error0 from move3: "<< move3d_count <<
	  "unexpected xtmp="<<xtmp <<
	  " ytmp="<<ytmp << " ztmp="<<ztmp <<
	  " dx="<<dx << " dy="<<dy << " dz="<<dz<<"\n";
		domain()->out_Flog.flush();

		exit(-10);
  }

  if (xtmp > 1 && dx > 0)
  {
    if (ytmp > 1 && dy > 0)
    {
      if (ztmp > 1 && dz > 0)
      {
				part1 = (1.-x)/dx;
				part2 = (1.-y)/dy;
				part3 = (1.-z)/dz;
				if (part1 < part2 && part1 < part3)
				{ // first MoveXp
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part2 = (1.-y)/dy;
				  part3 = (1.-z)/dz;
				  if (part2 < part3)
				  { // second MoveYp
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    part3 = (1.-z)/dz;
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZp
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    part2 = (1.-y)/dy;
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				if (part2 < part1 && part2 < part3)
				{ // first MoveYp
				  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part3 = (1.-z)/dz;
				  if (part1 < part3)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZp
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				{ // first MoveZp
				  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part2 = (1.-y)/dy;
				  if (part1 < part2)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveYp
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
      }
      else
      if (ztmp < 0 && dz < 0)
      {
				part1 = (1.-x)/dx;
				part2 = (1.-y)/dy;
				part3 = -z/dz;
				if (part1 < part2 && part1 < part3)
				{ // first MoveXp
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part2 = (1.-y)/dy;
				  part3 = -z/dz;
				  if (part2 < part3)
				  { // second MoveYp
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZm
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				if (part2 < part1 && part2 < part3)
				{ // first MoveYp
				  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part3 = -z/dz;
				  if (part1 < part3)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZm
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				{ // first MoveZm
				  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part2 = (1.-y)/dy;
				  if (part1 < part2)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveYp
				    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
      }
      else
      { // Z
				part1 = (1.-x)/dx;
				part2 = (1.-y)/dy;
				if (part1 < part2)
				{ // first MoveXp
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				}
				else
				{ // first MoveYp
				  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				}
      } // Z
    }
    else
    if (ytmp < 0 && dy < 0)
    {
      if (ztmp > 1 && dz > 0)
      {
				part1 = (1.-x)/dx;
				part2 = -y/dy;
				part3 = (1.-z)/dz;
				if (part1 < part2 && part1 < part3)
				{ // first MoveXp
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part2 = -y/dy;
				  part3 = (1.-z)/dz;
				  if (part2 < part3)
				  { // second MoveYm
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZp
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				if (part2 < part1 && part2 < part3)
				{ // first MoveYm
				  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part3 = (1.-z)/dz;
				  if (part1 < part3)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    			MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZp
				    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				{ // first MoveZp
				  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part2 = -y/dy;
				  if (part1 < part2)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveYm
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
      }
      else
      if (ztmp < 0 && dz < 0)
      {
				part1 = (1.-x)/dx;
				part2 = -y/dy;
				part3 = -z/dz;
				if (part1 < part2 && part1 < part3)
				{ // first MoveXp
				  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part2 = -y/dy;
				  part3 = -z/dz;
				  if (part2 < part3)
				  { // second MoveYm
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZm
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				}
				else
				if (part2 < part1 && part2 < part3)
				{ // first MoveYm
				  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  part1 = (1.-x)/dx;
				  part3 = -z/dz;
				  if (part1 < part3)
				  { // second MoveXp
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
				  else
				  { // second MoveZm
				    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
				  }
			}
			else
			{ // first MoveZp
			  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
			  part1 = (1.-x)/dx;
			  part2 = -y/dy;
			  if (part1 < part2)
			  { // second MoveXp
			    MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
			    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
			  }
			  else
			  { // second MoveYm
			    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    		MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
			  }
			}
     }
     else
     { // Z
				part1 = (1.-x)/dx;
				part2 = -y/dy;
				if (part1 < part2)
				{ // first MoveXp
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } // Z
    } else { // Y stays
      if (ztmp > 1 && dz > 0) {
	part1 = (1.-x)/dx;
	part3 = (1.-z)/dz;
	if (part1 < part3) { // first MoveXp
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else if (ztmp < 0 && dz < 0) {
	part1 = (1.-x)/dx;
	part3 = -z/dz;
	if (part1 < part3) { // first MoveXp
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZp
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else { // Z stays
	MoveXp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } // Z
    } // Y
  } else if (xtmp < 0 && dx < 0) {
    if (ytmp > 1 && dy > 0) {
      if (ztmp > 1 && dz > 0) {
	part1 = -x/dx;
	part2 = (1.-y)/dy;
	part3 = (1.-z)/dz;
	if (part1 < part2 && part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part2 = (1.-y)/dy;
	  part3 = (1.-z)/dz;
	  if (part2 < part3) { // second MoveYp
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZp
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else if (part2 < part1 && part2 < part3) { // first MoveYp
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part3 = (1.-z)/dz;
	  if (part1 < part3) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZp
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part2 = (1.-y)/dy;
	  if (part1 < part2) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveYp
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	}
      } else if (ztmp < 0 && dz < 0) {
	part1 = -x/dx;
	part2 = (1.-y)/dy;
	part3 = -z/dz;
	if (part1 < part2 && part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part2 = (1.-y)/dy;
	  part3 = -z/dz;
	  if (part2 < part3) { // second MoveYp
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZm
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else if (part2 < part1 && part2 < part3) { // first MoveYp
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part3 = -z/dz;
	  if (part1 < part3) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZm
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else { // first MoveZm
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part2 = (1.-y)/dy;
	  if (part1 < part2) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveYp
	    MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	}
      } else { // Z
	part1 = -x/dx;
	part2 = (1.-y)/dy;
	if (part1 < part2) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveYp
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } // Z
    } else if (ytmp < 0 && dy < 0) {
      if (ztmp > 1 && dz > 0) {
	part1 = -x/dx;
	part2 = -y/dy;
	part3 = (1.-z)/dz;
	if (part1 < part2 && part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part2 = -y/dy;
	  part3 = (1.-z)/dz;
	  if (part2 < part3) { // second MoveYm
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZp
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else if (part2 < part1 && part2 < part3) { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part3 = (1.-z)/dz;
	  if (part1 < part3) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZp
	    MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part2 = -y/dy;
	  if (part1 < part2) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveYm
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	}
      } else if (ztmp < 0 && dz < 0) {
	part1 = -x/dx;
	part2 = -y/dy;
	part3 = -z/dz;
	if (part1 < part2 && part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part2 = -y/dy;
	  part3 = -z/dz;
	  if (part2 < part3) { // second MoveYm
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZm
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else if (part2 < part1 && part2 < part3) { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part3 = -z/dz;
	  if (part1 < part3) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveZm
	    MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	} else { // first MoveZm
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  part1 = -x/dx;
	  part2 = -y/dy;
	  if (part1 < part2) { // second MoveXm
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  } else { // second MoveYm
	    MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	    MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  }
	}
      } else { // Z
	part1 = -x/dx;
	part2 = -y/dy;
	if (part1 < part2) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } // Z
    } else { // Y stays
      if (ztmp > 1 && dz > 0) {
	part1 = -x/dx;
	part3 = (1.-z)/dz;
	if (part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else if (ztmp < 0 && dz < 0) {
	part1 = -x/dx;
	part3 = -z/dz;
	if (part1 < part3) { // first MoveXm
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZm
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else { // Z stays
	MoveXm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } // Z
    } // Y
  } else { // X stays
    if (ytmp > 1 && dy > 0) {
      if (ztmp > 1 && dz > 0) {
	part2 = (1.-y)/dy;
	part3 = (1.-z)/dz;
	if (part2 < part3) { // first MoveYp
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else if (ztmp < 0 && dz < 0) {
	part2 = (1.-y)/dy;
	part3 = -z/dz;
	if (part2 < part3) { // first MoveYp
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZm
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else { // Z
	MoveYp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } // Z
    } else if (ytmp < 0 && dy < 0) {
      if (ztmp > 1 && dz > 0) {
	part2 = -y/dy;
	part3 = (1.-z)/dz;
	if (part2 < part3) { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZp
	  MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } else if (ztmp < 0 && dz < 0) {
	part2 = -y/dy;
	part3 = -z/dz;
	if (part2 < part3) { // first MoveYm
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	} else { // first MoveZm
	  MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	  MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
	}
      } // Zm
      else { // Z
	part2 = -y/dy;
	MoveYm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } // Z
    } else { // Y stays
      if (ztmp > 1 && dz > 0) {
	MoveZp(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } else if (ztmp < 0 && dz < 0) {
	MoveZm(p, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
      } else { // X Y Z stays
	domain()->out_Flog << "Error from move3:  "<< move3d_count <<
	  "unexpected xtmp="<<xtmp <<
	  " ytmp="<<ytmp << " ztmp="<<ztmp <<
	  " dx="<<dx << " dy="<<dy << " dz="<<dz<<"\n";
	domain()->out_Flog.flush();
	//	exit(-9);
      } // Z
    } // Y
  } // X
  MoveInCell(p, cSource, i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
}

//---Mesh::MoveXp ---------------------------------------------->
void Mesh::MoveXp(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = (1.-x)/dx0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  i++;
  x = 0.;      y += dy;   z += dz;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveXm ---------------------------------------------->
void Mesh::MoveXm(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = -x/dx0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  i--;
  x = 1.;      y += dy;   z += dz;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveYp ---------------------------------------------->
void Mesh::MoveYp(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = (1.-y)/dy0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  j++;
  x += dx;      y = 0.;   z += dz;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveYm ---------------------------------------------->
void Mesh::MoveYm(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = -y/dy0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  j--;
  x += dx;      y = 1.;   z += dz;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveZp ---------------------------------------------->
void Mesh::MoveZp(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = (1.-z)/dz0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  k++;
  x += dx;   y += dy;    z = 0.;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveZm ---------------------------------------------->
void Mesh::MoveZm(Particle *p, long &i, long &j, long &k,
	     double &x, double &dx0, double &y, double &dy0,
	     double &z, double &dz0, double djx, double djy, double djz)
{
  double part = -z/dz0;
  double dx = dx0*part;
  double dy = dy0*part;
  double dz = dz0*part;
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  k--;
  x += dx;     y += dy;   z  = 1.;
  dx0 -= dx; dy0 -= dy; dz0 -= dz;
}

//---Mesh::MoveInCell ---------------------------------------------->
void Mesh::MoveInCell(Particle *p, Cell* cOld,
		 long i, long j, long k,
		 double &x, double &dx, double &y, double &dy,
		 double &z, double &dz, double djx, double djy, double djz)
{
  Cell* cMoveIn = &GetCell( i, j, k);
  //  if (GetN(i,j,k)==3673) {
  //    domain()->out_Flog << "Move3d: i j k \n";
  //  };
  MoveSimple(i, j, k, x, dx, y, dy, z, dz, djx, djy, djz);
  x += dx; y += dy; z += dz;
  if (x<0||x>1 || y<0||y>1 || z<0||z>1) {
    domain()->out_Flog << "Wrong MoveInCell: "<< move3d_count <<"  x="
		       <<x<<" y="<<y<<" z="<<z
		       <<" dx="<<dx << " dy="<<dy << " dz="<<dz<<"\n";
    domain()->out_Flog << "Count =" << move3d_count << "\n";
    domain()->out_Flog.flush();
    if (x<0) x=0.;
    if (y<0) y=0.;
    if (z<0) z=0.;
    if (x>1) x=0.999999;
    if (y>1) y=0.999999;
    if (z>1) z=0.999999;

    //  exit(-13);
  }
  p->l_Cell = cMoveIn->l_N;
  dx = 0.; dy = 0.; dz = 0.;
  Hook( p, p_PrevPart, cOld, cMoveIn);
}
