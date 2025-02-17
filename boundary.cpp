#include <stdio.h>
#include <math.h>
#include <stdlib.h>

           
#include "vlpl3d.h"

//---------------------------- Boundary::Boundary -----------------------
Boundary::Boundary (char *nm, FILE *f, char where, Processor *pe) : NList (nm)
{
  if (where==TOXM || where==TOXP || 
      where==TOYM || where==TOYP ||
      where==TOZM || where==TOZP)  c_Where = where;
  else exit(where);
  p_MatePE = pe;

  /*/ <DEBUG>
    switch(where)
    {
    case TOXM:
    domain()-> out_Flog << "where = " << "TOXM" << endl;
    break;
		
    case TOXP:
    domain()-> out_Flog << "where = " << "TOXP" << endl;
    break;

    case TOYM:
    domain()-> out_Flog << "where = " << "TOYM" << endl;
    break;
		
    case TOYP:
    domain()-> out_Flog << "where = " << "TOYP" << endl;
    break;

    case TOZM:
    domain()-> out_Flog << "where = " << "TOZM" << endl;
    break;

    case TOZP:
    domain()-> out_Flog << "where = " << "TOZP" << endl;
    break;
    }

    // </DEBUG> */

  p_Zombies = NULL;

  AddEntry("FieldCondition", &i_FieldCnd);
  AddEntry("ParticlesCondition", &i_ParticleCnd);
  //   AddEntry("FieldMask", &i_RefreshN);

  if (f)
    {
      rewind(f);
      read(f);
    }

#ifdef V_MPI
  CBuffer *buf = domain()->GetMPP()->GetBuf();
  buf->reset();
  pack_nls(buf);
  domain()->BroadCast(buf);
  if (f==NULL) unpack_nls(buf);
#endif

  if (!domain()->XmEdge() && where==TOXM)
    {
      i_FieldCnd = i_ParticleCnd = 0;
      //  	domain()-> out_Flog << "where = " << "TOXM" << endl;
    }
  else
  	
    if (!domain()->XpEdge() && where==TOXP)
      {
  	i_FieldCnd = i_ParticleCnd = 0;
	//  	domain()-> out_Flog << "where = " << "TOXP" << endl;
      }
    else
  	
      if (!domain()->YmEdge() && where==TOYM)
	{
	  i_FieldCnd = i_ParticleCnd = 0;
	  //  	domain()-> out_Flog << "where = " << "TOYM" << endl;
	}
      else
  	
	if (!domain()->YpEdge() && where==TOYP)
	  {
	    i_FieldCnd = i_ParticleCnd = 0;
	    //  	domain()-> out_Flog << "where = " << "TOYP" << endl;
	  }
	else
  	
	  if (!domain()->ZmEdge() && where==TOZM)
	    {
	      i_FieldCnd = i_ParticleCnd = 0;
	      //  	domain()-> out_Flog << "where = " << "TOZM" << endl;
	    }
	  else
  	
	    if (!domain()->ZpEdge() && where==TOZP)
	      {
		i_FieldCnd = i_ParticleCnd = 0;
		//  	domain()-> out_Flog << "where = " << "TOZP" << endl;
	      }
	
  domain()-> out_Flog << "i_FieldCnd = " << i_FieldCnd << "\n" << "i_ParticleCnd = " << i_ParticleCnd << endl << endl;

}

// Boundary::AddZombie--------------

Particle* Boundary::AddZombie(Particle* p, Cell* c) {
  if (p==NULL) return NULL;
  if (p->GoingZombie()) {
    Specie *spec = p->GetSpecie();
    spec->Remove();
    c->RemoveParticle(p, NULL);
    p->p_Next = p_Zombies;
    p_Zombies = p;
  } else {
    c->KillParticle(p, NULL);
  };
  return p_Zombies;
};

// Boundary::SaveZombies--------------

long Boundary::SaveZombies(FILE *pFile) {
  long ncount = 0;
  if (pFile == NULL) return 0;
  Particle *p = p_Zombies;
  while(p) {
    long isort = p->GetSort();
    fwrite(&isort, sizeof(long), 1, pFile);
    fwrite(&p->f_X, sizeof(double), 1, pFile);
    fwrite(&p->f_Y, sizeof(double), 1, pFile);
    fwrite(&p->f_Z, sizeof(double), 1, pFile);
    fwrite(&p->f_Px, sizeof(double), 1, pFile);
    fwrite(&p->f_Py, sizeof(double), 1, pFile);
    fwrite(&p->f_Pz, sizeof(double), 1, pFile);
    fwrite(&p->f_Weight, sizeof(double), 1, pFile);
    fwrite(&p->f_Q2m, sizeof(double), 1, pFile);
    p = p->p_Next;
    ncount++;
  }
  return ncount;
};
