#ifndef H_MOVIEFRAME
#define H_MOVIEFRAME

#include <stdio.h>



typedef double (Cell::*GETTYPE)(void);

//---------------------------- Pulse class -----------------------

class MovieFrame : public NList {
	friend class Domain;
	friend class Mesh;
private:
	Domain *domain() {return Domain::p_D;};
	long Save(FILE* pFile);
	char **str_What;
	GETTYPE *Gets;
	//  long Save(FILE* pFile);
	MovieFrame(char *nm, FILE *f, int nframes=0);
};
#endif
