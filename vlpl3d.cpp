#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#define X_ACCESS

#include "vlpl3d.h"

#include "para.h"


int main(int argc, char** argv)
{
  char* infile;

  int rank = 0;
  int myid, numprocs;
  int  namelen;

#ifdef V_MPI
  char processor_name[MPI_MAX_PROCESSOR_NAME];
#endif

    ParallelInit(argc,argv);
    
/*    printf("after parallel init rank %d \n",GetRank());
    exit(0);

    ParallelExit();
    
    exit(0);
*/

  if (argc < 2) {
	cout << "Usage: v3d <ini-file> \n";
	cout << "Default name: v.ini will be used \n";
    //    return -10;
    infile = "v.ini";
  } else {
    infile = argv[1];
    cout << "The start file is: " << argv[1] << " \n";
  }

infile = "v.ini";


#ifdef V_MPI
    if(MPI_ERRORS_RETURN == MPI_Init(&argc,&argv))
    {
    	char * error = "MPI_ERRORS_RETURN";
    	printf("%s",error);
    	exit MPI_ERRORS_RETURN;
    };
    int buf_size = 100000000 + MPI_BSEND_OVERHEAD;
    char *p_b = new char[buf_size];
    int err_buf_att = MPI_Buffer_attach( p_b, buf_size);
 
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(processor_name,&namelen);
    fprintf(stderr,"Process %d on %s\n",
	    rank, processor_name);
#endif

  srand( time(NULL) + rank );
  Domain *domain = new Domain(infile, rank);

#ifdef V_MPI
  //  MPI_Buffer_detach(p_b, &buf_size);
  //  delete[] p_b;
#endif

#ifdef X_ACCESS
  Plot *interactive = domain->plot;
  if (interactive->nview) {
    interactive->Run();
  }
  else {
    domain->Run();
  }
#endif


//printf("PARA init rank %d l_Mx %d \n",GetRank(),l_Mx);

#ifdef NO_X_ACCESS
    domain->Run();
#endif
  delete domain;

#ifdef V_MPI
  MPI_Finalize();
#endif
  
  ParallelFinalize();
  
  return 0;
}

