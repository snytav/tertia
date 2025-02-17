#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

int MakeNewDirectory(char* str_Directory) {
   int ierr = mkdir(str_Directory, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
   //   if (ierr==ENOENT) exit(ENOENT);
   return ierr;
}
