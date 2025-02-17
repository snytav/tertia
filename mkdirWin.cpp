#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <errno.h>

int MakeNewDirectory(char* str_Directory) {
   int ierr = _mkdir(str_Directory);
   if (ierr==ENOENT) exit(ENOENT);
   return ierr;
}
