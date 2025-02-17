/***************************************************************************
                          defines.h  -  description
                             -------------------
    begin                : August 2007
    copyright            : (C) 2007 by Alexander PUKHOV
    email                : pukhov@tp1.uni-duesseldorf.de
 ***************************************************************************/
#ifndef DEFINES_H
#define DEFINES_H

// Also please see version.h

//<graphics/>
#define NO_X_ACCESS
#undef X_ACCESS
//</graphics>

#define NIL 0

#define S_BAVERAGE

//#define _SAVE_PARTICLES   // Save particles: Particles which reached boundaries will be saved into files.
                          // The postprocessing of these files has to be done with
                          // #define _LOAD_PARTICLES and #undef _SAVE_PARTICLES
                                                                        
//#define COLLISIONS      // see movepart.C
//#define CURRENT_COMPENSATION 

#define BUF_SIZE 16384

//#define MOVIE_CONVERT //Big endian <-> Little endian

#endif // DEFINES_H
