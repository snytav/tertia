/* File:    cuprintf_test.cu
 * Purpose: Illustrate the use of cuPrintf.cu
 *
 * Compile:  
 *    Linux:
 *       nvcc -o cuprintf_test cuprintf_test.cu 
 *          -I/usr/local/NVIDIA_GPU_Computing_SDK/shared/inc
 *          -I/usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc
 *    MacOS:
 *       nvcc -o cuprintf_test cuprintf_test.cu 
 *          -I/Developer/GPU\ Computing/shared/inc
 *          -I/Developer/GPU\ Computing/C/common/inc
 *
 * Run: ./cuprintf_test
 *
 * Input:   None
 * Output:  6 copies of "Value is 10", each preceded by the block and thread
 *          numbers of the the thread that printed.
 *
 * Notes:    
 * 1.  In addition to adding the include paths listed above, this 
 *     assumes you've copied cuPrint.cu and cuPrintf.cuh from the
 *
 *       /usr/local/NVIDIA_GPU_Computing_SDK/C/src/simplePrintf
 *
 *     on Linux or
 *          
 *       /Developer/GPU\ Computing/C/src/simplePrintf
 *
 *     on MacOS
 *
 * 2.  This is a much simplified version of the program simplePrintf.cu
 *     from the CUDA SDK.
 */
#include <stdio.h>
#include "cuPrintf.cu"
//#include "shrUtils.h"
//#include "cutil_inline.h"


__global__ void testKernel(double  val) {
   cuPrintf("\tValue is:%e\n", val);
}

int main(int argc, char *argv[]) {
   cudaPrintfInit();
   testKernel<<< 2, 3 >>>(10); 
   cudaPrintfDisplay(stdout, true);
   cudaPrintfEnd();

   return 0;
}
