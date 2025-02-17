#include <stdio.h>
#include <math.h>

int ComputePhaseShift(int n,double *alp,double *omg)
{
    double beta,beta1;
    for(int i = 0;i < n;i++)
    {
        beta = -((double)i*M_PI/n/2);
        alp[2*i  ] = cos(beta);
        alp[2*i+1] = sin(beta);
	
        beta1 =  ((double)i*M_PI/n/2+M_PI/n/4);
        omg[2*i  ] = cos(beta1);
        omg[2*i+1] = sin(beta1);
	
	printf("%d beta %e alp %e %e omg %e %e \n",i,beta,alp[2*i  ],alp[2*i +1 ],omg[2*i  ],omg[2*i + 1]);
    }
  
    return 0;
}
