#include "diagnostic_print.h"
#include "../run_control.h"
//#include "beam_copy.h"


int CUDA_WRAP_GetError()
{
#ifdef CUDA_WRAP_DEBUG_ERROR_MESSAGES   
    return cudaGetLastError();
#endif
    return 0;
}

int CUDA_DEBUG_print_error(cublasStatus_t stat,char *where)
{
    puts(where);
    
    if(stat == CUBLAS_STATUS_SUCCESS) puts("Success");
    if(stat == CUBLAS_STATUS_NOT_INITIALIZED) puts("not init");
    if(stat == CUBLAS_STATUS_INVALID_VALUE) puts("invalid value");
    if(stat == CUBLAS_STATUS_ARCH_MISMATCH) puts("arch");
    if(stat == CUBLAS_STATUS_EXECUTION_FAILED) puts("failed");

   return 0;
}

int CUDA_DEBUG_printDhost_array(int n,double *d,char *legend)
{
    for(int i = 0;i < n;i++)
    {
       printf("%s %5d %15.5e \n",legend,i,d[i]);
    }
    
    return 0;
}

int CUDA_DEBUG_printDhost_matrix(int n1,int n2,double *d,char *legend)
{

    
    
  
    for(int i = 0;i < n1;i++)
    {
       printf("%s %5d ",legend,i);
       for(int j = 0;j < n2;j++)
       {
           printf("%10.3e",d[i*n2 + j]);
       }
       printf("\n");
    }
    
    return 0;
}

int CUDA_DEBUG_printDdevice_matrix(int n1,int n2,double *d,char *legend)
{
#ifdef CUDA_WRAP_FFTW_ALLOWED
     return 0;
#endif

#ifndef CUDA_WRAP_PRINT_D_MATRIX
    return 0;
#endif
    int err = cudaGetLastError();
    double *h;
    
    h = (double *)malloc(n1*n2*sizeof(double));
    
    cudaMemcpy(h,d,n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    for(int i = 0;i < n1;i++)
    {
       printf("%s %5d ",legend,i);
       for(int j = 0;j < n2;j++)
       {
           printf("%15.5e",h[i*n2+j]);
       }
       printf("\n");
    }
    free(h);
    err = cudaGetLastError();
    return 0;
}

int CUDA_DEBUG_printDdevice_matrixCentre(int n1,int n2,double *d,char *legend)
{
  
    int err = cudaGetLastError();
    double *h;
    
    h = (double *)malloc(n1*n2*sizeof(double));
    
    cudaMemcpy(h,d,n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    for(int i = n1/2 - 2;i < n1/2 + 2;i++)
    {
       printf("%s %5d ",legend,i);
       for(int j = n2/2 - 2;j < n2/2+2;j++)
       {
           printf("%15.5e",h[i*n2+j]);
       }
       printf("\n");
    }
    free(h);
    err = cudaGetLastError();
    return 0;
}



int CUDA_DEBUG_printDdevice_array(int n,double *d,char *legend)
{
    double *h;
    
    h = (double *)malloc(n*sizeof(double));
    
    cudaMemcpy(h,d,n*sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n;i++)
    {
       printf("DEVICE: %s %5d %15.5e \n",legend,i,h[i]);
    }
    free(h);
    return 0;
}



int CUDA_DEBUG_printZdevice_matrix(int n1,int n2,double *d,char *legend)
{
    double *h;
    
    h = (double *)malloc(n1*n2*sizeof(cuDoubleComplex));
    
    cudaMemcpy(h,d,n1*n2*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);

    for(int i = 0;i < n1;i++)
    {
       printf("%s %5d ",legend,i);
       for(int j = 0;j < n2;j++)
       {
           printf(" (%10.3e,%10.3e)",h[2*(i*n2+j)],h[2*(i*n2+j)+1]);
       }
       printf("\n");
    }
    free(h);
    
    return 0;
}

int VerifyComplexMatrixTransposed(int n1,int n2,CMATRIX cm,CMATRIX exact,char *s)
{
    double t,dmax = 0.0;
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
	    if((t = fabs(cm[i][j][0]-exact[j][i][0])) > dmax) dmax = t;
	    //if(t > 1e-12) printf("%s %d %d (%10.3e,%10.3e) (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[i][j][0],exact[i][j][1]);
	    

	    if((t = fabs(cm[i][j][1]-exact[j][i][1])) > dmax) dmax = t;
	    printf("%s %d %d (%10.3e,%10.3e) exact (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[j][i][0],exact[j][i][1]);
	}
    }
    printf("matrix diff %e \n",dmax);
    return 0;
}

int VerifyComplexMatrix(int n1,int n2,CMATRIX cm,CMATRIX exact,char *s)
{
    double t,dmax = 0.0;
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
	    if((t = fabs(cm[i][j][0]-exact[i][j][0])) > dmax) dmax = t;
	    //if(t > 1e-12) printf("%s %d %d (%10.3e,%10.3e) (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[i][j][0],exact[i][j][1]);
	    

	    if((t = fabs(cm[i][j][1]-exact[i][j][1])) > dmax) dmax = t;
	    printf("%s %d %d (%10.3e,%10.3e) exact (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[i][j][0],exact[i][j][1]);
	}
    }
    printf("matrix diff %e \n",dmax);
    return 0;
}

int VerifyComplexMatrixTransposed_fromDevice(int n1,int n2,double *d_cm,CMATRIX exact,char *s)
{
    double t,dmax = 0.0;
    CMATRIX cm;
    
    cudaMemcpy((double *)cm,d_cm,n1*n2*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
	    if((t = fabs(cm[i][j][0]-exact[j][i][0])) > dmax) dmax = t;
	    //if(t > 1e-12) printf("%s %d %d (%10.3e,%10.3e) (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[i][j][0],exact[i][j][1]);
	    

	    if((t = fabs(cm[i][j][1]-exact[j][i][1])) > dmax) dmax = t;
	    printf("%s %d %d (%10.3e,%10.3e) exact (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[j][i][0],exact[j][i][1]);
	}
    }
    printf("matrix diff %e \n",dmax);
    return 0;
}

int VerifyComplexMatrix_fromDevice(int n1,int n2,double *d_cm,CMATRIX exact,char *s)
{
    double t,dmax = 0.0;
    double *cm = (double *)malloc(2*n1*n2*sizeof(double));
    
    cudaMemcpy((double *)cm,d_cm,n1*n2*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n1;i++)
    {
        for(int j = 0;j < n2;j++)
	{
	    if((t = fabs(cm[2*(i*n2+j)]-exact[i][j][0])) > dmax) dmax = t;
	    //if(t > 1e-12) printf("%s %d %d (%10.3e,%10.3e) (%10.3e,%10.3e)\n",s,i,j,cm[i][j][0],cm[i][j][1],exact[i][j][0],exact[i][j][1]);
	    

	    if((t = fabs(cm[2*(i*n2+j)+1]-exact[i][j][1])) > dmax) dmax = t;
	    printf("%s %d %d device (%10.3e,%10.3e) exact (%10.3e,%10.3e)\n",s,i,j,cm[2*(i*n2+j)],cm[2*(i*n2+j)+1],exact[i][j][0],exact[i][j][1]);
	}
    }
    printf("matrix diff %e \n",dmax);
    return 0;
}

int CUDA_WRAP_output_device_matrix(int n1,int n2,char *legend,int iLayer,double *d_m)
{
    double *h;
    char s[100];
    FILE *f;
    
    h = (double *)malloc(n1*n2*sizeof(double));
    
    cudaMemcpy(h,d_m,n1*n2*sizeof(double),cudaMemcpyDeviceToHost);

    sprintf(s,"%s%03d.dat",legend,iLayer);
    if((f = fopen(s,"wt")) == NULL) return 1;
    
    for(int i = 0;i < n1;i++)
    {
       for(int j = 0;j < n2;j++)
       {
           fprintf(f,"%5d %5d %25.15e \n",i,j,h[i*n2+j]);
       }
       fprintf(f,"\n");
    }
    fclose(f);
    
    free(h);
    
    return 0;
  

}

int CUDA_WRAP_output_fields(int n1,int n2,int iLayer,double *ex,double *ey,double *ez)
{
#ifndef CUDA_WRAP_FIELD_OUTPUT
    return 1;
#endif    
    
    CUDA_WRAP_output_device_matrix(n1,n2,"EX_",iLayer,ex);
    CUDA_WRAP_output_device_matrix(n1,n2,"EY_",iLayer,ey);
    CUDA_WRAP_output_device_matrix(n1,n2,"EZ_",iLayer,ez);
    
  
    return 0;
}


int CUDA_WRAP_3Dto2Dloc(int iLayer,int Ny,int Nz,double *d_3d,double *d_2d)
{
//    double *h_copy = (double *)malloc(sizeof(double)*Ny*Nz);
    
    cudaMemcpy(d_2d,d_3d+iLayer*Ny*Nz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToDevice);
    
  //  cudaMemcpy(h_copy,d_3d+iLayer*Ny*Nz,sizeof(double)*Ny*Nz,cudaMemcpyDeviceToHost);	
    
    return 0;
}


int CUDA_DEBUG_print3DmatrixLayer(double *d_Ey3D,int iLayer,int Ny,int Nz,char *legend)
{
        double *dm;
        cudaMalloc(&dm,sizeof(double)*Ny*Nz);
        CUDA_WRAP_3Dto2Dloc(iLayer,Ny,Nz,d_Ey3D,dm);

        CUDA_DEBUG_printDdevice_matrix(Ny,Nz,dm,legend);
        
        cudaFree(dm);
        return 0;
}  


