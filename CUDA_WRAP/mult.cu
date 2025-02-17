

#include "mult.h"

cublasHandle_t  handle;
cublasStatus_t  stat;
double         *d_D_UnityMatrix;
cuDoubleComplex *d_Z_alphaTurnVector,*d_Z_omegaTurnVector;
cuDoubleComplex *d_Z_alphaTurnMatrix,*d_Z_omegaTurnMatrix;

#define N 100

int CUDA_WRAP_blas_init()
{
    stat = cublasCreate(&handle);
    
//    cublasInit();

       puts("cublas init");
    
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
       puts("cublas init fail");
       exit(0);
    }
    
    return 0;
}

int CUDA_WRAP_copyMatrix_toDevice(int n1,int n2,double **d_cm,double *cm)
{
  
    int err = cudaMalloc((void **)d_cm,n1*n2*sizeof(cuDoubleComplex));
    printf("mtrx alloc error %d \n",err);
    err = cudaMemcpy(*d_cm,cm,n1*n2*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    printf("mtrx copy error %d \n",err);
    
    return 0;
}

// BOTH VECTORS ARE COMPLEX OF 2*n SIZE
int CUDA_WRAP_MakeTurnVectors(int n,double *h_alpha,double *h_omega)
{
    cudaMalloc((void **)&d_Z_alphaTurnVector,n*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_Z_omegaTurnVector,n*sizeof(cuDoubleComplex));
    
    cudaMemcpy(d_Z_alphaTurnVector,h_alpha,n*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z_omegaTurnVector,h_omega,n*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
    
    return 0;
}

// BOTH VECTORS ARE COMPLEX OF 2*n SIZE
// TURN VECTORS FOR;
int CUDA_WRAP_MakeTurnMatrices(int n1,int n2,double *h_alpha,double *h_omega)
{

    cudaMalloc((void **)&d_Z_alphaTurnMatrix,n1*n2*sizeof(cuDoubleComplex));
    cudaMalloc((void **)&d_Z_omegaTurnMatrix,n1*n2*sizeof(cuDoubleComplex));
    
    cudaMemset(d_Z_alphaTurnMatrix,0,sizeof(cuDoubleComplex)*n1*n2);
    cublasSetVector(n1,sizeof(cuDoubleComplex),h_alpha,1,d_Z_alphaTurnMatrix,n2+1);     

    cudaMemset(d_Z_alphaTurnMatrix,0,sizeof(cuDoubleComplex)*n1*n2);
    cublasSetVector(n1,sizeof(cuDoubleComplex),h_alpha,1,d_Z_alphaTurnMatrix,n2+1);     
    
    return 0;
}

int CUDA_WRAP_compareDmatrix_from_host(int n1,int n2,double *d,double *h)
{
    double *h_copy_of_d,t,dmax = 0;
    
    
    h_copy_of_d = (double*)malloc(n1*n2*sizeof(double));
    
    cudaMemcpy(h_copy_of_d,d,n1*n2*sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < n1;i++)
    {
       for(int j = 0;j < n2;j++)
       {
           if(fabs(h[i] - h_copy_of_d[i]) > dmax) dmax = t;
           if(t > CUDA_DOUBLE_TOLERANCE)
           {
              printf("%10d host %25.15e device %25.15e \n",h[i],h_copy_of_d[i]);
           }
       }
    }
     
    return 0; 
}




int CUDA_WRAP_print_error(cublasStatus_t stat,char *where)
{
    puts(where);
    
    if(stat == CUBLAS_STATUS_SUCCESS) puts("Success");
    if(stat == CUBLAS_STATUS_NOT_INITIALIZED) puts("not init");
    if(stat == CUBLAS_STATUS_INVALID_VALUE) puts("invalid value");
    if(stat == CUBLAS_STATUS_ARCH_MISMATCH) puts("arch");
    if(stat == CUBLAS_STATUS_EXECUTION_FAILED) puts("failed");

   return 0;
}



int CUDA_WRAP_vectorD_mult_vector_from_host(int n1,int n2,double *h_y1,double *h_x_re,double *h_a_re)
{
    double alpha = 1.0,beta = 0.0;
    double *d_x,*d_y,*d_a;
    
    cudaMalloc((void**)&d_x,sizeof(double)*n1);
    cudaMalloc((void**)&d_y,sizeof(double)*n1*n2);
    
    cudaMemset(d_y,0,sizeof(double)*n1*n2);
    
    cudaMalloc((void**)&d_a,sizeof(double)*n1);     

    cublasSetVector(n1,sizeof(double),h_y1,1,d_y,n2+1); 
    
    cublasSetVector(n1,sizeof(double),h_x_re,1,d_x,1); 

    stat = cublasDgemv(handle,            // 1
                CUBLAS_OP_N,              // 2
                n1,                       // 3
                n2,                       // 4
                &alpha,                   // 5
                (double*)d_y,             // 6
                n1,
                (double*)d_x,
                1,
                &beta,
                (double*)d_a,
                1);

    stat = cublasGetVector(n1,sizeof(double),d_a,1,h_a_re,1); 
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_a);

    return 0;
}

//COMPONENT-WISE MULTIPLICATION OF TWO COMPLEX VECTORS (A = XxY)
//PARAMETERS:
//h_y1 - COMPLEX VECTOR (Y)
//h_x_re,h_x_im _ REAL VECTORS ASSUMED TO BE REAL AND IMAGINARY PARTS OF THE X VECTOR
//h_a_re,h_a_im _ REAL VECTORS GIVING THE REAL AND IMAGINARY PARTS OF THE RESULT VECTOR A
int CUDA_WRAP_vectorZ_mult_vector_from_host(int n1,int n2,double *h_y1,double *h_x_re,double *h_x_im,double *h_a_re,double *h_a_im)
{
    cuDoubleComplex alpha = {1.0,0.0},beta = {0.0,0.0};
    cuDoubleComplex *d_x,*d_a,*d_y;
    double *h_x,*h_a;
    
    h_x = (double *)malloc(2*n1*sizeof(double));
    h_a = (double *)malloc(2*n1*sizeof(double));
    for(int i = 0;i < n1;i++)
    {
        h_x[2*i]   = h_x_re[i];
        h_x[2*i+1] = h_x_im[i];
    }
    
    cudaMalloc((void**)&d_a,sizeof(cuDoubleComplex)*n1);
    cudaMalloc((void**)&d_x,sizeof(cuDoubleComplex)*n1);
    cudaMemset(d_x,0,sizeof(cuDoubleComplex)*n1);
    
    cublasSetVector(2*n1,sizeof(double),h_x,1,d_x,1); 
    cudaMalloc((void**)&d_y,sizeof(cuDoubleComplex)*n1*n2);
    cudaMemset(d_y,0,sizeof(cuDoubleComplex)*n1*n2);
    cublasSetVector(n1,sizeof(cuDoubleComplex),h_y1,1,d_y,n2+1); 

    stat = cublasZgemv(handle,            // 1
                CUBLAS_OP_N,              // 2
                n1,                       // 3
                n2,                       // 4
                &alpha,                   // 5
                (cuDoubleComplex*)d_y,             // 6
                n1,
                (cuDoubleComplex*)d_x,
                1,
                &beta,
                (cuDoubleComplex*)d_a,
                1);

    stat = cublasGetVector(n1,sizeof(cuDoubleComplex),d_a,1,h_a,1); 
    for(int i = 0;i < n1;i++)
    {
        h_a_re[i] = h_a[2*i];
        h_a_im[i] = h_a[2*i+1];
    }
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_a);
    
    free(h_a);
    free(h_x);

    return 0;
}

//h_m - COMPLEX MATRIX TO ROTATE
//mode - 0 FOR alpha TURN MATRIX
//       1 FOR OMEGA TURN MATRIX 
int CUDA_WRAP_rotateZmatrix_from_host(int n1,int n2,double *h_m,int mode,double *h_result)
{
    cuDoubleComplex *d_m,alpha = {1.0,0.0},beta = {0.0,0.0},*turn,*d_C;
    struct timeval tv1,tv2;
    
    int err = cudaMalloc((void**)&d_m,sizeof(cuDoubleComplex)*n1*n2);
    printf("error %d\n",err);
    
    cudaMalloc((void**)&d_C,sizeof(cuDoubleComplex)*n1*n2);
    
    err = cudaMemcpy(d_m,h_m,sizeof(cuDoubleComplex)*n1*n2,cudaMemcpyHostToDevice);
    printf("error %d\n",err);
    //cublasSetVector(n1*n2,sizeof(cuDoubleComplex),h_m,1,d_m,1);
    
    if(mode == 0)
    {
       turn = d_Z_alphaTurnMatrix;
    }
    else
    {
       turn = d_Z_omegaTurnMatrix;
    }
    
//    CUDA_DEBUG_printZdevice_matrix(n1,n2,d_Z_alphaTurnMatrix,"turn");
//    CUDA_DEBUG_printZdevice_matrix(n1,n2,d_m,"target matrix");
    //exit(0);
    
    gettimeofday(&tv1,NULL);
    stat = cublasZgemm(handle,           // 1
                CUBLAS_OP_N,      // 2
                CUBLAS_OP_N,      // 3
                n1,               // 4
                n2,               // 5
                n1,               // 6
                &alpha,           // 7
                d_m,                // 8
                1,                // 9
                turn,             // 10
                1,                // 11 
                &beta,            // 12
                d_C,              // 13
                1                 // 14
                );
      CUDA_WRAP_print_error(stat,"zgemm");          
    gettimeofday(&tv2,NULL);
    printf("multiplication time %g\n",(tv2.tv_sec - tv1.tv_sec)+(tv2.tv_sec - tv1.tv_sec)*1e-6);
//    CUDA_DEBUG_printZdevice_matrix(n1,n2,d_C,"result");
    exit(0);
                
    cudaMemcpy(h_result,d_C,sizeof(cuDoubleComplex)*n1*n2,cudaMemcpyDeviceToHost);
    
    return 0;
}    
                
                
    
    
    



/*
int main()
{
   double x_re[N],x_im[N],b[2*N],c[N],d[N];
   
   CUDA_WRAP_blas_init();
  // CUDA_WRAP_make_Dunity_matrix(N,N);
   
   for(int i = 0;i < N ; i++)
   {
       x_re[i]     = 0.0;
       x_im[i]     = i+1;
       b[2*i]      = 0.1*(i+1);
       b[2*i+1]    = 0.0;
       printf("%5d a (%10.3e,%10.3e) (%10.3e,%10.3e) \n",i,x_re[i],x_im[i],b[2*i],b[2*i+1]);
   }
       
   CUDA_WRAP_vectorZ_mult_vector_from_host(N,N,b,x_re,x_im,c,d);

   for(int i = 0;i < N ; i++)
   {
       printf("%5d a %15f %15f \n",i,c[i],d[i]);
   }
   

//   printf("nmax %d max %g c %g\n",h_n_max,a[h_n_max - 1]);
   
   return 0;
}

*/