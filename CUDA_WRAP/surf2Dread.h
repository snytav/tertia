#ifndef SURF_2D_READ_H


class SurfaceEmulator{
    int N,M;
public:
    SurfaceEmulator(int n,int m){
        N = n; M = m;
        matrix = new double[N*M];
    }
    double *matrix;

    double surf2Dread( int i, int j)
    {
        return matrix[i*M+j];
    }
    double surf2Dwrite( int i, int j,double d)
    {
        matrix[i*M+j] = d;
    }

};

#endif
