//#include ""


int getCudaGrid(int ny,int nz,dim3 *dimBlock,dim3 *dimGrid)
{
    if(ny > 16) 
    {
       (*dimBlock).x = 16;
       (*dimGrid).x  = ny/16;
    }
    else
    {
       (*dimBlock).x = ny;
       (*dimGrid).x  = 1;
    }

    if(nz > 16) 
    {
       (*dimBlock).y = 16;
       (*dimGrid).y  = nz/16;
    }
    else
    {
       (*dimBlock).y = nz;
       (*dimGrid).y  = 1;
    } 
    
    return 0;
}