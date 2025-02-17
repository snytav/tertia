#ifndef H_STENCIL
#define H_STENCIL

//---------------------------- FieldStencil class -----------------------
class FieldStencil {
public:
  double f_Hy2Hx, f_Hz2Hx, f_S3, f_Co, f_BetaX, f_BetaY, f_BetaZ,
    f_AlfaX, f_AlfaY, f_AlfaZ;
  double f_Bav1, f_Bav2, f_Bav3;
};

#endif
