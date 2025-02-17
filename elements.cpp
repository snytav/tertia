#include <stdlib.h>
#include <math.h>

#include "vlpl3d.h"

//---------------------------- Specie::Specie -----------------------
Specie::Specie(char *nm, FILE *f) : NList(nm)
{
  Domain *d = domain();
  f_Px0 = f_Py0 = f_Pz0 = 0;
   AddEntry("Distribution", &i_Distribution, 0);  // uniform by Default
   AddEntry("f_qCGS", &f_qCGS);
   AddEntry("f_mCGS", &f_mCGS);
   AddEntry("f_q2mCGS", &f_q2mCGS);
   AddEntry("f_q", &f_q);
   AddEntry("f_m", &f_m);
   AddEntry("f_q2m", &f_q2m);
   AddEntry("f_dJx", &f_dJx);
   AddEntry("f_dJy", &f_dJy);
   AddEntry("f_dJz", &f_dJz);
   AddEntry("f_dRho", &f_dRho);
   AddEntry("f_OmegaParb", &f_OmegaParb);
   AddEntry("f_Ncgs", &f_Ncgs);
   AddEntry("f_OmegaPcgs", &f_OmegaPcgs);
   AddEntry("f_Weight", &f_Weight);
   AddEntry("f_WeightCGS", &f_WeightCGS);
   AddEntry("l_Np", &l_Np);
   //SCATTERING
    AddEntry("ScatterFlag", &i_ScatterFlag, 0);
  AddEntry("SigmaScattering", &d_Sigma, 1e-16);//Scattering cross-section/KeV in CGS 

  AddEntry("Density", &f_Narb);
  AddEntry("Density1", &f_Narb1);
  AddEntry("Scale", &f_Scale);
  AddEntry("AngleY", &f_AngleY);
  AddEntry("AngleZ", &f_AngleZ);
  AddEntry("Begin", &f_Begin);
  AddEntry("End", &f_End);
  AddEntry("PlateauBegin", &f_PlateauBegin);
  AddEntry("PlateauEnd", &f_PlateauEnd);
  AddEntry("Radius", &f_Radius);
  AddEntry("RadiusX", &f_RadiusX);
  AddEntry("RadiusY", &f_RadiusY);
  AddEntry("RadiusZ", &f_RadiusZ);
  AddEntry("Delta", &f_Delta, 0.);
  AddEntry("Xperiod", &f_Xperiod, 0.);
  AddEntry("Yperiod", &f_Yperiod, 0.);
  AddEntry("Zperiod", &f_Zperiod, 0.);
  AddEntry("Xcurvature", &f_Xcurvature, 0.);
  AddEntry("Ycurvature", &f_Ycurvature, 0.);
  AddEntry("Zcurvature", &f_Zcurvature, 0.);
  AddEntry("CurvatureBegin", &f_CurvatureBegin, 0.);
  AddEntry("CurvatureEnd", &f_CurvatureEnd, 1.e20);

  AddEntry("x0",&f_x0);
  AddEntry("y0",&f_y0);
  AddEntry("z0",&f_z0);
  AddEntry("P_perCell", (int*)&l_perCell);
  AddEntry("Px0", &f_Px0, 0.);
  AddEntry("Py0", &f_Py0, 0.);
  AddEntry("Pz0", &f_Pz0, 0.);
  AddEntry("PspreadX", &f_PspreadX, 0.);
  AddEntry("PspreadY", &f_PspreadY, 0.);
  AddEntry("PspreadZ", &f_PspreadZ, 0.);
  AddEntry("Zombie", &i_Zombie, 0.);
  AddEntry("PhaseSpaceFillFlag", &i_PhaseSpaceFillFlag, 0); 
  AddEntry("Beam", &i_BeamFlag, 0.);
  i_Type = 0;

  if (f) {
    rewind(f);
    read(f);
  }

#ifdef V_MPI
//<SergK>
    d->ResetBufMPP();
//		d->p_BufMPP->Reset();
//</SergK>

    pack_nls(d->GetBufMPP());
    d->BroadCast(d->GetBufMPP());
    if (f==NULL) unpack_nls(d->GetBufMPP());
#endif

  p_CGS = d->GetCGS();

  f_OmegaParb = sqrt(f_Narb);
  l_Np = 0;
  //  cout << "p_CGS = "<< p_CGS<<endl;
  //  cout << "f_qCGS = "<< p_CGS->f_qCGS<<"f_mCGS = "<< p_CGS->f_mCGS<<endl;
  f_qCGS = p_CGS->GetQe();
  f_mCGS = p_CGS->GetMe();
  //  cout << "BasicSpecie:: file read in"<<endl;
  f_q2mCGS = f_qCGS/f_mCGS;

  f_q = f_m = f_q2m = -1.;   // Electron parameters as default

  f_OmegaPcgs = p_CGS->GetOmegaLaser()*f_OmegaParb;
  f_Ncgs = p_CGS->GetCritDensity()*f_Narb;
  if (l_perCell) {
    f_Weight = 1./l_perCell;
    f_WeightCGS = p_CGS->GetCritDensity()*p_CGS->GetHx()*p_CGS->GetHy()*p_CGS->GetHz();
    f_dJx = -1.;
    f_dJy = -1.;
    f_dJz = -1.;
    f_dRho = -1.;
  }
  else { 
    f_Weight = 0.;
    f_WeightCGS = 0.;
    f_dJx = f_dJy = f_dJz = 0.;
  }
  i_Sort = 0;
}
//--------------Specie::GetScatteringCrossSection----------------------
double Specie::GetScatteringCrossSection(double P)
{
   //ToDo:Add functionality for diff. types.
   
   //switch(metal)
   //case Al:

   //case Au:

   //case Cu:

   //case Ti:
   
   return d_Sigma*pow((sqrt(1+P*P)+1)/(512*P*P),1);
}

 

//---------------------------- IonSpecie::IonSpecie -----------------------
IonSpecie::IonSpecie(int isort, char *nm, FILE *f) : Specie(nm,NULL)
{
  Domain *d = domain();
  i_Sort = isort;

  AddEntry("Z", &i_Zmax, 1);
  AddEntry("Type", &i_Type, 1);
  AddEntry("InitialState", &i_State0, 0);
  AddEntry("MassAE", &f_NuclMass, 1.); // Nuclear mass in a.e. 
  AddEntry("Polarity", &f_Polarity, 1.); // Basic charge in electron units
  AddEntry("AtomType", &i_AtomType, 0.);//Atoms for ionization, see below. 

  if (f) { rewind(f); read(f);};

  f_OmegaPcgs = p_CGS->GetOmegaLaser()*f_OmegaParb;
  f_Ncgs = p_CGS->GetCritDensity()*f_Narb;

  if (l_perCell) {
    f_Weight = 1./l_perCell;
    f_WeightCGS = p_CGS->GetCritDensity() * p_CGS->GetHx()*p_CGS->GetHy()*p_CGS->GetHz();
    f_dJx = f_Polarity;
    f_dJy = f_Polarity;
    f_dJz = f_Polarity;
    f_dRho = f_Polarity;
  }
  else { 
    f_Weight = 0.;
    f_WeightCGS = 0.;
    f_dJx = f_dJy = f_dJz = f_dRho = 0.;
  }

#ifdef V_MPI
//<SergK>
  d->ResetBufMPP();
  pack_nls(d->GetBufMPP());
//</SergK>
  d->BroadCast(d->GetBufMPP());
  if (f==NULL) unpack_nls(d->GetBufMPP());
#endif

  //  cout << "p_CGS = "<< p_CGS<<endl;
  //  cout << "f_qCGS = "<< p_CGS->f_qCGS<<"f_mCGS = "<< p_CGS->f_mCGS<<endl;
  f_qCGS = p_CGS->GetQe();
  f_mCGS = p_CGS->GetMp()*f_NuclMass;
  //  cout << "BasicSpecie:: file read in"<<endl;
  f_q2mCGS = f_qCGS/f_mCGS;

  f_q = f_Polarity;
  f_m = f_NuclMass*1836;
  f_q2m = f_q/f_m;

  c_name = nm;

   if (i_Type < 1) return;

//Ionization Potential Values of Species in eV
  cout<<i_AtomType<<endl;
  i_Zmax = i_AtomType;
  if (i_Zmax > 0) {
     Ionization_Levels.resize(i_Zmax);
  } else {
     Ionization_Levels.resize(1);
  }

  switch(i_AtomType)
  {
   case Xenon://Xenon
      Ionization_Levels[0] = 1.197e1;
      Ionization_Levels[1] = 2.354e1;
      Ionization_Levels[2] = 3.511e1;
      Ionization_Levels[3] = 4.668e1;
      Ionization_Levels[4] = 5.969e1;
      Ionization_Levels[5] = 7.183e1;
      Ionization_Levels[6] = 9.805e1;
      Ionization_Levels[7] = 1.123e2;
      Ionization_Levels[8] = 1.708e2;
      Ionization_Levels[9] = 2.017e2;
      Ionization_Levels[10] = 2.326e2;
      Ionization_Levels[11] = 2.635e2;
      Ionization_Levels[12] = 2.944e2;
      Ionization_Levels[13] = 3.253e2;
      Ionization_Levels[14] = 3.583e2;
      Ionization_Levels[15] = 3.896e2;
      Ionization_Levels[16] = 4.209e2;
      Ionization_Levels[17] = 4.522e2;
      Ionization_Levels[18] = 5.725e2;
      Ionization_Levels[19] = 6.077e2;
      Ionization_Levels[20] = 6.429e2;
      Ionization_Levels[21] = 6.781e2;
      Ionization_Levels[22] = 7.260e2;
      Ionization_Levels[23] = 7.627e2;
      Ionization_Levels[24] = 8.527e2;
      Ionization_Levels[25] = 8.906e2;
      Ionization_Levels[26] = 1.394e3;
      Ionization_Levels[27] = 1.491e3;
      Ionization_Levels[28] = 1.587e3;
      Ionization_Levels[29] = 1.684e3;
      Ionization_Levels[30] = 1.781e3;
      Ionization_Levels[31] = 1.877e3;
      Ionization_Levels[32] = 1.987e3;
      Ionization_Levels[33] = 2.085e3;
      Ionization_Levels[34] = 2.183e3;
      Ionization_Levels[35] = 2.281e3;
      Ionization_Levels[36] = 2.548e3;
      Ionization_Levels[37] = 2.637e3;
      Ionization_Levels[38] = 2.726e3;
      Ionization_Levels[39] = 2.814e3;
      Ionization_Levels[40] = 3.001e3;
      Ionization_Levels[41] = 3.093e3;
      Ionization_Levels[42] = 3.296e3;
      Ionization_Levels[43] = 3.386e3;
      Ionization_Levels[44] = 7.224e3;
      Ionization_Levels[45] = 7.491e3;
      Ionization_Levels[46] = 7.758e3;
      Ionization_Levels[47] = 8.024e3;
      Ionization_Levels[48] = 8.617e3;
      Ionization_Levels[49] = 8.899e3;
      Ionization_Levels[50] = 9.330e3;
      Ionization_Levels[51] = 9.569e3;
      Ionization_Levels[52] = 3.925e4;
      Ionization_Levels[53] = 4.027e4;
      //
      i_Zmax = i_AtomType = 54;
      break;
   case Hydrogen://----- Hydrogen
      Ionization_Levels[0] = 13.6;
      i_Zmax = i_AtomType = 1;
      break;
   case Helium://---- Helium
      Ionization_Levels[0] = 24.58;
      Ionization_Levels[1] = 54.4;
      //
      i_Zmax = i_AtomType = 2;
      break;
   case Carbon://---- Carbon
      Ionization_Levels[0] = 11.27;
      Ionization_Levels[1] = 24.41;
      Ionization_Levels[2] = 47.96;
      Ionization_Levels[3] = 64.58;
      Ionization_Levels[4] = 392.6;
      Ionization_Levels[5] = 491;
      //
      i_Zmax = i_AtomType = 6;
      break;
   case Aluminium://--Aluminium
      Ionization_Levels[0] = 5.968;
      Ionization_Levels[1] = 18.796;
      Ionization_Levels[2] = 28.399;
      Ionization_Levels[3] = 119.78;
      Ionization_Levels[4] = 153.561;
      Ionization_Levels[5] = 190.156;
      Ionization_Levels[6] = 241.34;
      Ionization_Levels[7] = 284.163;
      Ionization_Levels[8] = 329.564;
      Ionization_Levels[9] = 398.057;
      Ionization_Levels[10] = 441.232;
      Ionization_Levels[11] = 2082.379;
      Ionization_Levels[12] = 2300.16;
      //
      i_Zmax = i_AtomType = 13;
      break;
      
   case Silicon:
      Ionization_Levels[0] = 7.264e0;
      Ionization_Levels[1] = 1.695e1;
      Ionization_Levels[2] = 3.427e1;
      Ionization_Levels[3] = 4.665e1;
      Ionization_Levels[4] = 1.598e2;
      Ionization_Levels[5] = 2.105e2;
      Ionization_Levels[5] = 2.613e2;
      Ionization_Levels[7] = 3.120e2;
      Ionization_Levels[8] = 3.640e2;
      Ionization_Levels[9] = 4.151e2;
      Ionization_Levels[10] = 5.037e2;
      Ionization_Levels[11] = 5.522e2;
      Ionization_Levels[12] = 2.324e3;
      Ionization_Levels[13] = 2.569e3;
      //
      i_Zmax = i_AtomType = 14;
      break;

   case Copper:
      Ionization_Levels[0] = 7.70;
      Ionization_Levels[1] = 20.234;
      Ionization_Levels[2] = 36.739;
      Ionization_Levels[3] = 57.213;
      Ionization_Levels[4] = 79.577;
      Ionization_Levels[5] = 102.313;
      Ionization_Levels[5] = 138.48;
      Ionization_Levels[7] = 165.354;
      Ionization_Levels[8] = 198.425;
      Ionization_Levels[9] = 231.492;
      Ionization_Levels[10] = 264.566;
      Ionization_Levels[11] = 367.913;
      Ionization_Levels[12] = 399.951;
      Ionization_Levels[13] = 434.055;
      Ionization_Levels[14] = 482.627;
      Ionization_Levels[15] = 518.798;
      Ionization_Levels[16] = 554.970;
      Ionization_Levels[17] = 631.446;
      Ionization_Levels[18] = 668.672;
      Ionization_Levels[19] = 1691.78;
      Ionization_Levels[20] = 1799.261;
      Ionization_Levels[21] = 1916.0;
      Ionization_Levels[22] = 2060.0;
      Ionization_Levels[23] = 2182.0;
      Ionization_Levels[24] = 2308.0;
      Ionization_Levels[25] = 2478.0;
      Ionization_Levels[26] = 2587.5;
      Ionization_Levels[27] = 11062.38;
      Ionization_Levels[28] = 11567.617;
      //no of levels
      i_Zmax = i_AtomType = 29;
      break;
   case 0:
   default://---Pre-ionized
      i_Zmax = 1;
      i_AtomType = 0;
      Ionization_Levels[0] = 0.;
      i_State0 = 1;
      //
   }
   if (i_State0 > i_Zmax && i_Zmax > 0) {
      i_State0 = i_Zmax;
   };
   if (i_State0 < 0) {
      i_State0 = 0;
   };

   d->Getout_Flog()<<"AtomType="<<i_AtomType<<" State0="<<i_State0<<"\n";
   for (int i=0; i<i_AtomType; i++) {
      d->Getout_Flog()<<"Ionization_Levels["<< i <<"]="<<Ionization_Levels[i]<<"\n";
   };
   //--------------------------------------------------------->
   //End of introduction of Ionization Levels
   //--------------------------------------------------------->
}

//--------IonSpecie::GetIonizationLayer-------------------------->

double IonSpecie::GetIonizationLevel(int charge)
{

   Ionization_Levels.resize(i_Zmax);
   assert(charge >=0);
   if (charge < i_Zmax) return Ionization_Levels[charge];
   return -1.;
}



//---------IonSpecie::GetIonizationProbability--------------------
double IonSpecie::GetIonizationProbability(double E, int charge)
{
   if (E==0.) return 0;//check for field.

   Cell *c=NULL;
   E = c->ConvertFieldNum2Rel(E);
   E = fabs(E); //--[ Field > 0 now ]

   double IonizationLayer = GetIonizationLevel(charge);

   //----- Check for degenerate cases
   if(IonizationLayer == 0) return 1.;
   else if (IonizationLayer < 0.) return 0.;
   
   //critical Electric field for Ionization 'Ecrit'
   //Normalization parmeters
   double tsNum = domain()->GetTs(); //numerical Time Step
   double rElecCGS = p_CGS->Getre(); // Classical Electron radius.
   double WavelengthCGS = p_CGS->GetWavelength();//wavelength
   double QeCGS = - p_CGS->GetQe();//Elctron charge
   double IP_CGS = IonizationLayer*1.6022e-12;//I.Pot in CGS

   //---critical elecrtic field for BSI/ADK
//   double crit_E_field = (tsNum*IP_CGS*IP_CGS*rElecCGS*WavelengthCGS)/(8*pow(QeCGS,4));
   double crit_E_field = (IonizationLayer*IonizationLayer/5.12e5/5.12e5
      *WavelengthCGS/rElecCGS)/(8.*PI);
   
   if (E >= 2.5*crit_E_field) {
      return 1.;//BSI occurs
   }
   double probability = GetADKprob(E,charge);
//   probability = 0.;

   long totalNe = domain()->GetSpecie(0)->GetNp();
//   cout << "ionize We have " << totalNe << " electrons, E=" << E << " probability="<<probability<<endl;
   return probability;//tunneling occurs
}
