#include<string>
using namespace std;

string getLabel(int n)
{
	string labels[115];

	labels[0] = "j";
	labels[1] = "k";
	labels[2] = "weight";
	labels[3] = "xp";
	labels[4] = "yp";
	labels[5] = "zp";
	labels[6] = "px";
	labels[7] = "py";
	labels[8] = "pz";
	labels[11] = "gammap";
	labels[9] = "Vxp";
	labels[10] = "Vyp";
	labels[11] = "Vzp";
	labels[12] = "y_est";
	labels[13] = "z_est";
	labels[14] = "y_est";
	labels[15] = "z_est";
	labels[16] = "ym";
	labels[17] = "zm";
	labels[18] = "exp";
	labels[19] = "eyp";
	labels[20] = "ezp";
	labels[21] = "bxp";
// 	labels[22] = "accc";
// 	labels[23] = "appc";
// 	labels[24] = "apcp";
	labels[25] = "appp";
	labels[26] = "pl->Bx[npc]";
	labels[27] = "pl->Bx[npp]";
	labels[28] = "pl->Bx[ncp]";
	labels[29] = "pl->Bx[npp]";
	labels[30] = "byp";
	labels[31] = "bzp";
	labels[32] = "exm";
	labels[33] = "eym";
	labels[34] = "ezm";
	labels[35] = "bxm";
	labels[36] = "bym";
	labels[37] = "bzm";
	labels[38] = "ex";
	labels[39] = "ey";
	labels[40] = "ez";
	labels[22] = "bx";   // 22 - 24 doubled
	labels[23] = "by";
	labels[24] = "bz";
	labels[41] = "ex";
	labels[42] = "ey";
	labels[43] = "ez";
	labels[44] = "px";
	labels[45] = "py";
	labels[46] = "pz";
	labels[47] = "p3x";
	labels[48] = "p3y";
	labels[49] = "p3z";
	labels[50] = "px_new";
	labels[51] = "py_new";
	labels[52] = "pz_new";
	labels[53] = "px";
	labels[54] = "py";
	labels[55] = "pz";
	labels[56] = "Vx";
	labels[57] = "Vy";
	labels[58] = "Vz";
	labels[59] = "djx";
	labels[60] = "djy";
	labels[61] = "djz";
	labels[62] = "drho";
	labels[63] = "djx";
	labels[64] = "djy";
	labels[65] = "djz";
	labels[66] = "drho";
	labels[67] = "dy";
	labels[68] = "dz";
	labels[69] = "dy";
	labels[70] = "dz";
	labels[71] = "j_jump";
	labels[72] = "k_jump";
	labels[73] = "ytmp";
	labels[74] = "ztmp";
	labels[75] = "step";
	labels[76] = "partdx";
	labels[77] = "partdx";
	labels[78] = "part_step";
	labels[79] = "ytmp";
	labels[80] = "ztmp";
	labels[81] = "ytmp";
	labels[82] = "ztmp";
	labels[83] = "ytmp";
	labels[84] = "ztmp";
	labels[85] = "j_jump";
	labels[86] = "k_jump";
	labels[94] = "ytmp";
	labels[95] = "ztmp";
	labels[96] = "ytmp";
	labels[97] = "ztmp";
	labels[92] = "ytmp";
	labels[93] = "ztmp";
	labels[99] = "(double";
	labels[87] = "djx";
	labels[88] = "djy";
	labels[89] = "djz";
	labels[90] = "drho";
	labels[91] = "l_dMz + 2*l_dMy*ktmp";  // hereon
	//numbering is wrong
	labels[111] = "cl->Jx[ncc]";
	labels[112] = "cl->Jy[ncc]";
	labels[113] = "cl->Jz[ncc]";
	labels[114] = "cl->Rho[ncc]";

	if(n < 115)
	{
		return labels[n];
	}
	return "";

}
