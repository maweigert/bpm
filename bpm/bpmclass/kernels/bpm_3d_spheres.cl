#include <pyopencl-complex.h>


__kernel void fill_inds(__global float* points,const int Np,
						const float unit,
						const float rad,
						__global int * ind1Arr,__global int * ind2Arr){

  
  int i = get_global_id(0);

  float z = unit*i;

  int ind1 = 0;
  int ind2 = Np-1;
  

  while (points[3*ind1+2]<z-rad &&ind1<Np-1)
	ind1++;

  while (points[3*ind2+2]>z+rad &&ind2>0)
	ind2--;

  ind1Arr[i] = ind1;
  ind2Arr[i] = ind2;

}



__kernel void fill_dn(__global float* dn_buf,__constant float* points,const int Np,
						const float dx,const float dy,const float dz,
					  const float dn1,const float rad1,
					  const float dn2,const float rad2){

  
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float4 r = (float4)(dx*i,dy*j,dz*k,0.f);

  int ind1 = 0;
  int ind2 = Np-1;

  float dn = 0.f;
  
  while (points[3*ind1+2]<r.z-rad2 &&ind1<Np-1)
	ind1++;

  while (points[3*ind2+2]>r.z+rad2 &&ind2>0)
	ind2--;



  float dmin = 1000000000.f*rad2;
  
  for (int n = ind1; n <= ind2; n++) {
    float4 r2 = (float4)(points[3*n],points[3*n+1],points[3*n+2],0.f);

	float d = length(r-r2);
	dmin = (d<dmin)?d:dmin;
	
  }

  dn = (dmin<rad2)?(dmin<rad1)?dn1:dn2:0.f;
  
  dn_buf[i+Nx*j+Nx*Ny*k] = dn;

}
