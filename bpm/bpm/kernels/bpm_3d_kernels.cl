 #include <pyopencl-complex.h>

  #define M_PI 3.14159265358979f


__kernel void mult(__global cfloat_t* a,
				   __global cfloat_t* b){

  uint i = get_global_id(0);
    
  a[i] = cfloat_mul(a[i], b[i]);

}

__kernel void mult_dn(__global cfloat_t* input,
					  __global float* dn,const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  float dnDiff = -unit_k*dn[i+Nx*j+stride];

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5*(1-cos(M_PI*dist/absorb)):1.;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);






  res = cfloat_mul(res,cfloat_new(absorb_val,0.));

  //res = (cfloat_t)(absorb_val,0.);
  //res = (cfloat_t)(distx,disty);

  //if (i+j==0)
  //  printf("%d\n",absorb);


  //if (absorb_val<.5)
  //  printf("absorb %d %d: %.2f %.2f \n",i,j,res.x,res.y);

  input[i+Nx*j] = res;

}


__kernel void mult_dn_complex(__global cfloat_t* input,
					  __global cfloat_t* dn,
					  const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0,-unit_k),dn[i+Nx*j+stride]);

  cfloat_t dPhase = cfloat_exp(dnDiff);


  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

}
__kernel void mult_dn_image(__global cfloat_t* input,
							__read_only image3d_t dn,
							const float unit_k,
							const float n0,
							const int zpos,
							const int subsample){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx2 = get_global_size(0);

  float dn_val = read_imagef(dn, sampler, (float4)(1.f*i/subsample,1.f*j/subsample,1.f*zpos/subsample,0)).x;

  float dnDiff = -unit_k*dn_val;

  // dnDiff = -unit_k*dn_val*(1.f+.5f*dn_val/n0);
  
  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  input[i+Nx2*j] = cfloat_mul(input[i+Nx2*j],dPhase);


  // input[i+Nx2*j] = cfloat_new(dn_val,0.);
  
}




__kernel void mult_dn_complex_image(__global cfloat_t* input,
									__read_only image3d_t dn,
									const float unit_k,
									const float n0,

									const int zpos,
									const int subsample){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);

  float2 dn_val = read_imagef(dn, sampler, (float4)(1.f*i/subsample,1.f*j/subsample,1.f*zpos/subsample,0)).xy;
  
  cfloat_t dnDiff = cfloat_mul(cfloat_new(0,-unit_k),cfloat_new(dn_val.x,dn_val.y));
  cfloat_t dPhase = cfloat_exp(dnDiff);
  
  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

  //if ((i==64) &&(j==64))
  //  printf("kernel %.10f \n",dn_val.y);

  //input[i+Nx*j] = cfloat_new(1.f,0.f);

}


#define M_PI 3.14159265358979323846

__kernel void divide_dn_complex(__global cfloat_t* plane0,__global cfloat_t* plane1,
					  __global cfloat_t* dn,const float unit_k, const int stride){

  uint i = get_global_id(0);
  
  cfloat_t phase;

  float dn_val;
  // res = cfloat_divide(plane2[i],plane1[i]);

  phase = cfloat_divide(plane0[i],plane1[i]);

  dn_val = atan2(phase.y,phase.x);

  dn_val *= 1./unit_k;

  // dn_val = clamp(dn_val,0.f,4.f);
  cfloat_t res = cfloat_new(dn_val,0.);
  
  dn[i+stride] = res;

  
}


__kernel void copy_subsampled_buffer(__global cfloat_t* buffer,__global cfloat_t* plane,
									 const int subsample,
									 const int stride){

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  uint Nx = get_global_size(0);

  buffer[i+Nx*j+stride] = plane[i*subsample+subsample*subsample*Nx*j];  
}



__kernel void copy_complex(__global cfloat_t* input,__global cfloat_t* plane,
					  const int stride){

  uint i = get_global_id(0);
  plane[i] = input[i+stride];  
}
