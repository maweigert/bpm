    #include <pyopencl-complex.h>
    
__kernel void mult(__global cfloat_t* a,
				   __global cfloat_t* b){

  uint i = get_global_id(0);
    
  a[i] = cfloat_mul(a[i], b[i]);

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

  float dnDiff = unit_k*dn_val;

  dnDiff = unit_k*(dn_val+.5f*dn_val*dn_val);
  
  cfloat_t dPhase = (cfloat_t)(cos(dnDiff),sin(dnDiff));

  input[i+Nx2*j] = cfloat_mul(input[i+Nx2*j],dPhase);


  // input[i+Nx2*j] = (cfloat_t)(dn_val,0.);
  
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
  
  cfloat_t dnDiff = cfloat_mul((cfloat_t)(0,unit_k),(cfloat_t)(dn_val.x,dn_val.y));
  cfloat_t dPhase = cfloat_exp(dnDiff);
  
  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

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
