    #include <pyopencl-complex.h>
    
__kernel void mult(__global cfloat_t* a,
				   __global cfloat_t* b){

  uint i = get_global_id(0);
    
  a[i] = cfloat_mul(a[i], b[i]);

}

__kernel void mult_dn(__global cfloat_t* input,
					  __global float* dn,const float unit_k, const int stride){

  uint i = get_global_id(0);
  float dnDiff = unit_k*dn[i+stride];
  cfloat_t dPhase = (cfloat_t)(cos(dnDiff),sin(dnDiff));

  input[i] = cfloat_mul(input[i],dPhase);

}

__kernel void mult_dn_complex(__global cfloat_t* input,
					  __global cfloat_t* dn,const float unit_k, const int stride){

  uint i = get_global_id(0);
  cfloat_t dnDiff = cfloat_mul((cfloat_t)(0,unit_k),dn[i+stride]);

  cfloat_t dPhase = cfloat_exp(dnDiff);

  
  input[i] = cfloat_mul(input[i],dPhase);

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
  cfloat_t res = (cfloat_t)(dn_val,0.);
  
  dn[i+stride] = res;

  
}

__kernel void copy_complex(__global cfloat_t* input,__global cfloat_t* plane,
					  const int stride){

  uint i = get_global_id(0);
  plane[i] = input[i+stride];  
}
