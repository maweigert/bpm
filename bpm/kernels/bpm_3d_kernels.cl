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
