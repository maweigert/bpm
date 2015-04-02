#include <pyopencl-complex.h>
#include <bessel.cl>


#ifndef INT_STEPS
#define INT_STEPS 100
#endif

__kernel void debye_wolf(__global cfloat_t * Ex,
						 __global cfloat_t * Ey,
						 __global cfloat_t * Ez,
						 __global float * I,
						 const float Ex0,
						 const float Ey0,
						 const float x1,const float x2,
						 const float y1,const float y2,
						 const float z1,const float z2,
						 const float lam,
						 __constant float* alphas, const int Nalphas){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);
  float z = z1+k*(z2-z1)/(Nz-1.f);

  float kr = 2.f*M_PI/lam*sqrt(x*x+y*y);
  float kz = 2.f*M_PI/lam*z;
  
  float phi = atan2(y,x); 
  
  cfloat_t I0 = (cfloat_t)(0.f,0.f);
  cfloat_t I1 = (cfloat_t)(0.f,0.f);
  cfloat_t I2 = (cfloat_t)(0.f,0.f);


  // simple traziodal rule

  for (int i_n = 0; i_n < Nalphas/2; i_n++) {
    float alpha1 = alphas[2*i_n];
	float alpha2 = alphas[2*i_n+1];
	
	float dt = (alpha2-alpha1)/(INT_STEPS-1.f);

	for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	  float t = alpha1 + i_t *dt;
	  float co = cos(t);
	  float si = sin(t);
	  cfloat_t phase = (cfloat_t)(cos(kz*co),sin(kz*co));

	  float prefac = ((t==alpha1)||(t==alpha2))?.5f:1.f;

	  prefac *= dt*sqrt(co)*si;
	
	  I0 += prefac*(co+1.f)*bessel_jn(0,kr*si)*phase;
	  I1 += prefac*si*bessel_jn(1,kr*si)*phase;
	  I2 += prefac*(co-1.f)*bessel_jn(2,kr*si)*phase;

	}
  }

  cfloat_t ex = Ex0*(I0+I2*cos(2.f*phi))+Ey0*I2*sin(2.f*phi);
  cfloat_t ey = Ey0*(I0-I2*cos(2.f*phi))+Ex0*I2*sin(2.f*phi);
  cfloat_t ez = cfloat_mul((cfloat_t)(0.f,-2.f),I1)*(Ex0*cos(phi)+Ey0*sin(phi));

  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);

  Ex[i+j*Nx+k*Nx*Ny] = ex;
  Ey[i+j*Nx+k*Nx*Ny] = ey;
  Ez[i+j*Nx+k*Nx*Ny] = ez;

  I[i+j*Nx+k*Nx*Ny] = vx*vx+vy*vy+vz*vz;
}
 

__kernel void debye_wolf_slit(__global cfloat_t * Ex,
						 __global cfloat_t * Ey,
						 __global cfloat_t * Ez,
						 __global float * I,
						 const float Ex0,
						 const float Ey0,
						 const float x1,const float x2,
						 const float y1,const float y2,
						 const float z1,const float z2,
						 const float lam,
							  __constant float* alphas, const int Nalphas,
							  __constant float* slit_x, __constant float* slit_sigma,
							  const int Nslit_x
							  ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);
  float z = z1+k*(z2-z1)/(Nz-1.f);

  float kr = 2.f*M_PI/lam*sqrt(x*x+y*y);
  float kz = 2.f*M_PI/lam*z;
  
  float phi = atan2(y,x); 
  
  cfloat_t I0 = (cfloat_t)(0.f,0.f);
  cfloat_t I1 = (cfloat_t)(0.f,0.f);
  cfloat_t I2 = (cfloat_t)(0.f,0.f);


  // simple traziodal rule

  for (int i_n = 0; i_n < Nalphas/2; i_n++) {
    float alpha1 = alphas[2*i_n];
	float alpha2 = alphas[2*i_n+1];
	
	float dt = (alpha2-alpha1)/(INT_STEPS-1.f);

	for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	  float t = alpha1 + i_t *dt;
	  float co = cos(t);
	  float si = sin(t);
	  cfloat_t phase = (cfloat_t)(cos(kz*co),sin(kz*co));

	  float prefac = ((t==alpha1)||(t==alpha2))?.5f:1.f;

	  prefac *= dt*sqrt(co)*si;
	
	  I0 += prefac*(co+1.f)*bessel_jn(0,kr*si)*phase;
	  I1 += prefac*si*bessel_jn(1,kr*si)*phase;
	  I2 += prefac*(co-1.f)*bessel_jn(2,kr*si)*phase;

	}
  }

  cfloat_t ex = Ex0*(I0+I2*cos(2.f*phi))+Ey0*I2*sin(2.f*phi);
  cfloat_t ey = Ey0*(I0-I2*cos(2.f*phi))+Ex0*I2*sin(2.f*phi);
  cfloat_t ez = cfloat_mul((cfloat_t)(0.f,-2.f),I1)*(Ex0*cos(phi)+Ey0*sin(phi));

  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);

  Ex[i+j*Nx+k*Nx*Ny] = ex;
  Ey[i+j*Nx+k*Nx*Ny] = ey;
  Ez[i+j*Nx+k*Nx*Ny] = ez;

  I[i+j*Nx+k*Nx*Ny] = vx*vx+vy*vy+vz*vz;
}
