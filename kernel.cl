/**
 * This kernel function sums two arrays of integers and returns its result
 * through a third array.
 **/

 __kernel void vadd(float a, __global float* x, __global float* y, __global float* c){
     int index = get_global_id(0);
     c[index] = a * x[index] + y[index] * x[index];
 }