#include <math.h>

/*  Compute the cosine of each element in in_array, storing the result in
 *  out_array. */
void harmOsc(double * in_array, double * out_array, int size) {
    double omega = 3600 / 219474.63;
    double mass = 1.00782503 * 1.0/6.0221367E23/9.1093897E-28;
    int i;
    for(i=0;i<size;i++){
        out_array[i] = 0.5*mass*pow(omega,2)*pow(in_array[i],2);
    }
}