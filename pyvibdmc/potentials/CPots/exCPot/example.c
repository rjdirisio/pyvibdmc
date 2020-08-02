/* File : example.c */

 #include <time.h>
 #include <math.h>
double My_variable = 3.0;

int fact(int n) {
  if (n <= 1) return 1;
  else return n*fact(n-1);
}

int my_mod(int x, int y) {
  return (x%y);
}

double harmonic_oscillator(float x) {
    double omega = 3600 / 219474.63;
    double mass = 1.00782503 * 1.0/6.0221367E23/9.1093897E-28;
    return 0.5*mass*pow(omega,2)*pow(x,2);
}

char *get_time()
{
  time_t ltime;
  time(&ltime);
  return ctime(&ltime);
}
 
