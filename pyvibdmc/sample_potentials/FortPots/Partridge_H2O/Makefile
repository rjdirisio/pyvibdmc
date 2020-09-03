OPT = -O2

main : 
	gfortran -c -fPIC h2opes_v2.f
	gfortran -c -fPIC calc_h2o_pot.f
	python3 -m numpy.f2py -c -I. h2opes_v2.o -m h2o_pot calc_h2o_pot.f

clean :
	rm *.o *.so
