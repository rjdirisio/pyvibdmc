Excited States Calculation using DMC
=========================================================

While DMC is a ground state method, we can use some methods to obtain excited state information. Two such methods that are implemented in PyVibDMC are Fixed-node DMC Calculation and Importance Sampling for Excited State

Fixed-node DMC
--------------------------------

For theory of Fixed-node method, please refer to `McCoy group's tutorial on Fixed-node method <https://mccoygroup.github.io/References/References/Spectrum%20Generation/FixedNodeSpectra.html>`_.

PyVibDMC supports the usage of Fixed Node DMC to obtain the first excited state calculation along one coordinate. The coordinate and the Wilson's G-matrix diagonal element must be supplied by users.

There are two components needed in addition to the ground state DMC calculation. One is the coordinate function and the other is the Wilson's G-matrix diagonal element. The coordinate function takes the walkers' coordinates as an argument and return the signed distance (in AU) from the nodal surface. The coordinate function should return positive distance if the walker is in the the preferred side of the nodal surface and vice versa. The other component needed is the Wilson's G-matrix diagonal element which takes the walkers coordinate as an input::

    import pyvibdmc as pv
    pot_func = ...
    py_file = ...
    pot_dir = ...
    
    #Coordinate function
    def dist(cds,kwargs):
        return ...
    
    #G-matrix diagonal element function
    def g_mat(cds):
        return ...
    
    #Pass the coordinate function and g-matrix function
    my_sim = pv.DMC_Sim(...
                        fixed_node = {"function":dist,"g_matrix":g_mat}
                        ...)
    my_sim.run()
    
The Wilson's G-matrix diagonal element can be taken as a number as well::

    #Changing the line in the DMC_Sim argument to
    fixed_node = {"function":dist,"g_matrix":0.5}
    
Importance Sampling for Excited State
---------------------------------------

Another way of obtaining excited state information is to use importance sampling using excited state trial wave function. The problem is that the first and second derivative of the trial wave function diverged at the node. The potential fixed is proposed in a peper by `Umrigar, Nightingale and Runge <https://doi.org/10.1063/1.465195>`_ where the first and second derivative of the trial wave function is capped at some value.

PyVibDMC also supports the implementation of the derivative capped which can be done by toggleing the flag ``excited_state_imp_samp = True`` in ``DMC_Sim(...)`` function.
