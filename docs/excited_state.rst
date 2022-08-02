Excited States using Fixed Node DMC
=========================================================

PyVibDMC supports the usage of Fixed Node DMC to obtain the first excited state calculation along one coordinate. The coordinate and the Wilson's G-matrix diagonal element must be supplied by users.

Using the Code
--------------------------------

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
    


