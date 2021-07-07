Molecular Descriptors For NN-DMC
=========================================================

``PyVibDMC`` supports the use of GPU-accelerated neural network models as potential energy surfaces. However, in order to
predict energies in a neural network, one needs to transform the walker coordinates from Cartesian coordinates to
a vector of values that are (at least) invariant with respect to translations and rotations. The McCoy group currently
uses different types of atom-atom distance descriptors for our neural network models, and so these are packed away in ``PyVibDMC``
for fast, on-the-fly transformation when DMC simulations are running.

Using the Molecular Descriptors
--------------------------------

The descriptors are written so that they can take advantage of ``NumPy`` (CPU) or ``CuPy`` (GPU) acceleration. If ``CuPy``
is not installed, then the descriptor automatically defaults to use ``NumPy``. One can also force the code to use ``NumPy``
even if ``CuPy`` is installed. The three descriptors that are currently implemented are
the Coulomb Matrix, the Distance Matrix, and what we call the SPF (Simons–Parr–Finlan) matrix, whose matrix elements are
``R_{ij}-R_{ij,eq} / R_{ij}`` instead of simply using ``R_{ij}`` for the distance matrix. For the water trimer::

    from pyvibdmc import DistIt
    #zs here is the nuclear charges of each of the atoms
    #transformer_dist = DistIt(zs=[8, 1, 1] * 3,
    #                     method='distance',
    #                     full_mat=True) # False is the default, which returns the upper triangle (no diagonal included)
    #transformer_coul = DistIt(zs=[8, 1, 1] * 3,
    #                     method='coulomb',
    #                     full_mat=False)
    transformer_spf = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=equilibrium_coordinate, # required for spf descriptor
                         full_mat=False, # False is the default here
                         )
    coords = ... # (n x m x 3) numpy or cupy array, it'll convert internall to the appropriate one
    transformer_spf.run(coords) # cupy or numpy array depending on what is being used

Sorting these matrices by the norm of their columns makes them permutationally invariant. However, this leads to
discontinuities in the potential that employs them. For DMC, these discontinuities do not seem to be an issue for the
overall results of the DMC simulation (see `this paper <https://doi.org/10.1021/acs.jpca.1c03709>`_). For
the SPF matrix, the distance matrix is sorted and then the sorted equilibrium matrix is subtracted from it.

There are many ways to sort the descriptor matrix.  The standard and most straightforward is to sort everything based on
the norm of each column. To do this, one would do::

    transformer = DistIt(zs=[8, 1, 1] * 3,
                         method='spf',
                         eq_xyz=cds[0],
                         full_mat=False,
                         sorted_atoms=[[0, 1, 2, 3, 4, 5, 6, 7, 8]] # note, this must be a list of lists
                         )

NOTE: ``sorted_atoms`` must contain ALL of the atoms in the molecule.

To sort, for example, only the two Hs in each water molecule in the trimer, one can use ``sorted_atoms`` in a different way::

    transformer = DistIt(zs=[8, 1, 1] * 3,
                     method='spf',
                     eq_xyz=cds[0],
                     full_mat=False,
                     sorted_atoms=[[0], [1, 2], [3], [4, 5], [6], [7, 8]] #  note, this must be a list of lists
                     )

This will leave indices 0, 3, and 6 completely alone (the code says they will be sorted only with themselves) and
only 1 can swap with 2 and only 4 can swap with 5.

Another way to sort is by the norm of multiple columns of the matrix. For example, we can sort by each water molecule
in the water trimer by using the ``sorted_groups`` argument::

    def test_coulomb():
        """Sorted/Unsorted distance matrix descriptor"""
        from pyvibdmc import DistIt
        transformer = DistIt(zs=[8, 1, 1] * 3,
                             method='coulomb',
                             full_mat=True, #Get the whole matrix back rather than the upper triangle
                             sorted_groups=[[0, 1, 2], [6, 7, 8]] # note, this must be a list of lists
                             )

NOTE: for ``sorted_groups``, you do NOT have to pass in all the atoms. Here, we are only sorting the first two water molecules
of the three molecules in the trimer (no good reason to do this, just for illustrative purposes).

One can combine both methods by passing in both ``sorted_atoms`` and ``sorted_groups``, which sorts first by atom then
by group. This should only be done if you are only sorting a subset of the atoms before sorting the groups, much like how
one would sort the two Hs on each water molecule, then the water molecules themselves.
