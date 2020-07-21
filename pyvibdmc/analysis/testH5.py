import h5py
import matplotlib.pyplot as plt
with h5py.File('../simulation_results/DMC_disc_test_simInfo.hdf5','r') as f:
    print(list(f.keys()))
    dset = f['vrefVsTau'][:]
    print(dset.shape)
    plt.plot(dset[:,0],dset[:,1])
    plt.show()