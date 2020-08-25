import matplotlib.pyplot as plt
from ..simulation_utilities import *

class SimInfo():
    def __init__(self,h5Name):
        self.fname = h5Name    
        self.loadH5()
        self._mplIinit
    
    def _mplIinit(self):
        params = {'text.usetex': False,
        'mathtext.fontset': 'dejavusans',
        'font.size':14}
        plt.rcParams.update(params) 

    def loadH5(self):
        with h5py.File(self.fname,'r') as f:
            print(list(f.keys()))
            self.vrefVsTau = f['vrefVsTau'][:]
            self.popVsTau  = f['popVsTau'][:]
        
    def plt_vref_vs_tau(self,photoName="vref_vs_tau.png"):
        plt.plot(self.vrefVsTau[:,0],self.vrefVsTau[:,1],'k')
        plt.xlabel("Time step")
        plt.ylabel(r"Vref ($\mathrm{cm^{-1}}$)")
        plt.savefig(photoName,dpi=300)

    def plt_pop_vs_tau(self,photoName='pop_vs_tau.png'):
        plt.plot(self.popVsTau[:,0],self.popVsTau[:,1],'k')
        plt.xlabel("Time step")
        plt.ylabel(r"Population or Sum of Weights")
        plt.savefig(photoName,dpi=300)
        
    def avg_energy(self,onwards=1000):
        """onwards is an int that tells us where to start averaging (python indexing
        starts at 0)"""
        return np.average(self.vrefVsTau[onwards:,1])

class waveFunction():
    def __init__(self,wfnName):
        self.fname = fileName


