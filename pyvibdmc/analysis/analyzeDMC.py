import matplotlib.pyplot as plt
from ..simulation_utilities import *

class SimInfo():
    def __init__(self,h5Name):
        self.fname = h5Name    
        self._loadH5()
        self._mplIinit
    
    def _mplIinit(self):
        params = {'text.usetex': False,
        'mathtext.fontset': 'dejavusans',
        'font.size':14}
        plt.rcParams.update(params) 
        
    def _load_sim_H5(self):
        with h5py.File(self.fname,'r') as f:
            self.vrefVsTau = f['vrefVsTau'][:]
            self.popVsTau  = f['popVsTau'][:]

    @staticmethod
    def get_wfn(wfn_fl):
        with h5py.File(wfn_fl,'r') as f:
            cds = f['coords'][:]
            cds = Constants.convert(cds,'angstroms',to_AU=False)
            dw = f['desc_weights'][:]
        return cds,dw
        
    def get_vref(self):
        return self.vrefVsTau

    def get_pop(self):
        return self.popVsTau

    def plt_vref_vs_tau(self,photoName="vref_vs_tau.png"):
        plt.plot(self.vrefVsTau[:,0],self.vrefVsTau[:,1],'k')
        plt.xlabel("Time step")
        plt.ylabel(r"Vref ($\mathrm{cm^{-1}}$)")
        plt.savefig(photoName,dpi=300)
        plt.close()

    def plt_pop_vs_tau(self,photoName='pop_vs_tau.png'):
        plt.plot(self.popVsTau[:,0],self.popVsTau[:,1],'k')
        plt.xlabel("Time step")
        plt.ylabel(r"Population or Sum of Weights")
        plt.savefig(photoName,dpi=300)
        plt.close()
        
    def avg_energy(self,onwards=1000):
        """onwards is an int that tells us where to start averaging (python indexing
        starts at 0)"""
        return np.average(self.vrefVsTau[onwards:,1])

class waveFunction():
    def __init__(self,wfnName):
        self.fname = fileName


