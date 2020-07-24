import numpy as np
import h5py
#lightweight class to do some initial analysis of a DMC wave function.  Use another package to do more in depth analysis
#of the wave function.
#Must remake for HDF5
#To put in class:
#Vref vs Tau
#average ZPE
#OH bond lengths
#Weights over time
#descendant weight histogram
#This is to just give the user a feel of how the simulation went
class analyzeDMCSim:
    def __init__(self,simName,energies=None):
        self.dmcZ = np.load(simName)
        self.energies=energies
    def get_vref(self):
        """extract vref vs tau from 0 -> the time the wave function was collected. This is mainly for if the sim died.
          Otherwise, just load in energies.npy"""
        return self.dmcZ['vref']

    def get_cds(self):
        return self.dmcZ['cds']

    def get_dw(self):
        return self.dmcZ['weights']

    def get_atms(self):
        """The atom order from the simulation"""
        return self.dmcZ['atms']

    def pltVref(self,savefigN):
        """A look at vref vs tau"""
        mplHelper.initializeMpl()
        if self.energies is None:
            plt.plot(self.get_vref(),'k',linewidth=2)
        else:
            plt.plot(self.energies,'k',linewidth=2)
        mplHelper.mySave(savefigN)

    def avgVref(self,begin=None,end=None):
        """The way to calculate ZPE. Sometimes one wants to calculate ZPE over a window, or from the start of a sliding
        point.  this allows us to do it."""
