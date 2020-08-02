"""
pyvibdmc.py
A general purpose diffusion monte carlo code for studying vibrational problems

Handles the primary functions
"""
import numpy as np
import os

from .data import *
from .simulation_utilities import *
from .potentials import *


class DMC_Sim:
    def __init__(self,
                 simName="DMC_Sim",
                 outputFolder="pyvibdmc/exSimResults",
                 weighting='discrete',
                 initialWalkers=10000,
                 nTimeSteps=10000,
                 equilTime=2000,
                 ckptSpacing=1000,
                 wfnSpacing=1000,
                 DwSteps=50,
                 atoms=[],
                 dimensions=3,
                 deltaT=5,
                 potential=None,
                 masses=None,
                 startStructure=None,
                 branch_every=1,
                 timeStep=0
                 ):
        """
        :param simName:Simulation name for saving wavefunctions
        :type simName:str
        :param outputFolder:The folder where the results will be stored, including wavefunctions and energies
        :type outputFolder:str
        :param weighting:Discrete or Continuous weighting DMC.  Continuous means that there are fixed number of walkers
        :type weighting:str
        :param initialWalkers:Number of walkers we will start the simulation with
        :type initialWalkers:int
        :param nTimeSteps:Total time steps we will be propStepagating the walkers.  nTimeSteps*deltaT = total time in A.U.
        :type nTimeSteps:int
        :param equilTime: Time before we start collecting wavefunctions
        :type equilTime:int
        :param ckptSpacing:How many time steps in between we will propagate before collecting another wavefunction
        every time we collect a wavefunction, we checkpoint the simulation
        :type ckptSpacing:int
        :param DwSteps:Number of time steps for descendant weighting.
        :type DwSteps: int
        :param atoms:List of atoms for the simulation
        :type atoms:list
        :param dimensions: 3 leads to a 3N dimensional simulation. This should always be 3 for real systems.
        :type dimensions:int
        :param deltaT: The length of the time step; how many atomic units of time are you going in one time step.
        :type deltaT: int
        :param D: Diffusion Coefficient.  Usually set at 0.5
        :type D:float
        :param potential: Takes in coordinates, gives back energies
        :type potential: function
        :param masses:For feeding in artificial masses in atomic units.  If not, then the atoms param will designate masses
        :type masses: list
        :param startStructure:An initial structure to initialize all your walkers
        :type startStructure:np.ndarray
        """

        self.atoms = atoms
        self.simName = simName
        self.outputFolder = outputFolder
        self.initialWalkers = initialWalkers
        self.nTimeSteps = nTimeSteps
        self.potential = potential
        self.weighting = weighting
        self.DwSteps = DwSteps
        self.branch_every = branch_every
        self.deltaT = deltaT
        self._branch_step = np.arange(0, nTimeSteps, self.branch_every)
        self._chkptStep = np.arange(equilTime, nTimeSteps, ckptSpacing)
        self._wfnSaveStep = np.arange(equilTime, nTimeSteps, wfnSpacing)
        self._propSteps = np.arange(timeStep, nTimeSteps)
        self._dwSaveStep = self._chkptStep + self.DwSteps
        self._whoFrom = None  # Not descendant weighting yet
        self._walkerV = np.zeros(self.initialWalkers)
        self._vrefAr = np.zeros(self.nTimeSteps)
        self._popAr = np.zeros(self.nTimeSteps)
        self._alpha = 1.0 / (2.0 * deltaT)
        if startStructure is None:
            self.walkerC = np.zeros((self.initialWalkers, len(atoms), dimensions))
        else:
            self.walkerC = np.repeat(np.expand_dims(startStructure, axis=0), self.initialWalkers, axis=0)
        if masses is None:
            masses = np.array([Constants.mass(a) for a in self.atoms])
        self.sigmas = np.sqrt((2 * 0.5 * deltaT) / masses)
        if not os.path.isdir(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.weighting == 'continuous':
            self.contWts = np.ones(self.initialWalkers)
        else:
            self.contWts = None

    def birthOrDeath_vec(self, vref, Desc):
        """
        Chooses whether or not the walker made a bad enough random walk to be removed from the simulation.
        For discrete weighting, this leads to removal or duplication of the walkers.  For continuous, this leads
         to an update of the weights and a potential branching of a large weight walker to the smallest one
         """
        if self.weighting == 'discrete':
            randNums = np.random.random(len(self.walkerC))
            deathMask = np.logical_or((1 - np.exp(-1. * (self._walkerV - vref) * self.deltaT)) < randNums,
                                      self._walkerV < vref)
            self.walkerC = self.walkerC[deathMask]
            self._walkerV = self._walkerV[deathMask]
            randNums = randNums[deathMask]
            if Desc:
                self._whoFrom = self._whoFrom[deathMask]

            birthMask = np.logical_and((np.exp(-1. * (self._walkerV - vref) * self.deltaT) - 1) > randNums,
                                       self._walkerV < vref)
            self.walkerC = np.concatenate((self.walkerC, self.walkerC[birthMask]))
            self._walkerV = np.concatenate((self._walkerV, self._walkerV[birthMask]))
            if Desc:
                self._whoFrom = np.concatenate((self._whoFrom, self._whoFrom[birthMask]))
        else:
            self.contWts = self.contWts * np.exp(-1.0 * (self._walkerV - vref) * self.deltaT)
            thresh = 1.0 / self.initialWalkers
            killMark = np.where(self.contWts < thresh)[0]
            for walker in killMark:
                maxWalker = np.argmax(self.contWts)
                self.walkerC[walker] = np.copy(self.walkerC[maxWalker])
                self._walkerV[walker] = np.copy(self._walkerV[maxWalker])
                if Desc:
                    self._whoFrom[walker] = self._whoFrom[maxWalker]
                self.contWts[maxWalker] /= 2.0
                self.contWts[walker] = np.copy(self.contWts[maxWalker])
        return self.contWts, self._whoFrom, self.walkerC, self._walkerV

    def moveRandomly(self, walkerC):
        disps = np.random.normal(0.0, self.sigmas, size=np.shape(walkerC.transpose(0, 2, 1))).transpose(0, 2, 1)
        return walkerC + disps

    def getVref(self):  # Use potential of all walkers to calculate vref
        """
             Use the energy of all walkers to calculate vref with a correction for the fluctuation in the population
             or weight.
         """
        if self.weighting == 'discrete':
            Vbar = np.average(self._walkerV)
            correction = (len(self._walkerV) - self.initialWalkers) / self.initialWalkers
        else:
            Vbar = np.average(self._walkerV, weights=self.contWts)
            correction = (np.sum(self.contWts - np.ones(self.initialWalkers))) / self.initialWalkers
        vref = Vbar - (self._alpha * correction)
        return vref

    def countUpDescWeights(self, dwts):
        if self.weighting == 'discrete':
            unique, counts = np.unique(self._whoFrom, return_counts=True)
            dwts[unique] = counts
        else:
            for q in range(len(self.contWts)):
                dwts[q] = np.sum(self.contWts[self._whoFrom == q])

    def updateSimArs(self, propStep, Vref):
        self._vrefAr[propStep] = Vref
        if self.weighting == 'discrete':
            self._popAr[propStep] = len(self.walkerC)
        else:
            self._popAr[propStep] = np.sum(self.contWts)

    def propagate(self):
        """
             The main DMC loop.
             1. Move Randomly
             2. Calculate the Potential Energy
             3. Birth/Death
             4. Update Vref
             Additionally, checks when the wavefunction has hit a point where it should save / start descendent
             weighting.
         """
        DW = False
        for propStep in self._propSteps:
            self.walkerC = self.moveRandomly(self.walkerC)
            self._walkerV = self.potential(self.walkerC, self.atoms)

            if propStep == 0:
                v_ref = self.getVref()

            if propStep in self._chkptStep:
                SimArchivist.chkpt(self, propStep)

            if propStep in self._wfnSaveStep:
                dwts = np.zeros(len(self.walkerC))
                parent = np.copy(self.walkerC)
                self._whoFrom = np.arange(len(self.walkerC))
                DW = True

            if propStep in self._dwSaveStep:
                DW = False
                self.countUpDescWeights(dwts=dwts)
                SimArchivist.saveH5(
                    fname=self.outputFolder + "/" + self.simName + "_wfn_" + str(propStep - self.DwSteps) + "ts.hdf5",
                    keyz=['coords', 'desc_weights', 'atoms'],
                    valz=[parent, dwts, self.DwSteps, self.atoms])

            if propStep in self._branch_step:
                self.contWts, self._whoFrom, self.walkerC, self._walkerV = self.birthOrDeath_vec(v_ref, DW)
            else:
                if self.weighting == 'continuous':
                    self.contWts = self.contWts * np.exp(-1.0 * (self._walkerV - v_ref) * self.deltaT)

            v_ref = self.getVref()
            self.updateSimArs(propStep, v_ref)

    def run(self):
        self.propagate()
        vrefCM = Constants.convert(self._vrefAr, "wavenumbers", to_AU=False)
        ts = np.arange(self.nTimeSteps)
        SimArchivist.saveH5(fname=self.outputFolder + "/" + self.simName + "_simInfo.hdf5",
                            keyz=['vrefVsTau', 'popVsTau'],
                            valz=[np.column_stack((ts, vrefCM)), np.column_stack((ts, self._popAr))]
                            )


def DMC_Restart(ckptFolder="pyvibdmc/simulation_results/",
                simName='DMC_Sim',
                timeStep=100):
    dmcSim = SimArchivist.reloadSim(ckptFolder=ckptFolder,
                                    simName=simName,
                                    timeStep=timeStep)
    return dmcSim


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('hi')
