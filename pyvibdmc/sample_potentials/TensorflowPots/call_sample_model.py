from pyvibdmc.simulation_utilities import *


def sample_h4o2_pot(cds, model, extra_args):
    descriptor = extra_args['descriptor']
    batch_size = extra_args['batch_size']
    cds = descriptor.run(cds)
    pots_wn = (model.predict(cds, batch_size=batch_size)).flatten()
    return Constants.convert(pots_wn, 'wavenumbers', to_AU=True)
