from lsdensities.utils.rhoUtils import generate_seed, Inputs, init_precision


def rng_seeding():
    init_precision(128)
    par = Inputs()

    # Typical parameter setting
    par.time_extent = 32
    par.sigma = 0.25
    par.massNorm = 0.33
    par.Ne = 10
    par.emin = 0.3
    par.emax = 2.2

    seed1 = generate_seed(par)
    # Changing parameter
    par.Ne = 11

    seed2 = generate_seed(par)

    # Coming back to first setting
    par.Ne = 10
    seed3 = generate_seed(par)

    assert seed1 != seed2 and seed1 == seed3
