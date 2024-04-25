import argparse
from lsdensities.utils.rhoUtils import Inputs


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datapath",
        metavar="DataPile",
        type=str,
        help="Path to data file",
        required=True,
    )
    parser.add_argument(
        "--prec",
        metavar="NumericalPrecision",
        type=int,
        help="Numerical precision, approximatively in decimal digits. NOTE: if too high it will be automatically reduced to an optimal value. Default=105",
        default=105,
    )
    parser.add_argument(
        "--tmax",
        metavar="Tmax",
        type=int,
        help="The reconstruction will be performed using correlators c(1), ... c(tmax). If not specified, tmax will be inferred from the time extent of the lattice.",
        default=0,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel. Has units of energy.",
        default=0.1,
    )
    parser.add_argument(
        "--nboot",
        metavar="BootstrapSampleSize",
        type=int,
        help="Number of bootstrap samples. Default=300",
        default=300,
    )
    parser.add_argument(
        "--emax",
        type=float,
        help="Maximum energy at which the spectral density is evaluated. Default=1",
        default=1,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated. Default=1e-2.",
        default=1e-2,
    )
    parser.add_argument(
        "--e0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="Nalpha",
        type=int,
        help="Number of alpha parameters, defining different measure in the functional A, to be used. Allowed values=1,2,3. Default=1, corresponding to alpha=0.",
        default=1,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=20",
        default=20,
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        help="Accepted stirngs are 'EXP' or 'COSH', depending on the correlator being periodic or open.",
        default="EXP",
    )
    parser.add_argument(
        "--A0cut",
        type=float,
        help="Minimum value of A/A0 that is accepted, Default=0.1",
        default=0.1,
    )
    parser.add_argument(
        "--kerneltype",
        type=str,
        help="Accepted stirngs are 'FULLNORMGAUSS', 'HALFNORMGAUSS' or 'CAUCHY', depending on which smearing kernel.",
        default="FULLNORMGAUSS",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        help="Accepted stirngs are 'WARNING', 'INFO' or 'DEBUG'. Default='WARNING'. Setting 'INFO' leads to an extensive output tracking the details of the scan over lambda and alpha.",
        default="WARNING",
    )
    args = parser.parse_args()
    inputs = Inputs()
    inputs.datapath = args.datapath
    inputs.tmax = args.tmax
    inputs.periodicity = args.periodicity
    inputs.prec = args.prec
    inputs.outdir = args.outdir
    inputs.kerneltype = args.kerneltype
    inputs.num_boot = args.nboot
    inputs.sigma = args.sigma
    inputs.emax = args.emax
    inputs.emin = args.emin
    inputs.e0 = args.e0
    inputs.Ne = args.ne
    inputs.Na = args.Na
    inputs.A0cut = args.A0cut
    inputs.loglevel = args.loglevel
    return inputs


def parse_synthetic_inputs():
    """
    Parse command line for arguments which are turned into attributes of the Inputs class
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--T",
        metavar="TimeExtent",
        type=int,
        help="time extent of the lattice",
        required=True,
    )
    parser.add_argument(
        "--nms",
        metavar="SampleSize",
        type=int,
        help="Number of samples to be generated. Default=100",
        default=100,
    )
    parser.add_argument(
        "--prec",
        metavar="NumericalPrecision",
        type=int,
        help="Numerical precision, approximatively in decimal digits. NOTE: if too high it will be automatically reduced to an optimal value. Default=105",
        default=105,
    )
    parser.add_argument(
        "--tmax",
        metavar="Tmax",
        type=int,
        help="The reconstruction will be performed using correlators c(1), ... c(tmax). If not specified, tmax will be inferred from the time extent of the lattice.",
        default=0,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel. Has units of energy.",
        default=0.1,
    )
    parser.add_argument(
        "--nboot",
        metavar="BootstrapSampleSize",
        type=int,
        help="Number of bootstrap samples. Default=300",
        default=300,
    )
    parser.add_argument(
        "--emax",
        type=float,
        help="Maximum energy at which the spectral density is evaluated. Default=1",
        default=1,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated. Default=1e-2.",
        default=1e-2,
    )
    parser.add_argument(
        "--e0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="Nalpha",
        type=int,
        help="Number of alpha parameters, defining different measure in the functional A, to be used. Allowed values=1,2,3. Default=1, corresponding to alpha=0.",
        default=1,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=20",
        default=20,
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        help="Accepted stirngs are 'EXP' or 'COSH', depending on the correlator being periodic or open.",
        default="EXP",
    )
    parser.add_argument(
        "--A0cut",
        type=float,
        help="Minimum value of A/A0 that is accepted, Default=0.1",
        default=0.1,
    )
    parser.add_argument(
        "--kerneltype",
        type=str,
        help="Accepted stirngs are 'FULLNORMGAUSS', 'HALFNORMGAUSS' or 'CAUCHY', depending on which smearing kernel.",
        default="FULLNORMGAUSS",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        help="Accepted stirngs are 'WARNING', 'INFO' or 'DEBUG'. Default='WARNING'. Setting 'INFO' leads to an extensive output tracking the details of the scan over lambda and alpha.",
        default="WARNING",
    )
    args = parser.parse_args()
    inputs = Inputs()
    inputs.time_extent = args.T
    inputs.num_samples = args.nms
    inputs.tmax = args.tmax
    inputs.periodicity = args.periodicity
    inputs.prec = args.prec
    inputs.outdir = args.outdir
    inputs.kerneltype = args.kerneltype
    inputs.num_boot = args.nboot
    inputs.sigma = args.sigma
    inputs.emax = args.emax
    inputs.emin = args.emin
    inputs.e0 = args.e0
    inputs.Ne = args.ne
    inputs.Na = args.Na
    inputs.A0cut = args.A0cut
    inputs.loglevel = args.loglevel
    return inputs
