import argparse


def parseArgumentPeak():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-T",
        metavar="TimeExtent",
        type=int,
        help="time extent of the lattice (non periodic)",
        required=True,
    )
    parser.add_argument(
        "--tmax",
        metavar="Tmax",
        type=int,
        help="The reconstruction will be performed using correlators c(0), c(1), ... c(tmax). If not specified, tmax will be set to the largest correlator available.",
        default=0,
    )
    parser.add_argument(
        "--prec",
        metavar="NumericalPrecision",
        type=int,
        help="Numerical precision, approximatively in decimal digits. Default=105",
        default=105,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel",
        default=0.1,
    )
    parser.add_argument(
        "--nms",
        metavar="SampleSize",
        type=int,
        help="Number of samples to be generated. Default=100",
        default=100,
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
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=8",
        default=8,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="NAlpha",
        type=int,
        help="Number of alpha parameters defining different measure in the functional A. The default value, n_alpha=1, performs with alpha=0. n_alpha=2 uses alpha = 0, -1. Default=1.",
        default=1,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=50",
        default=50,
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        help="Accepted stirngs are 'EXP' or 'COSH', depending on the correlator being periodic or open.",
        default="EXP",
    )
    parser.add_argument(
        "--kerneltype",
        type=str,
        help="Accepted stirngs are 'FULLNORMGAUSS', 'HALFNORMGAUSS' or 'CAUCHY', depending on which smearing kernel.",
        default="FULLNORMGAUSS",
    )
    args = parser.parse_args()
    return args


def parseArgumentRhoFromData():
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
        help="The reconstruction will be performed using correlators c(0), c(1), ... c(tmax). If not specified, tmax will be set to the largest correlator available.",
        default=0,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel",
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
        "--mpi",
        metavar="Mpi",
        type=float,
        help="Reference mass used to normalise the energies. Default=1",
        default=1,
    )
    parser.add_argument(
        "--emax",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=8",
        default=8,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=Mpi/20",
        default=0.0,
    )
    parser.add_argument(
        "--e0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="NAlpha",
        type=int,
        help="Number of alpha parameters defining different measure in the functional A. The default value, n_alpha=1, performs with alpha=0. n_alpha=2 uses alpha = 0, -1. Default=1.",
        default=1,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=50",
        default=50,
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
    args = parser.parse_args()
    return args


def parseArgumentSynthData():
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
        help="The reconstruction will be performed using correlators c(0), c(1), ... c(tmax). If not specified, tmax will be set to the largest correlator available.",
        default=0,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel",
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
        "--mpi",
        metavar="Mpi",
        type=float,
        help="Reference mass used to normalise the energies. Default=1",
        default=1,
    )
    parser.add_argument(
        "--emax",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=8",
        default=8,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=Mpi/20",
        default=0.0,
    )
    parser.add_argument(
        "--e0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="NAlpha",
        type=int,
        help="Number of alpha parameters defining different measure in the functional A. The default value, n_alpha=1, performs with alpha=0. n_alpha=2 uses alpha = 0, -1. Default=1.",
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
    args = parser.parse_args()
    return args


def parseArgumentPrintSamples():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datapath",
        metavar="DataPile",
        type=str,
        help="Path to data file",
        required=True,
    )
    parser.add_argument(
        "-rhopath",
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
        help="The reconstruction will be performed using correlators c(0), c(1), ... c(tmax). If not specified, tmax will be set to the largest correlator available.",
        default=0,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--sigma",
        metavar="GaussianWidth",
        type=float,
        help="Radius of the smearing kernel",
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
        "--mpi",
        metavar="Mpi",
        type=float,
        help="Reference mass used to normalise the energies. Default=1",
        default=1,
    )
    parser.add_argument(
        "--emax",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=8",
        default=8,
    )
    parser.add_argument(
        "--emin",
        type=float,
        help="Maximum energy at which the spectral density is evaluated in unity of Mpi, which is set into main(), Default=Mpi/20",
        default=0.0,
    )
    parser.add_argument(
        "--e0",
        type=float,
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--Na",
        metavar="NAlpha",
        type=int,
        help="Number of alpha parameters defining different measure in the functional A. The default value, n_alpha=1, performs with alpha=0. n_alpha=2 uses alpha = 0, -1. Default=1.",
        default=1,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=50",
        default=50,
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
    args = parser.parse_args()
    return args
