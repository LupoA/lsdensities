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
        "--prec",
        metavar="NumericalPrecision",
        type=int,
        help="Numerical precision, approximatively in decimal digits. Default=64",
        default=64,
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
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--alpha",
        metavar="Alpha",
        type=float,
        help="alpha parameter define different measure in the functional A. Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=50",
        default=50,
    )
    parser.add_argument(
        "--plots",
        type=bool,
        help="Show a shitton of plots. Default=False.",
        default=False,
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
        help="Numerical precision, approximatively in decimal digits. Default=64",
        default=64,
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
        help="Lower integration bound for functional A, Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--alpha",
        metavar="Alpha",
        type=float,
        help="alpha parameter define different measure in the functional A. Default=0",
        default=0.0,
    )
    parser.add_argument(
        "--ne",
        type=int,
        help="Number of points in energy at which the reconstruction is evaluated, between 0 and emax. Default=50",
        default=50,
    )
    parser.add_argument(
        "--plots",
        type=bool,
        help="Show a shitton of plots. Default=False.",
        default=False,
    )
    args = parser.parse_args()
    return args
