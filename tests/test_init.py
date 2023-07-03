import sys

sys.path.append("..")
from importall import *


def main():
    print(LogMessage(), " Modules imported ")
    print(LogMessage(), " Success! ")
    exit(1)


if __name__ == "__main__":
    main()
