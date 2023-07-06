
class FitParameter:
    def __init__(self, val: float, upper: float, lower: float):
        self.value = val
        self.max = upper
        self.min = lower

class GaussianPar:
    def __init__(self, nstates: int = 2):
        self.e1 = FitParameter(0, 0, 0)
        self.w1 = FitParameter(0, 0, 0)
        if nstates == 2:
            self.e2 = FitParameter(0, 0, 0)
            self.w2 = FitParameter(0, 0, 0)
            if nstates ==3:
                self.e3 = FitParameter(0, 0, 0)
                self.w3 = FitParameter(0, 0, 0)
                if nstates==4:
                    self.e4 =FitParameter(0, 0, 0)
                    self.w4=FitParameter(0, 0, 0)

class GaussianFitter:
    def __init__(self):
        self.nboot = 0
