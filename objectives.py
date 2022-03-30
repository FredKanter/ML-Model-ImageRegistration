from regularizer import Regularizer
from distances import DistanceFunction
from interpolation import set_interpolater


class ObjectiveFunction:
    # class to combine dist and reg to one function for solver and autograd
    def __init__(self, distance, regularizer, alpha):
        super(ObjectiveFunction, self).__init__()
        if isinstance(distance, DistanceFunction):
            self.dist = distance
        else:
            raise RuntimeError(f'Given distance function is not a member of DistanceFunction class')
        if isinstance(regularizer, Regularizer):
            self.reg = regularizer
        else:
            self.reg = None
        if isinstance(alpha, (int, float)):
            self.alpha = alpha
        else:
            self.alpha = 0.5
        self.omega = distance.omega
        self.h = distance.h
        self.gridRef = distance.gridRef
        self.m = distance.m

    def evaluate(self, x0, **kwargs):
        # if regularizer is used, distance and reg has to be combined for autograd derivative calculation
        if self.reg is not None:
            return self.dist.evaluate(x0, **kwargs) + self.alpha * self.reg.evaluate(x0, **kwargs)
        else:
            return self.dist.evaluate(x0, **kwargs)

    def reset(self, omega, h, m, gridRef):
        # reset for combined objective / accessible for ML routine
        self.omega = omega
        self.h = h
        self.gridRef = gridRef
        self.m = m

        # reset parameter of fcn (dist, reg) which are part of objective, if these change (check call by reference).
        # needed for ML routine in optimization
        self.dist.omega = self.omega
        self.dist.h = self.h
        self.dist.gridRef = self.gridRef
        self.dist.m = m
        self.dist.inter = set_interpolater('linearFAIR', omega, m, h)

        if self.reg is not None:
            self.reg.h = self.h
            self.reg.omega = self.omega
            self.reg.gridRef = self.gridRef
            self.reg.m = m

    def __copy__(self):
        return ObjectiveFunction(self.dist, self.reg, self.alpha)
