from .numpy_cupy import xp


class Optimisation:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params_and_grads):
        self.t += 1
        for i, (p, g) in enumerate(params_and_grads):
            if g is None:
                continue
            if i not in self.m:
                self.m[i] = xp.zeros_like(p)
                self.v[i] = xp.zeros_like(p)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = self.m[i] / (1 - self.beta1 ** self.t)
            vhat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (xp.sqrt(vhat) + self.eps)
