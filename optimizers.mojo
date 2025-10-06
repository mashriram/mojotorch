from mojotorch import MojTensor, zeros
from math import *

trait Optimizer:
    fn step(mut self): ...
    fn zero_grad(mut self): ...

struct SGD(Optimizer):
    var params: List[MojTensor]
    var lr: Float32
    fn __init__(out self, params: List[MojTensor], lr: Float32=0.01):
        self.params = params.copy()
        self.lr = lr
    fn step(mut self):
        for i in range(len(self.params)):
            var p = self.params[i].copy()
            if p.grad:
                var grad = p.grad.value().copy()
                for j in range(p._get_size()):
                    var upd = p.data[j] - Float64(self.lr) * grad.data[j]
                    p.data[j] = upd
    fn zero_grad(mut self):
        for i in range(len(self.params)):
            self.params[i].zero_grad()

struct Adam(Optimizer):
    var params: List[MojTensor]
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var m: List[MojTensor]
    var v: List[MojTensor]
    var t: Int
    fn __init__(out self, params: List[MojTensor], lr: Float32=0.001, b1: Float32=0.9, b2: Float32=0.999, e: Float32=1e-8):
        self.params = params.copy()
        self.lr = lr
        self.beta1 = b1
        self.beta2 = b2
        self.eps = e
        self.t = 0
        self.m = List[MojTensor]()
        self.v = List[MojTensor]()
        for p in params:
            self.m.append(zeros(p.shape()))
            self.v.append(zeros(p.shape()))
    fn step(mut self):
        self.t += 1
        for i in range(len(self.params)):
            var p = self.params[i].copy()
            if p.grad:
                var grad = p.grad.value().copy()
                var mi = self.m[i].copy()
                var vi = self.v[i].copy()
                for j in range(p._get_size()):
                    var g = grad.data[j]
                    var new_m = Float64(self.beta1) * mi.data[j] + (Float64(1) - Float64(self.beta1)) * g
                    mi.data[j] = new_m
                    var new_v = Float64(self.beta2) * vi.data[j] + (Float64(1) - Float64(self.beta2)) * (g * g)
                    vi.data[j] = new_v
                    var m_hat = Float64(new_m) / (Float64(1) - Float64(pow(self.beta1, self.t)))
                    var v_hat = Float64(new_v )/ (Float64(1) - Float64(pow(self.beta2, self.t)))
                    var upd = p.data[j] - Float64(self.lr) * m_hat / (sqrt(v_hat) + Float64(self.eps))
                    p.data[j] = upd
    fn zero_grad(mut self):
        for i in range(len(self.params)):
            self.params[i].zero_grad()
