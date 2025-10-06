from random import random_float64, seed
from math import *

struct MojTensor(Copyable,Movable):
    var _shape: List[Int]
    var data: List[Float64]
    var requires_grad: Bool
    var grad: Optional[Self]

    fn __init__(out self, shape: List[Int], requires_grad: Bool = False):
        self._shape = shape.copy()
        var size = 1
        for d in shape:
            size *= d
        self.data = List[Float64]()
        for _ in range(size):
            self.data.append(0.0)
        self.requires_grad = requires_grad
        self.grad = Optional[Self]()

    fn _get_size(self) -> Int:
        var size = 1
        for d in self._shape:
            size *= d
        return size

    fn shape(self) -> List[Int]:
        return self._shape.copy()

    fn __getitem__(self, idx: Int) -> Float64:
        return self.data[idx]

    fn __setitem__(mut self, idx: Int, val: Float64):
        self.data[idx] = val

    fn zero_grad(mut self):
        if self.requires_grad:
            self.grad = Optional[Self]()

    fn backward(mut self):
        # Minimal stub: no autograd engine implemented.
        if not self.requires_grad:
            print("Cannot call backward on a tensor that does not require grad.")
            return
        if not self.grad:
            self.grad = Optional[Self](ones(self._shape, False))

    fn __str__(self) -> String:
        var s = "Tensor(shape=" + String(self._shape[0])
        var size = self._get_size()
        if size > 0:
            s += ", data=["
            var max_print = 5
            for i in range(min(size, max_print)):
                s += String(self.data[i])
                if i < min(size, max_print) - 1:
                    s += ", "
            if size > max_print:
                s += "..."
            s += "]"
        if self.requires_grad:
            s += ", requires_grad=True"
        return s + ")"

fn ones(shape: List[Int], requires_grad: Bool = False) -> MojTensor:
    var t = MojTensor(shape, requires_grad)
    for i in range(t._get_size()):
        t.data[i] = 1.0
    return t.copy()

fn zeros(shape: List[Int], requires_grad: Bool = False) -> MojTensor:
    return MojTensor(shape, requires_grad)

fn randn(shape: List[Int], requires_grad: Bool = False) -> MojTensor:
    seed(0)
    var t = MojTensor(shape, requires_grad)
    for i in range(t._get_size()):
        var u1 = random_float64()
        while u1 == 0.0:
            u1 = random_float64()
        var u2 = random_float64()
        var z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159 * u2)
        t.data[i] = z0
    return t.copy()
