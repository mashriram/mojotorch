from mojotorch import MojTensor, randn, zeros
from mojotorch import matmul, add, relu, sigmoid, log

trait Module(Copyable,Movable):
    fn forward(mut self, input: MojTensor) -> MojTensor: ...
    fn parameters(self) -> List[MojTensor]: ...
    fn zero_grad(mut self): ...

struct Linear(Module, Copyable, Movable):
    var weight: MojTensor
    var bias: MojTensor
    fn __init__(out self, in_feats: Int, out_feats: Int):
        self.weight = randn(List(in_feats, out_feats), requires_grad=True)
        self.bias = zeros(List(1, out_feats), requires_grad=True)
    fn forward(mut self, input: MojTensor) -> MojTensor:
        return add(matmul(input, self.weight), self.bias)
    fn parameters(self) -> List[MojTensor]:
        var params = List(self.weight.copy())
        params.append(self.bias.copy())
        return params.copy()
    fn zero_grad(mut self):
        self.weight.zero_grad()
        self.bias.zero_grad()

@register_passable("trivial")
struct ReLU(Module, Copyable, Movable):
    fn __init__(out self):
        pass
    fn forward(mut self, input: MojTensor) -> MojTensor:
        return relu(input)
    fn parameters(self) -> List[MojTensor]:
        return List[MojTensor]()
    fn zero_grad(mut self):
        pass

@register_passable("trivial")
struct Sigmoid(Module, Copyable, Movable):
    fn __init__(out self):
        pass
    fn forward(mut self, input: MojTensor) -> MojTensor:
        return sigmoid(input)
    fn parameters(self) -> List[MojTensor]:
        return List[MojTensor]()
    fn zero_grad(mut self):
        pass

struct Identity(Module, Copyable, Movable):
    fn __init__(out self):
        pass
    fn forward(mut self, input: MojTensor) -> MojTensor:
        return input.copy()
    fn parameters(self) -> List[MojTensor]:
        return List[MojTensor]()
    fn zero_grad(mut self):
        pass

struct Sequential[Head: Module, Tail: Module](Module, Copyable, Movable):
    var head: Head
    var tail: Tail

    fn __init__(out self, head: Head, tail: Tail):
        self.head = head.copy()
        self.tail = tail.copy()

    fn forward(mut self, input: MojTensor) -> MojTensor:
        var x = self.head.forward(input)
        return self.tail.forward(x)

    fn parameters(self) -> List[MojTensor]:
        var params = self.head.parameters()
        params.extend(self.tail.parameters())
        return params.copy()

    fn zero_grad(mut self):
        self.head.zero_grad()
        self.tail.zero_grad()
