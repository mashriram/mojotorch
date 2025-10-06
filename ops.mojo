from mojotorch import MojTensor
from math import exp, log as math_log

fn add(a: MojTensor, b: MojTensor) -> MojTensor:
    # Supports broadcasting for b (e.g., bias)
    var a_shape = a.shape()
    var b_shape = b.shape()
    var out = MojTensor(a_shape, a.requires_grad or b.requires_grad)

    if a_shape == b_shape:
        for i in range(out._get_size()):
            out.data[i] = a[i] + b[i]
    elif len(a_shape) == 2 and len(b_shape) == 2 and a_shape[1] == b_shape[1] and a_shape[0] == 1:
        # Broadcasting b across rows of a
        var rows = a_shape[0]
        var cols = a_shape[1]
        for r in range(rows):
            for c in range(cols):
                out.data[r * cols + c] = a.data[r * cols + c] + b.data[c]
    return out.copy()

fn sub(a: MojTensor, b: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad or b.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = a[i] - b[i]
    return out.copy()

fn mul(a: MojTensor, b: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad or b.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = a[i] * b[i]
    return out.copy()

fn neg(a: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = -a[i]
    return out.copy()

fn matmul(a: MojTensor, b: MojTensor) -> MojTensor:
    var A = a.shape()[0]
    var K = a.shape()[1]
    var B = b.shape()[1]
    var out = MojTensor(List(A, B), a.requires_grad or b.requires_grad)
    for i in range(A):
        for j in range(B):
            var sumv = 0.0
            for k in range(K):
                sumv += a.data[i * K + k] * b.data[k * B + j]
            out.data[i * B + j] = sumv
    return out.copy()

fn transpose(a: MojTensor) -> MojTensor:
    var M = a.shape()[0]
    var N = a.shape()[1]
    var out = MojTensor(List(N, M), a.requires_grad)
    for i in range(M):
        for j in range(N):
            out.data[j * M + i] = a.data[i * N + j]
    return out.copy()

fn relu(a: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = a[i] if a[i] > 0.0 else 0.0
    return out.copy()

fn sigmoid(a: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = 1.0 / (1.0 + exp(-a[i]))
    return out.copy()

fn sum(a: MojTensor) -> MojTensor:
    var s = 0.0
    for i in range(a._get_size()):
        s += a[i]
    var out = MojTensor(List(1), a.requires_grad)
    out.data[0] = s
    return out.copy()
fn log(a: MojTensor) -> MojTensor:
    var out = MojTensor(a.shape(), a.requires_grad)
    for i in range(out._get_size()):
        out.data[i] = math_log(a.data[i])
    return out.copy()
