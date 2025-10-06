from mojotorch import MojTensor,ones
from mojotorch import sub, mul, sum, log, add, neg
from math import *

fn mse_loss(y_pred: MojTensor, y_true: MojTensor) -> MojTensor:
    var diff = sub(y_pred, y_true)
    var sq_diff = mul(diff, diff)
    var s = sum(sq_diff)
    var loss_val = s[0] / y_pred._get_size()
    var loss = MojTensor(List(1), y_pred.requires_grad)
    loss[0] = loss_val
    return loss.copy()

fn bce_loss(y_pred: MojTensor, y_true: MojTensor) -> MojTensor:
    var term1 = mul(y_true, log(y_pred))
    var one = ones(y_true.shape())
    var one_minus_true = sub(one, y_true)
    var one_minus_pred = sub(one, y_pred)
    var term2 = mul(one_minus_true, log(one_minus_pred))
    var combined = add(term1, term2)
    var loss = neg(sum(combined))
    return loss.copy()
