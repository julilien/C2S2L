import torch
import torch.nn.functional as F
import numpy as np

from gen_lr_torch import gen_lr, conv_lr_loss

y_true = torch.Tensor(
    np.array([[1., 0.25, 0.25], [0.25, 1., 0.25], [0.25, 1., 0.25], [0.25, 1., 0.25]], dtype=np.float32))
y_pred = torch.Tensor(
    np.array([[0.675, 0.175, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15], [0.15, 0.7, 0.15]], dtype=np.float32))
exp_result = (0.01342932 + 0.006164 * 3) / 4
loss_result = gen_lr(y_pred, y_true)
print("Loss result:", loss_result)
print("Exp result:", exp_result)
np.testing.assert_almost_equal(loss_result.numpy(), exp_result, decimal=4)

y_true = torch.Tensor(
    np.array([[1., 0.0, 0.0]], dtype=np.float32))
y_pred = torch.Tensor(
    np.array([[0.675, 0.175, 0.15]], dtype=np.float32))
exp_result = F.kl_div(y_pred.log(), y_true, log_target=False, reduction="batchmean")
loss_result = gen_lr(y_pred, y_true)
print("Loss result:", loss_result)
print("Exp result:", exp_result)
np.testing.assert_almost_equal(loss_result.numpy(), exp_result, decimal=3)

y_true = torch.Tensor(
    np.array([[1., 1e-5, 0.0]], dtype=np.float32))
y_pred = torch.Tensor(
    np.array([[1. - 1e-3, 1e-3, 0]], dtype=np.float32))
exp_result = F.kl_div(y_pred.log(), y_true, log_target=False, reduction="batchmean")
loss_result = gen_lr(y_pred, y_true)
print("Loss result:", loss_result)
print("Exp result:", exp_result)
np.testing.assert_almost_equal(loss_result.numpy(), exp_result, decimal=4)

lr_loss = conv_lr_loss(torch.Tensor(np.array([[0.0, 1.0, 0.0], [0.5, 0.4, 0.1]], dtype=np.float32)),
                       torch.Tensor(np.array([[1., 0., 0.], [1., 0., 0.]])),
                       torch.Tensor([1e-3, 1e-3]))
print(lr_loss)
