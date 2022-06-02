import numpy as np
import torch
from metrics.ece import ece_score

preds = np.array([[1., 0., 0.], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3]])
y_test = np.array([0, 1, 2])
n_bins = 4

result = ece_score(torch.Tensor(preds), torch.Tensor(y_test), n_bins, device="cpu")
expected_result = (1 / preds.shape[0]) * ((2 * np.abs(1 - 0.9)) + (1 * np.abs(0 - 0.4)))
np.testing.assert_almost_equal(result.numpy(), expected_result, decimal=4)
