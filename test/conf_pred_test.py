import argparse
import numpy as np
import torch

from conf_pred import non_conformity_score_diff, non_conformity_score_prop, p_value, norm_p_value, calculate_coverage, \
    calculate_strong_coverage

predictions = torch.Tensor([[0.1, 0.5, 0.4], [0.3, 0.6, 0.1]])
targets = torch.Tensor([1, 0]).type(torch.int64)
args = argparse.Namespace()
args.num_classes = 3
args.device = "cpu"

# Difference-based non-conformity scores
exp_results = np.array([-0.1, 0.3])
act_results = non_conformity_score_diff(predictions, targets, args)
np.testing.assert_almost_equal(exp_results, act_results.numpy(), decimal=4)

# Proportion-based non-conformity scores
args.non_conf_score_prop_gamma = 0.1
exp_results = np.array([0.4 / (0.5 + args.non_conf_score_prop_gamma), 0.6 / (0.3 + args.non_conf_score_prop_gamma)])
act_results = non_conformity_score_prop(predictions, targets, args)
np.testing.assert_almost_equal(exp_results, act_results.numpy(), decimal=4)

# Test extreme values
predictions = torch.Tensor([[0, 1., 0.], [0., 0., 1.]])
act_results = non_conformity_score_diff(predictions, targets, args)
print("act_results:", act_results)
act_results = non_conformity_score_prop(predictions, targets, args)
print("act_results:", act_results)

# p-value
non_conf_scores = torch.Tensor([0.1, 0.4, 0.3])
act_score = torch.Tensor([0.2, 0.35])
exp_results = np.array([3 / 4, 2 / 4])
act_result = p_value(non_conf_scores, act_score)
np.testing.assert_almost_equal(act_result.numpy(), exp_results)

# Norm p-values
p_values = torch.Tensor(np.array([[0.2, 0.1, 0.9], [0.5, 0.4, 0.1]]))
exp_result_var0 = np.array([[0.2 / 0.9, 0.1 / 0.9, 1.], [1., 0.4 / 0.5, 0.1 / 0.5]])
exp_result_var1 = np.array([[0.2, 0.1, 1.], [1., 0.4, 0.1]])
var_0_res = norm_p_value(p_values, 0)
np.testing.assert_almost_equal(var_0_res.numpy(), exp_result_var0)
var_1_res = norm_p_value(p_values, 1)
np.testing.assert_almost_equal(var_1_res.numpy(), exp_result_var1)

# Coverage
p_values = torch.Tensor(np.array([[0.91, 0.3, 1.], [1., 0., 0.]]))
targets = torch.Tensor(np.array([1, 0]))
exp_result_0 = 0.5
np.testing.assert_almost_equal(calculate_coverage(p_values, targets, 0.95).numpy(), exp_result_0)
exp_result_1 = 1.0
np.testing.assert_almost_equal(calculate_coverage(p_values, targets, 0.25).numpy(), exp_result_1)

# Strong-validity coverage
p_values = torch.Tensor(np.array([[0.91, 0.3, 1.], [1., 0., 0.]]))
targets = torch.Tensor(np.array([1, 0]))
exp_result_0 = 0.0
np.testing.assert_almost_equal(calculate_strong_coverage(p_values, targets, 0.1).numpy(), exp_result_0)
exp_result_1 = 0.5
np.testing.assert_almost_equal(calculate_strong_coverage(p_values, targets, 0.5).numpy(), exp_result_1)
