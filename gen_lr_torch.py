import torch
import torch.nn.functional as F


def det_lookahead(p_hat, pi, ref_idx, proj, precision=1e-5):
    for i in range(ref_idx):
        prop = p_hat[i:ref_idx + 1] / torch.sum(p_hat[i:ref_idx + 1])
        prop *= (pi[i] - torch.sum(proj[ref_idx + 1:]))

        # Check violation
        violates = False
        for j in range(len(prop)):
            if (torch.sum(prop[j:]) + torch.sum(proj[ref_idx + 1:])) > (torch.max(pi[i + j:]) + precision):
                violates = True
                break

        if not violates:
            return i

    return ref_idx


def is_in_credal_set(p_hat, pi):
    if len(p_hat.shape) == 1:
        p_hat = p_hat.unsqueeze(0)
    if len(pi.shape) == 1:
        pi = pi.unsqueeze(0)

    c = torch.cumsum(torch.flip(p_hat, dims=[-1]), dim=-1)
    rev_pi = torch.flip(pi, dims=[-1])
    return torch.all(c <= rev_pi, dim=-1)


def gen_lr(p_hat, pi):
    if len(p_hat.shape) < 2:
        p_hat = p_hat.unsqueeze(0)
    if len(pi.shape) < 2:
        pi = pi.unsqueeze(0)

    with torch.no_grad():
        # Sort values
        sorted_pi_rt = pi.sort(descending=True)

        sorted_pi = sorted_pi_rt.values
        sorted_p_hat = torch.gather(p_hat, 1, sorted_pi_rt.indices)

        def search_fn(sorted_p_hat, sorted_pi, sorted_pi_rt_ind):
            result_probs = torch.zeros_like(sorted_p_hat)

            for i in range(sorted_p_hat.shape[0]):
                # Search for loss
                proj = torch.zeros_like(sorted_p_hat[i])

                j = sorted_p_hat[i].shape[0] - 1
                while j >= 0:
                    lookahead = det_lookahead(sorted_p_hat[i], sorted_pi[i], j, proj)
                    proj[lookahead:j + 1] = sorted_p_hat[i][lookahead:j + 1] / torch.sum(
                        sorted_p_hat[i][lookahead:j + 1]) * (
                                                    sorted_pi[i][lookahead] - torch.sum(proj[j + 1:]))

                    j = lookahead - 1

                # e-arrange projection again according to original order
                proj = proj[sorted_pi_rt_ind[i].sort().indices]

                result_probs[i] = proj
            return result_probs

        is_c_set = is_in_credal_set(sorted_p_hat, sorted_pi)

        sorted_p_hat_non_c = sorted_p_hat[~is_c_set]
        sorted_pi_non_c = sorted_pi[~is_c_set]
        sorted_pi_ind_c = sorted_pi_rt.indices[~is_c_set]

        result_probs = torch.zeros_like(sorted_p_hat)
        result_probs[~is_c_set] = search_fn(sorted_p_hat_non_c, sorted_pi_non_c, sorted_pi_ind_c)
        result_probs[is_c_set] = p_hat[is_c_set]

    p_hat = torch.clip(p_hat, 1e-5, 1.)
    result_probs = torch.clip(result_probs, 1e-5, 1.)

    divergence = F.kl_div(p_hat.log(), result_probs, log_target=False, reduction="none")
    divergence = torch.sum(divergence, dim=-1)

    result = torch.where(is_c_set, torch.zeros_like(divergence), divergence)

    return torch.mean(result)


def conv_lr_loss(preds, targets, relax_alpha):
    # Targets must be one-hot encoded
    with torch.no_grad():
        sum_y_hat_prime = torch.sum((1. - targets) * preds, dim=-1)
        y_pred_hat = relax_alpha.unsqueeze(-1) * preds / (sum_y_hat_prime.unsqueeze(-1) + 1e-5)
        y_target_credal = torch.where(targets > 0.1, torch.ones_like(targets) - relax_alpha.unsqueeze(-1), y_pred_hat)

        y_target_credal = torch.clip(y_target_credal, 1e-5, 1.)

    preds = torch.clip(preds, 1e-5, 1.)
    divergence = torch.sum(F.kl_div(preds.log(), y_target_credal, log_target=False, reduction="none"), dim=-1)

    preds = torch.sum(preds * targets, dim=-1)

    result = torch.where(torch.gt(preds, 1. - relax_alpha), torch.zeros_like(divergence), divergence)
    return torch.mean(result)
