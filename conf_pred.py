import torch


def non_conformity_score_diff(predictions, targets, args) -> torch.Tensor:
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    class_val = torch.gather(predictions, 1, targets.type(torch.int64))

    # Exclude the target class here
    indices = torch.arange(0, args.num_classes).view(1, -1).repeat(predictions.shape[0], 1).to(args.device)
    mask = torch.zeros_like(indices).bool().to(args.device)
    mask.scatter_(1, targets.type(torch.int64), True)

    selected_predictions = predictions[~mask].view(-1, args.num_classes - 1)

    return torch.max(selected_predictions - class_val, dim=-1).values


def non_conformity_score_prop(predictions, targets, args) -> torch.Tensor:
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    class_val = torch.gather(predictions, 1, targets.type(torch.int64))

    # Exclude the target class here
    indices = torch.arange(0, args.num_classes).view(1, -1).repeat(predictions.shape[0], 1).to(args.device)
    mask = torch.zeros_like(indices).bool().to(args.device)
    mask.scatter_(1, targets.type(torch.int64), True)

    selected_predictions = predictions[~mask].view(-1, args.num_classes - 1)

    return torch.max(selected_predictions, dim=-1).values.squeeze() / (
            class_val.squeeze() + args.non_conf_score_prop_gamma + 1e-5)


def p_value(non_conf_scores, act_score):
    if len(act_score.shape) < 2:
        act_score = act_score.unsqueeze(-1)

    return (torch.sum(non_conf_scores >= act_score, dim=-1) + 1) / (len(non_conf_scores) + 1)


def norm_p_value(p_values, variant):
    if len(p_values.shape) < 2:
        p_values = p_values.unsqueeze(0)

    if variant == 0:
        norm_p_values = p_values / (torch.max(p_values, dim=-1).values.unsqueeze(-1))
    else:
        norm_p_values = p_values.scatter_(1, torch.max(p_values, dim=-1).indices.unsqueeze(-1),
                                          torch.ones_like(p_values))
    return norm_p_values


def calculate_coverage(p_values, targets, confidence):
    # This is the ``classical'' coverage property
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    class_p_values = torch.gather(p_values, 1, targets.type(torch.int64))
    return torch.mean(torch.greater_equal(class_p_values, confidence).float())


def calculate_strong_coverage(p_values, targets, error):
    """
    Functions that measures Pr(p_val(y) <= error) => strong validity.

    :param p_values:
    :param targets:
    :param error:
    :return:
    """
    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    class_p_values = torch.gather(p_values, 1, targets.type(torch.int64))
    return torch.mean(torch.less_equal(class_p_values, error).float())


def construct_p_values(non_conf_scores, preds, non_conf_score_fn, args):
    tmp_non_conf = torch.zeros([preds.shape[0], args.num_classes], device=args.device).detach()
    p_values = torch.zeros([preds.shape[0], args.num_classes]).detach()
    for clz in range(args.num_classes):
        tmp_non_conf[:, clz] = non_conf_score_fn(preds, torch.tensor(clz, device=args.device).repeat(preds.shape[0]),
                                                 args)
        p_values[:, clz] = p_value(non_conf_scores, tmp_non_conf[:, clz])
    return p_values
