from typing import Dict, Any
import hashlib
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def generate_run_uid(args):
    result = "{}@{}.{}_ts{}_es{}_mod{}_mu{}_lu{}_T{}_lr{}_bs{}".format(args.dataset, args.num_labeled, args.seed,
                                                                       args.total_steps, args.eval_step,
                                                                       args.arch, args.mu,
                                                                       args.lambda_u, args.T, args.lr, args.batch_size)
    if "cssl" in args and args.cssl:
        result += "_cssl"
    elif "flex" in args and args.flex:
        result += "_flex_p{}".format(args.p_cutoff)
    elif "uda" in args and args.uda:
        result += "_uda"
    elif "cp" not in args or not args.cp:
        result += "_fixmatch_t{}_da{}".format(args.threshold, args.da)
    else:
        result += "_cp_cs{}_var{}_gamma{}_pnorm{}_wcal{}".format(args.calibration_split, args.non_conf_score_variant,
                                                                 args.non_conf_score_prop_gamma, args.p_val_norm_var,
                                                                 args.calibration_weak_aug)

    if args.validation_scoring:
        result += "_val"
        # Assume this is a hyperparameter run
        result += "_ho{}".format(dict_hash(args.__dict__))

    return result

