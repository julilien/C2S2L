import logging
import math
import os
import time
import wandb

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from conf_pred import non_conformity_score_diff, non_conformity_score_prop, norm_p_value, calculate_strong_coverage, \
    construct_p_values
from dataset.ds_meta import DATASET_GETTERS
from gen_lr_torch import gen_lr
from train import save_checkpoint, get_cosine_schedule_with_warmup, interleave, de_interleave, create_model, \
    get_default_arguments, log_metrics_to_file, init_run
from metrics.misc import AverageMeter, accuracy

from metrics.ece import ece_score, brier_score

logger = logging.getLogger(__name__)
best_acc = 0


def test_cp(args, test_loader, model, epoch, calib_non_conf_scores, non_conf_score_fn, scoring_pref="test"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ece = AverageMeter()
    brier = AverageMeter()

    # Strong validity metrics
    test_strong_validity_005 = AverageMeter()
    test_strong_validity_01 = AverageMeter()
    test_strong_validity_025 = AverageMeter()

    # Credal set size
    test_p_values_mean = AverageMeter()

    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # Outputs are the logits
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            ece.update(ece_score(F.softmax(outputs, dim=-1), targets).item(), inputs.shape[0])
            brier.update(brier_score(F.softmax(outputs, dim=-1),
                                     F.one_hot(targets.type(torch.int64), num_classes=args.num_classes)).item(),
                         inputs.shape[0])

            # Strong validity
            if args.calc_cp_test_stats:
                outputs_softmax = torch.softmax(outputs.detach() / args.T, dim=-1)
                p_values = construct_p_values(calib_non_conf_scores, outputs_softmax, non_conf_score_fn, args)
                norm_p_values = norm_p_value(p_values, variant=args.p_val_norm_var)
                test_strong_validity_005.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets, 0.05).item(),
                    n=norm_p_values.shape[0])
                test_strong_validity_01.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets, 0.1).item(),
                    n=norm_p_values.shape[0])
                test_strong_validity_025.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets, 0.25).item(),
                    n=norm_p_values.shape[0])

                # Credal set size
                test_p_values_mean.update(norm_p_values.mean().item(), n=norm_p_values.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "{score_pref} Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. "
                    "top1: {top1:.2f}. top5: {top5:.2f}. ece: {ece:.2f}. brier: {brier:.2f}. "
                    "sval (0.05/0.1/0.25): {sval005:.2f}/{sval01:.2f}/{sval025:.2f}. P_vals: {mean_p_val:.2f}".format(
                        score_pref=scoring_pref,
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ece=ece.avg,
                        brier=brier.avg,
                        sval005=test_strong_validity_005.avg,
                        sval01=test_strong_validity_01.avg,
                        sval025=test_strong_validity_025.avg,
                        mean_p_val=test_p_values_mean.avg
                    ))
        if not args.no_progress:
            test_loader.close()

    wandb.log({"{}_top1".format(scoring_pref): top1.avg, "{}_top5".format(scoring_pref): top5.avg,
               "{}_loss".format(scoring_pref): losses.avg, "{}_ece".format(scoring_pref): ece.avg,
               "{}_brier".format(scoring_pref): brier.avg})

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    logger.info("ECE: {:.2f}".format(ece.avg))
    logger.info("Brier: {:.2f}".format(brier.avg))
    logger.info("Sval 0.05: {:.2f}".format(test_strong_validity_005.avg))
    logger.info("Sval 0.1: {:.2f}".format(test_strong_validity_01.avg))
    logger.info("Sval 0.25: {:.2f}".format(test_strong_validity_025.avg))
    logger.info("P_val: {:.2f}".format(test_p_values_mean.avg))
    return losses.avg, top1.avg, ece.avg, brier.avg, test_strong_validity_005.avg, test_strong_validity_01.avg, \
           test_strong_validity_025.avg, test_p_values_mean.avg


def main():
    parser = get_default_arguments()

    # Conformal prediction
    parser.add_argument('--cp', default=True, type=bool)
    parser.add_argument('--calibration_split', type=float, default=0.25)
    parser.add_argument('--non_conf_score_variant', type=int, default=0)
    parser.add_argument('--non_conf_score_prop_gamma', type=float, default=0.1)
    parser.add_argument('--p_val_norm_var', type=int, default=0)
    parser.add_argument('--calibration_weak_aug', action='store_true')
    parser.add_argument('--calc_cp_test_stats', default=True, type=bool)

    args = parser.parse_args()
    global best_acc

    args = init_run(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, calib_dataset, unlabeled_dataset, test_dataset, _ = DATASET_GETTERS[args.dataset](
        args, args.dataset_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    calib_loader = DataLoader(
        calib_dataset,
        sampler=SequentialSampler(calib_dataset),
        batch_size=1,
        num_workers=args.num_workers)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, calib_loader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

    wandb.finish()


def train(args, labeled_trainloader, calib_loader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        calib_loader.sampler.set_epochs(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    n_classes = args.num_classes

    if args.non_conf_score_variant == 0:
        non_conformity_score = non_conformity_score_diff
    else:
        non_conformity_score = non_conformity_score_prop

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        p_values_mean = AverageMeter()

        acc_u_w = AverageMeter()
        acc_u_s = AverageMeter()
        acc_u_p = AverageMeter()

        conf_u_w = AverageMeter()
        conf_u_w_min = AverageMeter()
        conf_u_s = AverageMeter()
        conf_u_s_min = AverageMeter()

        # Coverage metrics
        strong_validity_005 = AverageMeter()
        strong_validity_01 = AverageMeter()
        strong_validity_025 = AverageMeter()

        train_sup_acc = AverageMeter()
        calib_acc = AverageMeter()

        # Perform calibration step
        model.eval()
        with torch.no_grad():
            non_conf_scores = torch.zeros(len(calib_loader), device=args.device)

            for idx, ((inputs_w, inputs_s), targets) in enumerate(calib_loader):
                if not args.calibration_weak_aug:
                    inputs = inputs_s
                else:
                    inputs = inputs_w

                inputs = inputs.to(args.device)
                targets = targets.to(args.device)

                preds_logits = model(inputs)
                preds = torch.softmax(preds_logits / args.T, dim=-1)

                # Calculate non-conformity scores
                non_conf_scores[idx] = non_conformity_score(preds, targets.int(), args).squeeze()

                calib_acc.update(torch.eq(torch.max(preds_logits, dim=-1).indices.squeeze(),
                                          targets).float().mean().item(), n=calib_loader.batch_size)
            # print("Non conf scores:", non_conf_scores)

        model.train()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            # Targets are only used to report debugging stats
            try:
                (inputs_u_w, inputs_u_s), targets_u = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_u = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]

            train_sup_acc.update(torch.eq(torch.max(logits_x, dim=-1).indices, targets_x).float().mean().item(),
                                 n=logits_x.shape[0])
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            with torch.no_grad():
                one_hot_targets = F.one_hot(targets_x.type(torch.int64), num_classes=args.num_classes).float()
            preds_x = torch.softmax(logits_x / args.T, dim=-1)

            preds_x = torch.clip(preds_x, 1e-5, 1.)
            one_hot_targets = torch.clip(one_hot_targets, 1e-5, 1.)

            Lx = F.kl_div(preds_x.log(), one_hot_targets, log_target=False, reduction='batchmean')

            # Unsupervised part
            with torch.no_grad():
                pseudo_label_w = torch.softmax(logits_u_w.detach() / args.T, dim=-1)

                p_values = construct_p_values(non_conf_scores, pseudo_label_w, non_conformity_score, args)

                norm_p_values = norm_p_value(p_values, variant=args.p_val_norm_var)

                # norm_p_values_entropy = torch.distributions.Categorical(norm_p_values).entropy().mean()
                norm_p_values_mean = norm_p_values.mean()
                p_values_mean.update(norm_p_values_mean.item(), n=norm_p_values.shape[0])

                # Question: Is the largest p value also the u_pred?
                # print("Largest p=:", torch.max(norm_p_values, dim=-1).indices.to(args.device) == u_pred.to(args.device))
                preds_u_w = torch.max(pseudo_label_w, dim=-1).indices.to(args.device)
                preds_u_p = torch.max(norm_p_values, dim=-1).indices.to(args.device)
                targets_u = targets_u.to(args.device)

                acc_u_w_tmp = torch.mean(torch.eq(preds_u_w, targets_u).float())
                acc_u_p_tmp = torch.mean(torch.eq(preds_u_p, targets_u).float())

                acc_u_w.update(acc_u_w_tmp.item(), n=preds_u_w.shape[0])
                acc_u_p.update(acc_u_p_tmp.item(), n=preds_u_p.shape[0])

                conf_u_w.update(torch.mean(torch.max(pseudo_label_w, dim=-1).values), n=preds_u_w.shape[0])
                conf_u_w_min.update(torch.mean(torch.min(pseudo_label_w, dim=-1).values), n=preds_u_w.shape[0])

            pseudo_label_s = torch.softmax(logits_u_s / args.T, dim=-1)
            with torch.no_grad():
                preds_u_s = torch.max(pseudo_label_s, dim=-1).indices.to(args.device)
                acc_u_s_tmp = torch.mean(torch.eq(preds_u_s, targets_u).float())
                acc_u_s.update(acc_u_s_tmp.item(), n=preds_u_s.shape[0])

                conf_u_s.update(torch.mean(torch.max(pseudo_label_s, dim=-1).values), n=preds_u_s.shape[0])
                conf_u_s_min.update(torch.mean(torch.min(pseudo_label_s, dim=-1).values), n=preds_u_s.shape[0])

                strong_validity_005.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets_u, 0.05).item(),
                    n=norm_p_values.shape[0])
                strong_validity_01.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets_u, 0.1).item(),
                    n=norm_p_values.shape[0])
                strong_validity_025.update(
                    calculate_strong_coverage(norm_p_values.to(args.device), targets_u, 0.25).item(),
                    n=norm_p_values.shape[0])

            # Execute on CPU as this is faster at the moment
            Lu = gen_lr(pseudo_label_s.to("cpu"), norm_p_values.to("cpu")).to(args.device)

            mask = torch.ones(pseudo_label_w.shape[0]).to(args.device)

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. "
                    "Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. "
                    "Mask: {mask:.2f}. P val: {p_val:.2f}. u_acc (w/s/p): ({acc_u_w:.2f}/{acc_u_s:.2f}/{acc_u_p:.2f}). "
                    "Conf_u_w (max/min): {conf_u_w:.2f}/{conf_u_w_min:.2f}. "
                    "Conf_u_s (max/min): {conf_u_s:.2f}/{conf_u_s_min:.2f}. "
                    "Acc train (sup): {train_acc:.2f}. Acc calib: {calib_acc:.2f}.".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        p_val=p_values_mean.avg,
                        mask=mask_probs.avg,
                        acc_u_w=acc_u_w.avg,
                        acc_u_s=acc_u_s.avg,
                        acc_u_p=acc_u_p.avg,
                        conf_u_w=conf_u_w.avg,
                        conf_u_w_min=conf_u_w_min.avg,
                        conf_u_s=conf_u_s.avg,
                        conf_u_s_min=conf_u_s_min.avg,
                        train_acc=train_sup_acc.avg,
                        calib_acc=calib_acc.avg
                    ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            scoring_pref = "test"
            if args.validation_scoring:
                scoring_pref = "val"
            test_loss, test_acc, test_ece, test_brier, test_sval005, test_sval01, test_sval025, test_p_vals = test_cp(
                args,
                test_loader,
                test_model,
                epoch,
                non_conf_scores,
                non_conformity_score,
                scoring_pref=scoring_pref)

            # Calculate test strong validity

            args.writer.add_scalar('train/train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/p_val', p_values_mean.avg, epoch)
            args.writer.add_scalar('train/acc_u_w', acc_u_w.avg, epoch)
            args.writer.add_scalar('train/acc_u_s', acc_u_s.avg, epoch)
            args.writer.add_scalar('train/acc_u_p', acc_u_p.avg, epoch)
            args.writer.add_scalar('train/conf_u_w', conf_u_w.avg, epoch)
            args.writer.add_scalar('train/conf_u_w_min', conf_u_w_min.avg, epoch)
            args.writer.add_scalar('train/conf_u_s', conf_u_s.avg, epoch)
            args.writer.add_scalar('train/conf_u_s_min', conf_u_s_min.avg, epoch)
            args.writer.add_scalar('train/sup_acc', train_sup_acc.avg, epoch)  # sup acc
            args.writer.add_scalar('train/sval_005', strong_validity_005.avg, epoch)
            args.writer.add_scalar('train/sval_01', strong_validity_01.avg, epoch)
            args.writer.add_scalar('train/sval_025', strong_validity_025.avg, epoch)
            args.writer.add_scalar('calib/non_conf_scores', torch.mean(non_conf_scores).item(), epoch)
            args.writer.add_scalar('calib/acc', calib_acc.avg, epoch)
            args.writer.add_scalar('{}/{}_acc'.format(scoring_pref, scoring_pref), test_acc, epoch)
            args.writer.add_scalar('{}/{}_loss'.format(scoring_pref, scoring_pref), test_loss, epoch)
            args.writer.add_scalar('{}/{}_ece'.format(scoring_pref, scoring_pref), test_ece, epoch)
            args.writer.add_scalar('{}/{}_brier'.format(scoring_pref, scoring_pref), test_brier, epoch)
            if args.calc_cp_test_stats:
                args.writer.add_scalar('{}/{}_sval_005'.format(scoring_pref, scoring_pref), test_sval005, epoch)
                args.writer.add_scalar('{}/{}_sval_01'.format(scoring_pref, scoring_pref), test_sval01, epoch)
                args.writer.add_scalar('{}/{}_sval_025'.format(scoring_pref, scoring_pref), test_sval025, epoch)
                args.writer.add_scalar('{}/{}_p_val'.format(scoring_pref, scoring_pref), test_p_vals, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            wandb.log({"train_loss": losses.avg, "train_loss_x": losses_x.avg, "train_loss_u": losses_u.avg,
                       "mask": mask_probs.avg, "p_val": p_values_mean.avg, "acc_u_w": acc_u_w.avg,
                       "acc_u_s": acc_u_s.avg, "acc_u_p": acc_u_p.avg, "conf_u_w": conf_u_w.avg,
                       "conf_u_w_min": conf_u_w_min.avg, "conf_u_s": conf_u_s.avg,
                       "conf_u_s_min": conf_u_s_min.avg, "non_conf_scores": torch.mean(non_conf_scores).item(),
                       "calib_acc": calib_acc.avg, "train_sup_acc": train_sup_acc.avg})

            # Log metrics to file
            log_metrics_to_file(args, epoch, test_acc, test_ece, test_brier, scoring_pref=scoring_pref)

    if args.local_rank in [-1, 0]:
        args.writer.close()


if __name__ == '__main__':
    main()
