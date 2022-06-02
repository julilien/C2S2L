import logging
import math
import os
import time
from copy import deepcopy
from typing import Optional, Sequence

from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t

import wandb
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset.ds_meta import DATASET_GETTERS
from train import get_default_arguments, create_model, get_cosine_schedule_with_warmup, interleave, de_interleave, \
    save_checkpoint, test, log_metrics_to_file, init_run
from metrics.misc import AverageMeter

logger = logging.getLogger(__name__)
best_acc = 0


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


class UnlabeledDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False,
                 sampler: Optional[Sampler] = None, batch_sampler: Optional[Sampler[Sequence]] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)




def main():
    parser = get_default_arguments()
    parser.add_argument('--flex', default=True, type=bool)
    parser.add_argument('--p_cutoff', default=0.95, type=float)
    parser.add_argument('--thresh_warmup', default=True, type=bool)
    parser.set_defaults(T=0.5)

    args = parser.parse_args()
    global best_acc

    args = init_run(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, _ = DATASET_GETTERS[args.dataset](
        args, args.dataset_dir, return_idxs=True)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlab_ds_len = len(unlabeled_dataset)
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

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
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, unlab_ds_len)

    wandb.finish()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, unlab_ds_len):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    selected_label = (torch.ones((unlab_ds_len,), dtype=torch.long, ) * -1).to(args.device)
    classwise_acc = torch.zeros((args.num_classes,)).to(args.device)

    it = 0
    p_fn = Get_Scalar(args.p_cutoff)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        acc_u_w = AverageMeter()
        acc_u_s = AverageMeter()
        acc_u_s_masked = AverageMeter()

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
                (inputs_u_w, inputs_u_s), targets_u, unlabeled_idxs = unlabeled_iter.next()

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < unlab_ds_len:  # not all(5w) -1
                if args.thresh_warmup:
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                else:
                    wo_negative_one = deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(args.num_classes):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            # Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            log_pred_x = F.log_softmax(logits_x, dim=-1)
            Lx = F.nll_loss(log_pred_x, targets_x, reduction='mean')

            # Unsupervised loss
            p_cutoff = p_fn(it)

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))).float()  # convex
            select = max_probs.ge(p_cutoff).long()
            masked_loss = ce_loss(logits_u_s, max_idx, True, reduction='none') * mask
            Lu = masked_loss.mean()

            if unlabeled_idxs[select == 1].nelement() != 0:
                selected_label[unlabeled_idxs[select == 1]] = max_idx.long()[select == 1]


            preds_u_w = torch.max(logits_u_w, dim=-1).indices.to(args.device)
            preds_u_s = torch.max(logits_u_s, dim=-1).indices.to(args.device)
            targets_u = targets_u.to(args.device)

            acc_u_w_tmp = torch.mean(torch.eq(preds_u_w, targets_u).float())
            acc_u_s_tmp = torch.mean(torch.eq(preds_u_s, targets_u).float())
            acc_u_s_masked_tmp = torch.sum(torch.eq(preds_u_s, targets_u).float() * mask) / torch.sum(mask)

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            acc_u_w.update(acc_u_w_tmp.item())
            acc_u_s.update(acc_u_s_tmp.item())
            acc_u_s_masked.update(acc_u_s_masked_tmp.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. "
                                      "Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. "
                                      "Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. "
                                      "u_acc (w/s/m): ({acc_u_w:.2f}/{acc_u_s:.2f}/{acc_u_s_masked:.2f}).".format(
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
                    mask=mask_probs.avg,
                    acc_u_w=acc_u_w.avg,
                    acc_u_s=acc_u_s.avg,
                    acc_u_s_masked=acc_u_s_masked.avg))
                p_bar.update()

            it += 1

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

            test_loss, test_acc, test_ece, test_brier = test(args, test_loader, test_model, epoch,
                                                             scoring_pref=scoring_pref)

            args.writer.add_scalar('train/train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/acc_u_w', acc_u_w.avg, epoch)
            args.writer.add_scalar('train/acc_u_s', acc_u_s.avg, epoch)
            args.writer.add_scalar('train/acc_u_s_masked', acc_u_s_masked.avg, epoch)
            args.writer.add_scalar('{}/{}_acc'.format(scoring_pref, scoring_pref), test_acc, epoch)
            args.writer.add_scalar('{}/{}_loss'.format(scoring_pref, scoring_pref), test_loss, epoch)
            args.writer.add_scalar('{}/{}_ece'.format(scoring_pref, scoring_pref), test_ece, epoch)
            args.writer.add_scalar('{}/{}_brier'.format(scoring_pref, scoring_pref), test_brier, epoch)

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
                       "mask": mask_probs.avg, "acc_u_w": acc_u_w.avg, "acc_u_s": acc_u_s.avg,
                       "acc_u_s_masked": acc_u_s_masked.avg})

            # Log metrics to file
            log_metrics_to_file(args, epoch, test_acc, test_ece, test_brier, scoring_pref=scoring_pref)

    if args.local_rank in [-1, 0]:
        args.writer.close()


if __name__ == '__main__':
    main()
