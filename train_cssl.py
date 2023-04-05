import logging
import math
import os
import time
from contextlib import nullcontext

import wandb

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.ds_meta import DATASET_GETTERS
from gen_lr_torch import conv_lr_loss
from train import get_default_arguments, set_seed, update_dataset_args, create_model, get_cosine_schedule_with_warmup, \
    interleave, de_interleave, save_checkpoint, test, log_metrics_to_file, init_device
from utils.utils import generate_run_uid
from metrics.misc import AverageMeter

logger = logging.getLogger(__name__)
best_acc = 0


def main():
    parser = get_default_arguments()
    parser.add_argument('--cssl', default=True, type=bool)

    args = parser.parse_args()
    global best_acc

    args.out = args.out + "/" + generate_run_uid(args)

    wandb.init(config=args.__dict__, project=args.wandb_project, sync_tensorboard=True,
               settings=wandb.Settings(start_method="fork"))
    args = init_device(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    update_dataset_args(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset, ds_stats = DATASET_GETTERS[args.dataset](
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
          model, optimizer, ema_model, scheduler, ds_stats)

    wandb.finish()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model, scheduler,
          p_data):
    if args.amp:
        from torch.cuda import amp

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

    mov_avg_buffer_size = 128
    p_model_mov_avg = torch.ones([mov_avg_buffer_size, args.num_classes], device=args.device,
                                 requires_grad=False) / args.num_classes
    p_data = p_data.detach()

    if args.amp:
        scaler = amp.GradScaler()

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        alphas = AverageMeter()

        acc_u_w = AverageMeter()
        acc_u_s = AverageMeter()
        acc_u_s_masked = AverageMeter()

        # Coverage metrics
        strong_validity_005 = AverageMeter()
        strong_validity_01 = AverageMeter()
        strong_validity_025 = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            with amp.autocast() if args.amp else nullcontext():
                try:
                    inputs_x, targets_x = next(labeled_iter)
                except:
                    if args.world_size > 1:
                        labeled_epoch += 1
                        labeled_trainloader.sampler.set_epoch(labeled_epoch)
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = next(labeled_iter)

                # Targets are only used to report debugging stats
                try:
                    (inputs_u_w, inputs_u_s), targets_u = next(unlabeled_iter)
                except:
                    if args.world_size > 1:
                        unlabeled_epoch += 1
                        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), targets_u = next(unlabeled_iter)

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

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label_s = torch.softmax(logits_u_s / args.T, dim=-1)

                with torch.no_grad():
                    pseudo_label_w = torch.softmax(logits_u_w.detach() / args.T, dim=-1)

                    _, targets_u_w = torch.max(pseudo_label_w, dim=-1)
                    # mask = max_probs.ge(args.threshold).float()

                    preds_u_w = torch.max(logits_u_w, dim=-1).indices.to(args.device)
                    preds_u_s = torch.max(logits_u_s, dim=-1).indices.to(args.device)
                    targets_u = targets_u.to(args.device)

                    # Determine p_data and p_model for target normalization
                    p_model = p_model_mov_avg.mean(axis=0)
                    p_model /= p_model.sum()
                    # print("p_model:", p_model)
                    guess = pseudo_label_w * (p_data.to(args.device) + 1e-6) / (p_model.to(args.device) + 1e-6)
                    guess /= torch.sum(guess, dim=1, keepdim=True)

                    # Determine imprecisiation (alphas)
                    max_probs = torch.max(guess, dim=-1).values

                    # Set relaxation alpha
                    relax_alpha = torch.maximum(1. - max_probs, torch.ones_like(max_probs) * 1e-3)

                    possibilities = torch.where(torch.eq(preds_u_w, targets_u), torch.ones_like(targets_u).float(),
                                                relax_alpha)
                    strong_validity_005.update(torch.mean(torch.less_equal(possibilities, 0.05).float()).item(),
                                               n=possibilities.shape[0])
                    strong_validity_01.update(torch.mean(torch.less_equal(possibilities, 0.1).float()).item(),
                                              n=possibilities.shape[0])
                    strong_validity_025.update(torch.mean(torch.less_equal(possibilities, 0.25).float()).item(),
                                               n=possibilities.shape[0])

                # Calculate label relaxation loss
                Lu = conv_lr_loss(pseudo_label_s, F.one_hot(targets_u_w, num_classes=args.num_classes), relax_alpha)

                mask = torch.ones(pseudo_label_w.shape[0]).to(args.device)

                acc_u_w_tmp = torch.mean(torch.eq(preds_u_w, targets_u).float())
                acc_u_s_tmp = torch.mean(torch.eq(preds_u_s, targets_u).float())
                acc_u_s_masked_tmp = torch.sum(torch.eq(preds_u_s, targets_u).float() * mask) / torch.sum(mask)

                loss = Lx + args.lambda_u * Lu

                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                alphas.update(relax_alpha.mean().item())
                acc_u_w.update(acc_u_w_tmp.item())
                acc_u_s.update(acc_u_s_tmp.item())
                acc_u_s_masked.update(acc_u_s_masked_tmp.item())

                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                if args.use_ema:
                    ema_model.update(model)
                model.zero_grad()

                # Update p_model_mov_avg by preds_u_w
                with torch.no_grad():
                    preds_u_w_avg = pseudo_label_w.float().mean(dim=0).to(args.device)
                    p_model_mov_avg = torch.cat((p_model_mov_avg[1:], preds_u_w_avg.unsqueeze(0)), dim=0)

                batch_time.update(time.time() - end)
                end = time.time()
                mask_probs.update(mask.mean().item())
                if not args.no_progress:
                    p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. "
                                          "Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. "
                                          "Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. Alpha: {alpha:.2f}. "
                                          "u_acc (w/s/m): ({acc_u_w:.2f}/{acc_u_s:.2f}/{acc_u_s_masked:.2f}). "
                                          "s_val (0.05/0.1/0.25): ({sval005:.2f}/{sval010:.2f}/{sval025:.2f}).".format(
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
                        alpha=alphas.avg,
                        acc_u_w=acc_u_w.avg,
                        acc_u_s=acc_u_s.avg,
                        acc_u_s_masked=acc_u_s_masked.avg,
                        sval005=strong_validity_005.avg,
                        sval010=strong_validity_01.avg,
                        sval025=strong_validity_025.avg,
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
            test_loss, test_acc, test_ece, test_brier = test(args, test_loader, test_model, epoch,
                                                             scoring_pref=scoring_pref)

            args.writer.add_scalar('train/train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/acc_u_w', acc_u_w.avg, epoch)
            args.writer.add_scalar('train/acc_u_s', acc_u_s.avg, epoch)
            args.writer.add_scalar('train/acc_u_s_masked', acc_u_s_masked.avg, epoch)
            args.writer.add_scalar('train/alphas', alphas.avg, epoch)
            args.writer.add_scalar('train/sval_005', strong_validity_005.avg, epoch)
            args.writer.add_scalar('train/sval_01', strong_validity_01.avg, epoch)
            args.writer.add_scalar('train/sval_025', strong_validity_025.avg, epoch)
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
                       "mask": mask_probs.avg, "alphas": alphas.avg, "acc_u_w": acc_u_w.avg, "acc_u_s": acc_u_s.avg,
                       "acc_u_s_masked": acc_u_s_masked.avg})

            # Log metrics to file
            log_metrics_to_file(args, epoch, test_acc, test_ece, test_brier, scoring_pref=scoring_pref)

    if args.local_rank in [-1, 0]:
        args.writer.close()


if __name__ == '__main__':
    main()
