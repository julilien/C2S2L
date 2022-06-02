import argparse
import logging
import math
import os
import random
import shutil
import time
import wandb

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.ds_meta import DATASET_GETTERS
from metrics.ece import ece_score, brier_score
from utils.utils import generate_run_uid
from metrics.misc import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def get_default_arguments():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'tinyimagenet', 'svhn_extra'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true", default=True,
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='results',
                        help='directory to output the results')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--validation_scoring', default=False, type=bool)

    parser.add_argument('--dataset_dir', type=str, default="./data")
    parser.add_argument('--wandb_project', type=str, default="CCSSL")

    return parser


def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))
    return model


def update_dataset_args(args):
    if args.dataset in ['cifar10', 'svhn', 'svhn_extra']:
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'stl10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            # This is different than the one from FixMatch
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        if args.arch == 'wideresnet':
            args.model_depth = 37
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    else:
        raise ValueError("Unrecognized dataset {}...".format(args.dataset))


def init_device(args):
    if args.gpu_id == -1:
        device = torch.device('cpu')
        args.world_size = 1
        args.n_gpu = 0
    elif args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device
    return args


def init_run(args):
    args.out = args.out + "/" + generate_run_uid(args)

    wandb.init(config=args.__dict__, project=args.wandb_project, sync_tensorboard=True)
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

    return args


def main():
    parser = get_default_arguments()
    parser.add_argument('--da', default=False, type=bool, help="Distribution alignment")

    args = parser.parse_args()
    global best_acc

    args = init_run(args)
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
          model, optimizer, ema_model, scheduler, ds_stats)

    wandb.finish()


def log_metrics_to_file(args, epoch, test_acc, test_ece, test_brier, scoring_pref="test"):
    logfile = open('%s/{}_acc.txt'.format(scoring_pref) % (args.out), 'a')
    print("[epoch: %d] %.4f" % (epoch + 1, test_acc), file=logfile)
    logfile.close()

    logfile = open('%s/{}_ece.txt'.format(scoring_pref) % (args.out), 'a')
    print("[epoch: %d] %.4f" % (epoch + 1, test_ece), file=logfile)
    logfile.close()

    logfile = open('%s/{}_brier.txt'.format(scoring_pref) % (args.out), 'a')
    print("[epoch: %d] %.4f" % (epoch + 1, test_brier), file=logfile)
    logfile.close()


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, p_data):
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

    if args.da:
        mov_avg_buffer_size = 128
        p_model_mov_avg = torch.ones([mov_avg_buffer_size, args.num_classes], device=args.device,
                                     requires_grad=False) / args.num_classes
        p_data = p_data.detach()

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
                (inputs_u_w, inputs_u_s), targets_u_orig = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), targets_u_orig = unlabeled_iter.next()

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

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            if not args.da:
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            else:
                # Determine p_data and p_model for target normalization
                p_model = p_model_mov_avg.mean(axis=0)
                p_model /= p_model.sum()

                guess = pseudo_label * (p_data.to(args.device) + 1e-6) / (p_model.to(args.device) + 1e-6)
                guess /= torch.sum(guess, dim=1, keepdim=True)
                max_probs, targets_u = torch.max(guess, dim=-1)

            mask = max_probs.ge(args.threshold).float()

            targets_u = targets_u.to(args.device)
            targets_u_orig = targets_u_orig.to(args.device)

            with torch.no_grad():
                preds_u_w = torch.max(logits_u_w, dim=-1).indices.to(args.device)
                preds_u_s = torch.max(logits_u_s, dim=-1).indices.to(args.device)

                acc_u_w_tmp = torch.mean(torch.eq(preds_u_w, targets_u_orig).float())
                acc_u_s_tmp = torch.mean(torch.eq(preds_u_s, targets_u_orig).float())
                acc_u_s_masked_tmp = torch.sum(torch.eq(preds_u_s, targets_u_orig).float() * mask) / torch.sum(mask)

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

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

            # Update p_model_mov_avg by preds_u_w
            if args.da:
                with torch.no_grad():
                    preds_u_w_avg = pseudo_label.float().mean(dim=0).to(args.device)
                    p_model_mov_avg = torch.cat((p_model_mov_avg[1:], preds_u_w_avg.unsqueeze(0)), dim=0)

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


def test(args, test_loader, model, epoch, scoring_pref="test"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ece = AverageMeter()
    brier = AverageMeter()

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

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "{score_pref} Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. "
                    "top1: {top1:.2f}. top5: {top5:.2f}. ece: {ece:.2f}. brier: {brier:.2f}.".format(
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
    return losses.avg, top1.avg, ece.avg, brier.avg


if __name__ == '__main__':
    main()
