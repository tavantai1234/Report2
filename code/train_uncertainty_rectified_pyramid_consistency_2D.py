import argparse
import copy
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from utils import losses, ramps
from val_2D import test_single_volume_ds
from networks.net_factory import net_factory


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Uncertainty_Rectified_Pyramid_Consistency', help='experiment_name')
parser.add_argument('--model', type=str, default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iterations to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')

# costs (stage 1 tuned)
parser.add_argument('--consistency', type=float, default=0.3, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=1000.0, help='consistency_rampup')

# EMA teacher (stage 2)
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
# EMA consistency loss weight (optional). Keep small for stability.
parser.add_argument('--ema_consistency', type=float, default=0.1, help='extra ema consistency loss weight')

args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120, "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError("Unknown dataset for patients_to_slices")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(step):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(step, args.consistency_rampup)


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / (global_step + 1.0), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1.0 - alpha))


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()

    # EMA teacher model (safe setup)
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    ema_model.load_state_dict(model.state_dict())  # IMPORTANT: start from student weights
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    # IMPORTANT: start below any possible dice so we always save at least once if val runs
    best_performance = -1.0

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()

            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # supervised (baseline)
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs], label_batch[:args.labeled_bs].long())

            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux1 = dice_loss(outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux2 = dice_loss(outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux3 = dice_loss(outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = (
                loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3 +
                loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3
            ) / 8.0

            # URPC consistency (baseline)
            preds = (outputs_soft + outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft) / 4.0

            variance_main = torch.sum(
                kl_distance(torch.log(outputs_soft[args.labeled_bs:]), preds[args.labeled_bs:]),
                dim=1, keepdim=True
            )
            exp_variance_main = torch.exp(-variance_main)

            consistency_dist_main = (preds[args.labeled_bs:] - outputs_soft[args.labeled_bs:]) ** 2

            consistency_loss_main = torch.mean(consistency_dist_main * exp_variance_main) / \
                                    (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

            # EMA teacher consistency (NEW, small)
            with torch.no_grad():
                t_out, t_aux1, t_aux2, t_aux3 = ema_model(volume_batch)
                t_soft = torch.softmax(t_out, dim=1)

            ema_cons_loss = F.mse_loss(
                outputs_soft[args.labeled_bs:],  # student on unlabeled
                t_soft[args.labeled_bs:]         # teacher on unlabeled
            )

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = supervised_loss + consistency_weight * consistency_loss_main + \
                   (args.ema_consistency * consistency_weight) * ema_cons_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LR schedule (baseline)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # IMPORTANT: increase step BEFORE updating EMA
            iter_num += 1

            # EMA update
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/supervised_loss', supervised_loss.item(), iter_num)
            writer.add_scalar('info/urpc_consistency_loss', consistency_loss_main.item(), iter_num)
            writer.add_scalar('info/ema_consistency_loss', ema_cons_loss.item(), iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            if iter_num % 200 == 0:
                # Validate using STUDENT to keep test pipeline unchanged
                model.eval()
                metric_list = 0.0
                for _, val_batch in enumerate(valloader):
                    metric_i = test_single_volume_ds(
                        val_batch["image"], val_batch["label"], model, classes=num_classes
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                performance = np.mean(metric_list, axis=0)[0]  # mean dice
                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance

                    save_best = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
                    torch.save(model.state_dict(), save_best)

                    # Optional: also save EMA best (not used by default test unless you change it)
                    save_best_ema = os.path.join(snapshot_path, f'{args.model}_best_model_ema.pth')
                    torch.save(ema_model.state_dict(), save_best_ema)

                model.train()

            if iter_num % 3000 == 0:
                # save checkpoint (baseline style)
                torch.save(model.state_dict(), os.path.join(snapshot_path, f'iter_{iter_num}.pth'))
                torch.save(ema_model.state_dict(), os.path.join(snapshot_path, f'iter_{iter_num}_ema.pth'))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    # Fallback save: ensure test ALWAYS finds best_model.pth
    final_best = os.path.join(snapshot_path, f'{args.model}_best_model.pth')
    if not os.path.exists(final_best):
        print("⚠️ No best model saved during validation. Saving final student model as best.")
        torch.save(model.state_dict(), final_best)

    final_best_ema = os.path.join(snapshot_path, f'{args.model}_best_model_ema.pth')
    if not os.path.exists(final_best_ema):
        torch.save(ema_model.state_dict(), final_best_ema)

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # keep baseline behavior: copy code snapshot
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)