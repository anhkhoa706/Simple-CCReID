import os
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from apex import amp

from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from train import train_cal, train_cal_with_memory
from test import test, test_prcc

VID_DATASET = ['ccvid']

def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config

def analyze_pid2clothes(pid2clothes):
    num_ids, num_clothes = pid2clothes.shape
    print(f"pid2clothes shape: {pid2clothes.shape}")
    
    clothes_per_id = pid2clothes.sum(dim=1)
    ids_with_1_cloth = (clothes_per_id == 1).sum().item()
    ids_with_2_or_more = (clothes_per_id >= 2).sum().item()
    
    print(f"- Total identities: {num_ids}")
    print(f"- Total clothes   : {num_clothes}")
    print(f"- IDs with 1 cloth: {ids_with_1_cloth}")
    print(f"- IDs with ≥2     : {ids_with_2_or_more}")
    print(f"- Min clothes/id  : {clothes_per_id.min().item()}")
    print(f"- Max clothes/id  : {clothes_per_id.max().item()}")
    print(f"- Avg clothes/id  : {clothes_per_id.float().mean().item():.2f}")

def main(config):
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)

    pid2clothes = torch.from_numpy(dataset.pid2clothes)

    model, classifier, clothes_classifier = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)
    criterion_cla, criterion_pair, criterion_clothes, criterion_adv = build_losses(config, dataset.num_train_clothes)

    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(parameters, lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        optimizer_cc = optim.AdamW(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if config.LOSS.CAL == 'calwithmemory':
            criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        else:
            clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    device = torch.device("cuda")
    model = model.to(device)
    classifier = classifier.to(device)
    if config.LOSS.CAL == 'calwithmemory':
        criterion_adv = criterion_adv.to(device)
    else:
        clothes_classifier = clothes_classifier.to(device)

    if config.TRAIN.AMP:
        print("Using automatic mixed precision (AMP) for training")
        [model, classifier], optimizer = amp.initialize([model, classifier], optimizer, opt_level="O1")
        if config.LOSS.CAL != 'calwithmemory':
            clothes_classifier, optimizer_cc = amp.initialize(clothes_classifier, optimizer_cc, opt_level="O1")
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return
    
    # Tets pid2clothes before training
    analyze_pid2clothes(pid2clothes)
    
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start_train_time = time.time()
        if config.LOSS.CAL == 'calwithmemory':
            train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, criterion_adv, optimizer, trainloader, pid2clothes)
        else:
            train_cal(config, epoch, model, classifier, clothes_classifier, criterion_cla, criterion_pair, criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes)

        train_time += round(time.time() - start_train_time)

        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                rank1 = test(config, model, queryloader, galleryloader, dataset)
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'clothes_classifier_state_dict': criterion_adv.state_dict() if config.LOSS.CAL == 'calwithmemory' else clothes_classifier.state_dict(),
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

if __name__ == '__main__':
    config = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    set_seed(config.SEED)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create new run subfolder
    run_output_dir = osp.join(config.OUTPUT, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # Allow modifying the config
    config.defrost()
    config.OUTPUT = run_output_dir
    config.freeze()  # Optional: freeze again after setting

    # Setup logger
    log_type = 'test' if config.EVAL_MODE else 'train'
    output_file = osp.join(run_output_dir, f'log_{log_type}_{timestamp}.log')
    logger = get_logger(output_file, 0, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config)