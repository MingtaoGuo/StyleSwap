'''
This is a simplified training code of StyleSwap. It achieves comparable performance as in the paper.

@Created by rosinality and yangxy

@Modified by Mingtao Guo (gmt798714378@hotmail.com)
'''
import argparse
from ctypes import resize
import math
from queue import Full
import random
import os
import cv2
import glob
from tqdm import tqdm

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils

import __init_paths
from training.data_loader.dataset_face import FaceDataset
from face_model.gpen_model import FullGenerator, Discriminator, VGG19
from face_model.arcface import iresnet50

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):    
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def train(args, loader, generator, discriminator, arcface, vgg19, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(0, args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator
 
    accum = 0.5 ** (32 / (10 * 1000))
    f = open("loss.txt", "w")
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        target, source, mask, same = next(loader)
        target = target.to(device)
        source = source.to(device)
        mask = mask.to(device)
        same = same.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        with torch.no_grad():
            z_id = arcface(F.interpolate(source, [143, 143], mode="bilinear")[..., 15:127, 15:127])
        fake_img, _, _ = generator(target, z_id.detach())
        fake_pred, _ = discriminator(fake_img.detach())

        real_pred, real_feats = discriminator(target)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            target.requires_grad = True
            real_pred, _ = discriminator(target)
            r1_loss = d_r1_loss(real_pred, target)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_img, _, fake_mask = generator(target, z_id.detach())
        fake_pred, fake_feats = discriminator(fake_img)
        # ------------- adv loss -------------
        adv_loss = g_nonsaturating_loss(fake_pred)
        # ------------- id loss --------------
        fake_z_id = arcface(F.interpolate(fake_img, [143, 143], mode="bilinear")[..., 15:127, 15:127])
        id_loss = (1 - torch.cosine_similarity(fake_z_id, z_id.detach())).mean()
        # -------feature matching loss -------
        fm_loss = 0
        for fm_r, fm_f in zip(real_feats, fake_feats):
            fm_loss += F.l1_loss(fm_r.detach(), fm_f).mean()
        # ------------ rec loss --------------
        rec_pix = (torch.abs(fake_img - target).mean([1, 2, 3]) * same).sum() / (same.sum() + 1e-8)
        rec_lpip = 0
        weights_vgg = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for vgg_f, vgg_r, vgg_w in zip(vgg19(fake_img), vgg19(target), weights_vgg):
            rec_lpip += vgg_w * (torch.abs(vgg_f - vgg_r).mean([1, 2, 3]) * same).sum() / (same.sum() + 1e-8)
        rec_loss = rec_pix + rec_lpip
        # ------------ mask loss -------------
        bce_loss = F.binary_cross_entropy(fake_mask, mask).mean()
        # bce_loss = F.l1_loss(fake_mask, mask).mean()
        # ----------- total loss -------------
        g_loss = adv_loss + 10 * id_loss + bce_loss + 10 * fm_loss + 10 * rec_loss
        loss_dict['g'] = g_loss
        loss_dict['adv'] = adv_loss
        loss_dict['id'] = id_loss
        loss_dict['fm'] = fm_loss
        loss_dict['rec'] = rec_loss
        loss_dict['bce'] = bce_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            fake_img, latents, _ = generator(target, z_id, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        adv_loss_val = loss_reduced['adv'].mean().item()
        id_loss_val = loss_reduced['id'].mean().item()
        fm_loss_val = loss_reduced['fm'].mean().item()
        rec_loss_val = loss_reduced['rec'].mean().item()
        bce_loss_val = loss_reduced['bce'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; adv: {adv_loss_val:.4f}; id: {id_loss_val:.4f}; fm: {fm_loss_val:.4f}; rec: {rec_loss_val:.4f}; bce: {bce_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )
            
            if i % args.save_freq == 0:
                f.write(f'itr: {i}; d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; adv: {adv_loss_val:.4f}; id: {id_loss_val:.4f}; fm: {fm_loss_val:.4f}; rec: {rec_loss_val:.4f}; bce: {bce_loss_val:.4f}; r1: {r1_val:.4f}; \n')
                f.flush()
                with torch.no_grad():
                    g_ema.eval()
                    sample, _, fake_mask = g_ema(target, z_id)
                    sample = sample * fake_mask + target * (1 - fake_mask)
                    sample = torch.cat((target, sample, (torch.cat([fake_mask, fake_mask, fake_mask], dim=1)-0.5)/0.5, source), 0) 
                    utils.save_image(
                        sample,
                        f'{args.sample}/{str(i).zfill(6)}.png',
                        nrow=args.batch,
                        normalize=True,
                        range=(-1, 1),
                    )

            if i and i % 5000 == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'{args.ckpt}/{str(i).zfill(6)}.pth',
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, default="/data1/GMT/Dataset/FFHQ256/")
    parser.add_argument('--mask_path', type=str, default="/data1/GMT/Dataset/FFHQ128parsing/")
    parser.add_argument('--base_dir', type=str, default='./')
    parser.add_argument('--arcface', type=str, default='saved_models/backbone.pth')
    parser.add_argument('--iter', type=int, default=4000000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--narrow', type=float, default=1.0)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='ckpts')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--sample', type=str, default='sample')
    parser.add_argument('--val_dir', type=str, default='val')

    args = parser.parse_args()

    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.sample, exist_ok=True)

    device = 'cuda'

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    g_ema = FullGenerator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, narrow=args.narrow, device=device
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    arcface = iresnet50().to(device)
    arcface.eval()
    arcface.load_state_dict(torch.load(args.arcface))

    vgg19 = VGG19().to(device)
    vgg19.eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.pretrain is not None:
        print('load model:', args.pretrain)
        
        ckpt = torch.load(args.pretrain)

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])
            
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        arcface = nn.parallel.DistributedDataParallel(
            arcface,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        vgg19 = nn.parallel.DistributedDataParallel(
            vgg19,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )        

    dataset = FaceDataset(args.img_path, args.mask_path, resolution=args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(args, loader, generator, discriminator, arcface, vgg19, g_optim, d_optim, g_ema, device)
   
