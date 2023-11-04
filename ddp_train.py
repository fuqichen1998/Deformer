import argparse
import os
import random
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import (
    Monitor,
    get_dataset,
    get_network,
    print_args,
    save_args,
    ddp_load_checkpoint,
    save_checkpoint,
)
from utils.epoch import single_epoch
from utils.options import add_opts

import wandb

from networks.motion_discriminator import MotionDiscriminator


def main(args):
    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # get ddp rank
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # create exp result dir
    os.makedirs(args.host_folder, exist_ok=True)
    # Initialize model
    model = get_network(args, ddp=True)

    print("Using {} GPUs !".format(torch.cuda.device_count()))
    model.to(local_rank)
    start_epoch = 0
    if dist.get_rank() == 0 and args.resume:
        start_epoch = ddp_load_checkpoint(model, resume_path=args.resume, strict=False)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not args.evaluate:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters()]},
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma
        )

        if args.motion_discrimination:
            motion_discriminator = MotionDiscriminator(
                rnn_size=1024,
                input_size=46,
                num_layers=2,
                output_size=1,
                feature_pool="attention",
                attention_size=1024,
                attention_layers=3,
                attention_dropout=0.2,
            )
            motion_discriminator.cuda()
            motion_dis_optimizer = torch.optim.AdamW(
                motion_discriminator.parameters(),
                lr=args.motion_dis_lr,
                weight_decay=args.motion_dis_weight_decay,
            )
            motion_dis_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                motion_dis_optimizer,
                args.motion_dis_lr_decay_step,
                gamma=args.motion_dis_lr_decay_gamma,
            )
        else:
            motion_discriminator = None
            motion_dis_optimizer = None
            motion_dis_lr_scheduler = None

        for _ in range(start_epoch):
            scheduler.step()
            if motion_dis_lr_scheduler is not None:
                motion_dis_lr_scheduler.step()
        train_dat = get_dataset(args, mode="train")
        print("training dataset size: {}".format(len(train_dat)))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dat)
        train_loader = torch.utils.data.DataLoader(
            train_dat,
            batch_size=args.train_batch,
            num_workers=int(args.workers),
            pin_memory=True,
            drop_last=False,
            sampler=train_sampler,
        )
        monitor = Monitor(hosting_folder=args.host_folder)

    else:
        args.epochs = start_epoch + 1

    # Initialize validation dataset
    val_dat = get_dataset(args, mode="val")
    print("evaluation dataset size: {}".format(len(val_dat)))
    val_loader = torch.utils.data.DataLoader(
        val_dat,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=False,
    )

    for epoch in range(start_epoch, args.epochs):
        train_dict = {}
        if not args.evaluate:
            train_loader.sampler.set_epoch(epoch)
            print("Using lr {}".format(optimizer.param_groups[0]["lr"]))
            train_avg_meters = single_epoch(
                loader=train_loader,
                model=model,
                optimizer=optimizer,
                motion_dis_model=motion_discriminator,
                motion_dis_optimizer=motion_dis_optimizer,
                motion_dis_loss_weight=args.motion_dis_loss_weight,
                epoch=epoch,
                save_path=args.host_folder,
                train=True,
                save_results=False,
                use_cuda=args.use_cuda,
            )

            train_dict = {
                meter_name: meter.avg
                for (meter_name, meter) in train_avg_meters.average_meters.items()
            }
            monitor.log_train(epoch + 1, train_dict)
            if local_rank == 0:
                wandb.log(
                    {**train_dict, "lr": optimizer.param_groups[0]["lr"]}
                )  # add lr

        # Evaluate on validation set
        if args.evaluate or (epoch + 1) % args.test_freq == 0:
            with torch.no_grad():
                single_epoch(
                    loader=val_loader,
                    model=model,
                    epoch=epoch if not args.evaluate else None,
                    optimizer=None,
                    save_path=args.host_folder,
                    train=False,
                    save_results=args.save_results,
                    use_cuda=args.use_cuda,
                    indices_order=val_dat.jointsMapSimpleToMano
                    if hasattr(val_dat, "jointsMapSimpleToMano")
                    else None,
                )

        if not args.evaluate:
            if dist.get_rank() == 0 and (epoch + 1) % args.snapshot == 0:
                print(f"save epoch {epoch+1} checkpoint to {args.host_folder}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.module.state_dict(),
                    },
                    checkpoint=args.host_folder,
                    filename=f"checkpoint_{epoch+1}.pth.tar",
                )
                if motion_discriminator is not None:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": motion_discriminator.state_dict(),
                        },
                        checkpoint=args.host_folder,
                        filename=f"motion_dis_checkpoint_{epoch+1}.pth.tar",
                    )

            scheduler.step()
            if motion_dis_lr_scheduler is not None:
                motion_dis_lr_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deformer DDP training")
    add_opts(parser)

    args = parser.parse_args()
    args.save_results = True

    local_rank = args.local_rank
    if local_rank == 0:
        print_args(args)
        save_args(args, save_folder=args.host_folder, opt_prefix="option")
        if args.evaluate or "debug" in args.run_name:
            run = wandb.init(project="deformer", mode="disabled")
        else:
            run = wandb.init(project="deformer", mode="online")
        if args.run_name is not None:
            wandb.run.name = args.run_name
        wandb.config.update(args)
    if local_rank == 0:
        with run:
            main(args)
    else:
        main(args)
    print("All done !")
