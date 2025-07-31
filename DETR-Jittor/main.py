import argparse, datetime, json, random, time
from pathlib import Path

import numpy as np
import jittor as jt
jt.flags.log_silent = 1
jt.flags.auto_mixed_precision_level = 2 
jt.cudnn.set_max_workspace_ratio(0.0)

import datasets
import types
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch         
from models import build_model
from jittor.dataset import DataLoader                   

def jt_save(obj, path):
    jt.save(obj, str(path))



# ---------------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector (Jittor)', add_help=False)
    # ---------- 超参数 ------------------------------------------------------
    parser.add_argument('--lr', default=0.125e-4, type=float)
    parser.add_argument('--lr_backbone', default=0.125e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    # ---------- 模型 --------------------------------------------------------
    parser.add_argument('--frozen_weights', default=None, type=str)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', choices=('sine', 'learned'))
    # ---------- Transformer -------------------------------------------------
    parser.add_argument('--enc_layers', default=6,  type=int)
    parser.add_argument('--dec_layers', default=6,  type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout',  default=0.1, type=float)
    parser.add_argument('--nheads',   default=8,   type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # ---------- Segmentation -----------------------------------------------
    parser.add_argument('--masks', action='store_true')
    # ---------- Matcher / Loss ---------------------------------------------
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox',  default=5, type=float)
    parser.add_argument('--set_cost_giou',  default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef',       default=0.1, type=float)
    # ---------- Dataset -----------------------------------------------------
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    # ---------- 运行环境 ----------------------------------------------------
    parser.add_argument('--output_dir', default='output/7_31_test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed',  default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    # ---------- 分布式 -------------------------------------------------------
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    # ---------- 打印频率 -----------------------------------------------------
    parser.add_argument('--print_freq', default=125, type=int)
    return parser
# ---------------------------------------------------------------------------

def save_eval_summary(epoch, stats, file_path: Path, mode="val"):
    with file_path.open("a") as f:
        prefix = f"[{mode}] epoch={epoch:>3d} " if isinstance(epoch, int) else f"[{mode}] "
        if "coco_eval_bbox" in stats:
            ap = stats["coco_eval_bbox"]
            f.write(f"{prefix}AP={ap[0]:.8f} | AP50={ap[1]:.8f} | AP75={ap[2]:.8f}\n")
        else:
            f.write(prefix + json.dumps(stats) + "\n")

# ---------------------------------------------------------------------------
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    # ----------- 随机种子 ---------------------------------------------------
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)

    # ----------- 设备 -------------------------------------------------------
    device = jt.device(args.device)

    # ----------- 构建模型 / 损失 / 后处理 -----------------------------------
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # ----------- 优化器 & LR 调度 ------------------------------------------
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = jt.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # 简易 StepLR
    def step_lr(ep):
        factor = ep // args.lr_drop
        return args.lr * (0.1 ** factor)
    # 调度器实现为 lambda，每个 epoch 手动设置
    for g in optimizer.param_groups:
        lr_val = (g.get("lr")                # Jittor ≥1.4 常用
                  or g.get("learning_rate")  # 某些旧版本
                  or args.lr)                # 保险兜底
        g["initial_lr"] = lr_val             # 一定写一个 float 进去

    train_set = build_dataset("train", args)
    train_set.batchify_fn = utils.detr_batchify_fn   # ① 先加进去
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )                                               # ② 再 DataLoader
    
    val_set = build_dataset("val", args)
    val_set.batchify_fn = utils.detr_batchify_fn
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )


    # COCO api（用于评估）----------------------------------------------------
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds  = get_coco_api_from_dataset(coco_val)
    else:
        base_ds  = get_coco_api_from_dataset(val_set)

    # ----------- 可选加载冻结权重 ------------------------------------------
    if args.frozen_weights:
        cp = jt.load(args.frozen_weights)
        model.detr.load_state_dict(cp["model"])

    output_dir = Path(args.output_dir)
    if args.resume:
        cp = jt.load(args.resume)
        model.load_state_dict(cp["model"])
        if not args.eval and all(k in cp for k in ("optimizer", "epoch")):
            optimizer.load_state_dict(cp["optimizer"])
            args.start_epoch = int(cp["epoch"]) + 1

    # ----------- 只评估 -----------------------------------------------------
    if args.eval:
        test_stats, coco_eval = evaluate(model, criterion, postprocessors,
                                         val_loader, base_ds, device,
                                         args.output_dir, print_freq=args.print_freq)
        if args.output_dir and utils.is_main_process():
            jt_save(coco_eval.coco_eval["bbox"].eval, output_dir / "eval.pkl")
            save_eval_summary("eval", test_stats, output_dir / "eval_summary.txt", mode="test")
        return

    # ----------- 训练循环 ---------------------------------------------------
    print("Start training")
    start_time = time.time()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(args.start_epoch, args.epochs):

        # 更新学习率（简单 StepLR）
        for g in optimizer.param_groups:
            base_lr = g.get("initial_lr", args.lr)
            factor  = epoch // args.lr_drop
            new_lr  = base_lr * (0.1 ** factor)
        
            if "lr" in g:
                g["lr"] = new_lr
            else:                     # 旧字段名
                g["learning_rate"] = new_lr

        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, args.clip_max_norm, print_freq=args.print_freq)

        # 保存 checkpoint ---------------------------------------------------
        if args.output_dir and utils.is_main_process():
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            jt_save(ckpt, output_dir / 'checkpoint.pkl')
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 50 == 0:
                jt_save(ckpt, output_dir / f'checkpoint{epoch:04}.pkl')

        # 评估 --------------------------------------------------------------
        test_stats, coco_eval = evaluate(
            model, criterion, postprocessors,
            val_loader, base_ds, device,
            args.output_dir, print_freq=args.print_freq)

        save_eval_summary(epoch, test_stats, output_dir / "eval_summary.txt")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}":  v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if coco_eval and "bbox" in coco_eval.coco_eval:
                (output_dir / "eval").mkdir(exist_ok=True)
                jt_save(coco_eval.coco_eval["bbox"].eval, output_dir / "eval" / "latest.pkl")

    # ----------- 结束 -------------------------------------------------------
    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR (Jittor) Training & Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
