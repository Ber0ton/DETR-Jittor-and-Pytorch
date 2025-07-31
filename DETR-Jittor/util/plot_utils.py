import jittor as jt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        # 使用Jittor的加载函数
        data = jt.load(f)
        
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs if hasattr(data['params'], 'recThrs') else data['params']['recThrs']
        scores = data['scores']
        
        # take precision for all classes, all areas and 100 detections
        # 在Jittor中，张量索引和均值计算的语法基本相同
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        
        # 将Jittor张量转换为Python标量用于打印
        prec_val = float(prec.data) if hasattr(prec, 'data') else float(prec)
        rec_val = float(rec.data) if hasattr(rec, 'data') else float(rec)
        scores_mean = float(scores.mean().data) if hasattr(scores.mean(), 'data') else float(scores.mean())
        
        print(f'{naming_scheme} {name}: mAP@50={prec_val * 100: 05.1f}, ' +
              f'score={scores_mean:0.3f}, ' +
              f'f1={2 * prec_val * rec_val / (prec_val + rec_val + 1e-8):0.3f}'
              )
        
        # 转换为numpy数组用于matplotlib绘图
        recall_np = recall.numpy() if hasattr(recall, 'numpy') else np.array(recall)
        precision_np = precision.numpy() if hasattr(precision, 'numpy') else np.array(precision)
        scores_np = scores.numpy() if hasattr(scores, 'numpy') else np.array(scores)
        
        axs[0].plot(recall_np, precision_np, c=color)
        axs[1].plot(recall_np, scores_np, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def jt_to_numpy(tensor):
    """
    辅助函数：将Jittor张量转换为numpy数组
    """
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    elif hasattr(tensor, 'data'):
        return np.array(tensor.data)
    else:
        return np.array(tensor)