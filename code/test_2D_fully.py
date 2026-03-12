import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/ACDC", help="dataset root")
parser.add_argument("--exp", type=str, default="ACDC/Fully_Supervised", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--num_classes", type=int, default=4, help="num classes")
parser.add_argument("--labeled_num", type=int, default=3, help="labeled data")


def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    # both empty -> perfect
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    # one empty -> zero overlap (avoid medpy crash)
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0

    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(f"{FLAGS.root_path}/data/{case}.h5", "r")
    image = h5f["image"][:]
    label = h5f["label"][:]

    prediction = np.zeros_like(label)

    net.eval()
    with torch.no_grad():
        for ind in range(image.shape[0]):
            slice_ = image[ind, :, :]
            x, y = slice_.shape

            slice_rs = zoom(slice_, (256 / x, 256 / y), order=0)

            inp = torch.from_numpy(slice_rs).unsqueeze(0).unsqueeze(0).float().cuda()

            outputs = net(inp)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]  # take main output if multi-output

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy()

            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # Save prediction (optional)
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        img_itk.SetSpacing((1, 1, 10))
        prd_itk.SetSpacing((1, 1, 10))
        lab_itk.SetSpacing((1, 1, 10))

        sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))

    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(f"{FLAGS.root_path}/test.list", "r") as f:
        image_list = sorted([item.strip().split(".")[0] for item in f.readlines()])

    snapshot_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}_labeled/{FLAGS.model}"
    test_save_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}_labeled/{FLAGS.model}_predictions"

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path, exist_ok=True)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes).cuda()

    save_mode_path = os.path.join(snapshot_path, f"{FLAGS.model}_best_model.pth")
    if not os.path.exists(save_mode_path):
        raise FileNotFoundError(f"Model weight not found: {save_mode_path}")

    net.load_state_dict(torch.load(save_mode_path, map_location="cuda"))
    print(f"init weight from {save_mode_path}")
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

    avg_metric = [
        first_total / len(image_list),
        second_total / len(image_list),
        third_total / len(image_list),
    ]
    return avg_metric


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    metric_res = Inference(FLAGS)
    print("Per class metric:", metric_res)
    print("Mean metric:", (metric_res[0] + metric_res[1] + metric_res[2]) / 3)