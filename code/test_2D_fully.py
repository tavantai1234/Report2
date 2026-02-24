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
parser.add_argument("--root_path", type=str, default="../data/ACDC")
parser.add_argument("--exp", type=str, default="ACDC/Fully_Supervised")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_num", type=int, default=3)


# -------------------------------------------------
# SAFE METRIC (không crash khi mask rỗng)
# -------------------------------------------------
def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 0.0, 0.0

    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


# -------------------------------------------------
# TEST ONE VOLUME
# -------------------------------------------------
def test_single_volume(case, net, test_save_path, FLAGS):

    h5f = h5py.File(f"{FLAGS.root_path}/data/{case}.h5", "r")
    image = h5f["image"][:]
    label = h5f["label"][:]
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape

        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input_tensor = (
            torch.from_numpy(slice)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .cuda()
        )

        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)

            # xử lý multi-output model
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            out = torch.argmax(
                torch.softmax(outputs, dim=1), dim=1
            ).squeeze(0)

            out = out.cpu().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # save results
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

    img_itk.SetSpacing((1, 1, 10))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")

    return first_metric, second_metric, third_metric


# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
def Inference(FLAGS):

    with open(f"{FLAGS.root_path}/test.list", "r") as f:
        image_list = f.readlines()

    image_list = sorted(
        [item.replace("\n", "").split(".")[0] for item in image_list]
    )

    snapshot_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}_labeled/{FLAGS.model}"
    test_save_path = f"../model/{FLAGS.exp}_{FLAGS.labeled_num}_labeled/{FLAGS.model}_predictions/"

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(
        net_type=FLAGS.model,
        in_chns=1,
        class_num=FLAGS.num_classes,
    ).cuda()

    save_mode_path = os.path.join(
        snapshot_path, f"{FLAGS.model}_best_model.pth"
    )

    net.load_state_dict(
        torch.load(save_mode_path, map_location="cuda")
    )

    print(f"Loaded weight from {save_mode_path}")
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0

    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS
        )
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
    metric = Inference(FLAGS)
    print("Per class metric:", metric)
    print("Mean metric:", (metric[0] + metric[1] + metric[2]) / 3)