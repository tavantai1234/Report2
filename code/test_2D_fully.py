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
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/ACDC",
    help="Path to dataset root",
)
parser.add_argument(
    "--exp",
    type=str,
    default="ACDC/Fully_Supervised",
    help="Experiment name",
)
parser.add_argument(
    "--model",
    type=str,
    default="unet",
    help="Model name",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=4,
    help="Number of output classes",
)
parser.add_argument(
    "--labeled_num",
    type=int,
    default=3,
    help="Number of labeled samples",
)


def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 0.0

    if pred.sum() == 0 and gt.sum() != 0:
        return 0.0, 0.0, 0.0

    if pred.sum() != 0 and gt.sum() == 0:
        return 0.0, 0.0, 0.0

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd95, asd


def get_main_output(outputs):
    """
    Return the main logits tensor from the network output.

    Some models return:
    - a single tensor
    - a tuple/list like (main_output, aux1, aux2, aux3)

    This helper removes the need for notebook-side monkey patching.
    """
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def test_single_volume(case, net, test_save_path, FLAGS):
    h5_path = os.path.join(FLAGS.root_path, "data", f"{case}.h5")
    with h5py.File(h5_path, "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]

    prediction = np.zeros_like(label)

    net.eval()
    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]

        slice_resized = zoom(slice_img, (256 / x, 256 / y), order=0)
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            outputs = net(input_tensor)
            out_main = get_main_output(outputs)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().numpy()

        pred = zoom(out, (x / 256, y / 256), order=0)
        prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))

    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))

    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    test_list_path = os.path.join(FLAGS.root_path, "test.list")
    with open(test_list_path, "r") as f:
        image_list = f.readlines()

    image_list = sorted([item.replace("\n", "").split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model
    )
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model
    )

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(
        net_type=FLAGS.model,
        in_chns=1,
        class_num=FLAGS.num_classes,
    )

    save_mode_path = os.path.join(snapshot_path, "{}_best_model.pth".format(FLAGS.model))
    checkpoint = torch.load(save_mode_path, map_location="cuda")
    net.load_state_dict(checkpoint)

    print("init weight from {}".format(save_mode_path))
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
    metric_result = Inference(FLAGS)
    print(metric_result)
    print((metric_result[0] + metric_result[1] + metric_result[2]) / 3)