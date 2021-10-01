import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import glob
import cv2

from dataset import Dataset, create_datasets, CuratorPairedDataset
from loss import compute_center_loss, get_center_delta
from models import Resnet101CuratorModel
from device import device
from trainer import Trainer
from utils import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training


def main(args):
    if args.evaluate:
        evaluate(args)
    elif args.verify_model:
        verify(args)
    else:
        train(args)


def get_dataset_dir(args):
    return args.dataset_dir


def get_log_dir(args):
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    return log_dir


def get_model_class(args):
    if args.arch == 'resnet101':
        model_class = Resnet101CuratorModel

    return model_class



def train(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    training_set, validation_set, num_classes = create_datasets(dataset_dir)

    training_dataset = Dataset(
            training_set, transform_for_training(model_class.IMAGE_SHAPE))
    validation_dataset = Dataset(
        validation_set, transform_for_infer(model_class.IMAGE_SHAPE))

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False
    )

    model = model_class(num_classes).to(device)

    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if
                          param.requires_grad and 'bn' in name]

    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=args.lr, momentum=0.9)

    trainer = Trainer(
        optimizer,
        model,
        training_dataloader,
        validation_dataloader,
        max_epoch=args.epochs,
        resume=args.resume,
        log_dir=log_dir
    )
    trainer.train()


def evaluate(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    dataset = CuratorModelPairedDataset(
        dataset_dir, None, transform_for_infer(model_class.IMAGE_SHAPE))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    model = model_class(False).to(device)

    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    embedings_a = torch.zeros(len(dataset), model.FEATURE_DIM)
    embedings_b = torch.zeros(len(dataset), model.FEATURE_DIM)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(tqdm(dataloader)):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a = model(images_a)
        _, batched_embedings_b = model(images_b)

        start = args.batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = batched_embedings_a.data
        embedings_b[start:end, :] = batched_embedings_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 4, 0.1)
    distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)

    tpr, fpr, accuracy, best_thresholds = compute_roc(
        distances,
        matches,
        thresholds
    )

    roc_file = args.roc if args.roc else os.path.join(log_dir, 'roc.png')
    generate_roc_curve(fpr, tpr, roc_file)
    print('Model accuracy is {}'.format(accuracy))
    print('ROC curve generated at {}'.format(roc_file))


def verify(args):
    dataset_dir = get_dataset_dir(args)
    log_dir = get_log_dir(args)
    model_class = get_model_class(args)

    model = model_class(False).to(device)
    checkpoint = torch.load(args.verify_model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    image_a, image_b = args.verify_images.split(',')

    if os.path.isdir(image_a):
        aux = glob.glob(os.path.join(image_a, "*.jpg"))
        image_a = aux[np.random.randint(0, len(aux))]
    
    if os.path.isdir(image_b):
        aux = glob.glob(os.path.join(image_b, "*.jpg"))
        image_b = aux[np.random.randint(0, len(aux))]

    image_a_c = image_loader(image_a)
    image_b_c = image_loader(image_b)

    image_a = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_a_c)
    image_b = transform_for_infer(
        model_class.IMAGE_SHAPE)(image_b_c)
    images = torch.stack([image_a, image_b]).to(device)

    _, (embedings_a, embedings_b) = model(images)

    distance = torch.sum(torch.pow(embedings_a - embedings_b, 2)).item()
    
    image_a_c = cv2.resize(image_a_c, (model_class.IMAGE_SHAPE[1], model_class.IMAGE_SHAPE[0]))
    image_b_c = cv2.resize(image_b_c, (model_class.IMAGE_SHAPE[1], model_class.IMAGE_SHAPE[0]))
    im_c = np.concatenate((image_a_c, image_b_c), axis=1)

    print("distance: {}".format(distance))

    cv2.imshow("image", im_c)
    cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='center loss example')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--log_dir', type=str,
                        help='log directory')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--arch', type=str, default='resnet101',
                        help='network arch to use, support resnet101 and '
                             'resnet101 (default: resnet101)')
    parser.add_argument('--resume', type=str,
                        help='model path to the resume training',
                        default=False)
    parser.add_argument('--dataset_dir', type=str,
                        help='directory with lfw dataset'
                             ' (default: $HOME/datasets/lfw)')
    parser.add_argument('--weights', type=str,
                        help='pretrained weights to load '
                             'default: ($LOG_DIR/resnet18.pth)')
    parser.add_argument('--evaluate', type=str,
                        help='evaluate specified model on lfw dataset')
    parser.add_argument('--pairs', type=str,
                        help='path of pairs.txt '
                             '(default: $DATASET_DIR/pairs.txt)')
    parser.add_argument('--roc', type=str,
                        help='path of roc.png to generated '
                             '(default: $DATASET_DIR/roc.png)')
    parser.add_argument('--verify-model', type=str,
                        help='verify 2 images of Curator belong to one person,'
                             'the param is the model to use')
    parser.add_argument('--verify-images', type=str,
                        help='verify 2 images of Curator belong to one person,'
                             'split image pathes by comma')

    args = parser.parse_args()
    main(args)