from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import os
import glob
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
from collections import defaultdict
import timm

from typing import Dict, Optional, Callable, Any, Tuple, Dict
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torchvision.models.resnet import resnet101, resnet50
from functools import reduce


class CustomImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(CustomImageFolder, self).__init__(
            root=root, 
            transform=transform, 
            target_transform=target_transform, 
            loader=loader, 
            is_valid_file=None
        )


    def _find_classes(self, directory):
        return (
            ['rejected', 'approved'],
            {'rejected': 0, 'approved': 1}
        )
    
    def make_dataset(self, 
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        print(directory)
        instances_tmp = defaultdict(list)
        instances_counter = defaultdict(lambda: 0)

        for class_name, class_label in class_to_idx.items():
            folder_images_path = os.path.join(directory, class_name)
            for im_path in glob.glob(os.path.join(folder_images_path, "*")):
                instances_tmp[class_name].append((im_path, class_label))
            instances_counter[class_name] = len(instances_tmp[class_name])

        print(instances_counter)
        max_class = max(instances_counter.values())

        for class_name in class_to_idx.keys():
            current_class_counter = instances_counter[class_name]
            num_samples = max_class - current_class_counter
            instances_tmp[class_name] += [instances_tmp[class_name][idx] for idx in np.random.choice(len(instances_tmp[class_name]), num_samples)]
            instances_counter[class_name] = len(instances_tmp[class_name])

        print(instances_counter)
        print()
        print()
        return functools.reduce(lambda a, b: a + b, instances_tmp.values())



class Resnet101CuratorModel(nn.Module):
    IMAGE_SHAPE = (512, 512)
    FEATURE_DIM = 2048

    def __init__(self):
        super().__init__()
        self.base = resnet101(pretrained=True)
        self.proj = nn.Linear(self.FEATURE_DIM, 512, bias=True)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feature = x.view(x.size(0), -1)
        feature = self.proj(feature)

        # feature_normed = feature.div(
        #     torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return feature



class Resnext50dCuratorModel(nn.Module):
    IMAGE_SHAPE = (384, 384)
    FEATURE_DIM = 2048

    def __init__(self):
        super().__init__()
        self.base = timm.create_model("resnext50d_32x4d", pretrained=True)
        self.proj = nn.Linear(self.FEATURE_DIM, 128, bias=True)

    def forward(self, x):
        # x = self.base.conv1(x)
        # x = self.base.bn1(x)
        # x = self.base.relu(x)
        # x = self.base.maxpool(x)
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        # x = self.base.layer4(x)
        x = self.base.forward_features(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feature = x.view(x.size(0), -1)
        feature = self.proj(feature)

        # feature_normed = feature.div(
        #     torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return feature

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    num_iterations = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}/{}: Loss = {}, Number of mined triplets = {}".format(epoch, batch_idx, num_iterations, loss, mining_func.num_triplets))

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

metric_value = -1
def test(train_set, test_set, model, accuracy_calculator):
    global metric_value
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                test_labels,
                                                train_labels,
                                                False)
    if accuracies["precision_at_1"] > metric_value:
        metric_value = accuracies["precision_at_1"]
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(model.state_dict(), './checkpoints/model_best.pt')
        with open('./checkpoints/best.txt', 'w') as f:
            f.write(f'Precision@1: {metric_value}')

    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

device = torch.device("cuda")
model = Resnext50dCuratorModel().to(device)
batch_size = 16

transform_test = transforms.Compose([
    # transforms.GaussianBlur((15, 15), sigma=(0.01, 4.0)),
    # NewPad(),
    transforms.Resize(model.IMAGE_SHAPE),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
])

transform_train = transforms.Compose([
        transforms.Resize(model.IMAGE_SHAPE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

dataset1 = CustomImageFolder("/media/totolia/datos_3/photoslurp/dataset/images_oriented/bonprix/train", transform=transform_train)
dataset2 = CustomImageFolder("/media/totolia/datos_3/photoslurp/dataset/images_oriented/bonprix/test", transform=transform_test)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 150

distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low = 0)
loss_func = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer)
mining_func = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")
accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

for epoch in range(1, num_epochs+1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    test(dataset1, dataset2, model, accuracy_calculator)