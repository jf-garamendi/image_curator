
from typing import Optional, Callable, Any

from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torchvision.models.resnet import resnet101

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
        