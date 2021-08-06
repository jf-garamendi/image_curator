from typing import Optional, Callable, Any
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

class CustomImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, transform, target_transform, loader, IMG_EXTENSIONS if is_valid_file is None else None)

    def find_classes(self, directory):
        return (
            ['rejected', 'approved'],
            {'rejected': 0, 'approved': 1}
        ),