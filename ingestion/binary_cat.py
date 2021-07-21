from torch.utils.data import Dataset

class Binary_Class(Dataset):
    def __init__(self, root_dir, **kwargs):
    
        self.root_dir = root_dir
