import torch
from torchvision import transforms
import os

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from utils.config import *
#from agents import *
import graphs.models as model_modules
MODELS = vars(model_modules)

class ModelInference:
    def __init__(
        self,
        agent_config_path,

        load_client_callback,
        load_image_callback,
        save_results_callback,

        device='cpu',
        half=False
    ):

        # parse the config json file
        config = process_config(agent_config_path)

        # Device & Half
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half if self.device != torch.device('cpu') else False

        # Create the Agent and pass all the configuration to it then run it..
        # agent_class = globals()[config.general.agent]
        # self.agent = agent_class(config)
        model_class = MODELS[config.model.model] #globals()[config.model.model]
        self.model = model_class(config.model.model_name, config.model.type).to(self.device)
        self.dataset_config = config.dataset

        # Move model
        self.model = self.model.to(self.device)
        if self.half:
            self.model = self.model.half()
        self.model.eval()

        # Define Transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.dataset_config.W, self.dataset_config.H)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_client_callback = load_client_callback
        self.load_image_callback = load_image_callback
        self.save_results_callback = save_results_callback

    def __call__(
        self, 
        client_id, 
        images, 
        batch_size=1
    ):
        (
            client_weights_path, 
            client_threshold
        ) = self.load_client_callback(client_id)
        self.model.load_state_dict(torch.load(client_weights_path)['model_state_dict'], strict=True)
        self.model.eval()
        
        results = []
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))

            # Prepare batch
            batch_images = torch.stack([
                self.image_transforms(self.load_image_callback(client_id, im_info)) \
                for im_info in images[batch_start:batch_end]
            ], dim=0).to(device=self.device)

            if self.half:
                batch_images = batch_images.half()

            # Inference
            with torch.no_grad():
                pred_bool = self.model(batch_images) >= client_threshold
            results += pred_bool.cpu().tolist()

        # Save results
        self.save_results_callback(client_id, zip(images, results))
            
class AbstractModelRequestManager:
    def load_client_callback(self, client_id):
        raise NotImplementedError

    def load_image_callback(self, client_id, images):
        raise NotImplementedError

    def save_results_callback(self, client_id, dict_of_results):
        raise NotImplementedError

    def get_request(self):
        raise NotImplementedError

    def __init__(
        self,
        model_class,
        
        agent_config_path,

        device,
        half,
        batch_size
    ):
        self.model = model_class(
            agent_config_path=agent_config_path,

            load_client_callback=self.load_client_callback,
            load_image_callback=self.load_image_callback,
            save_results_callback=self.save_results_callback,

            device=device,
            half=half
        )
        self.is_finish = True
        self.batch_size = batch_size

    def run(self):
        self.is_finish = False
        while not self.is_finish:
            aux = self.get_request()
            if aux is None:
                continue

            client_id, images = aux
            self.model(
                client_id=client_id,
                images=images,
                batch_size=self.batch_size
            )

    def stop(self):
        self.is_finish = True