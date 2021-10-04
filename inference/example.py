import os
import requests
from PIL import Image
from threading import Thread

from urllib.parse import urljoin
from multiprocessing import Queue
from model_inference import ModelInference, AbstractModelRequestManager

import uvicorn
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request

class APIRest_ModelRequestManager(AbstractModelRequestManager):
    def __init__(self, model_class, agent_config_path, device, half, batch_size, clients_folder, clients_url):
        super().__init__(model_class, agent_config_path, device, half, batch_size)
        self.queue = Queue()

        self.clients_folder = clients_folder
        os.makedirs(self.clients_folder, exist_ok=True)
        self.clients_url = clients_url

        # APIRest Definition
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.post("/put")
        async def put(
            client_id: Optional[str] = Form(None, description="Inference client id."),
            images_url: Optional[List[str]] = Form(None, description="Inference images using an URL."),
            
        ):
            self.queue.put((client_id, images_url))

    def load_client_callback(self, client_id):
        client_path = os.path.join(self.clients_folder, client_id + ".tar")
        if os.path.exists(client_path):
            return client_path, 0.5

        else:
            client_url = urljoin(self.clients_url, client_id)
            with open(client_path, 'wb') as f:
                f.write(requests.get(client_url, stream=True))
            return client_path, 0.5

    def load_image_callback(self, client_id, image):
        return Image.open(requests.get(image, stream=True).raw)

    def save_results_callback(self, client_id, dict_of_results):
        print(dict(dict_of_results))

    def get_request(self):
        return self.queue.get()

    def __run_api(self):
        uvicorn.run(self.app, host='0.0.0.0', port=8000)

    def run(self):
        t = Thread(target=self.__run_api)
        t.start()

        super().run()

if __name__ == '__main__':
    APIRest_ModelRequestManager(
        ModelInference,
        agent_config_path="/media/totolia/datos_3/photoslurp/image_curator2/configs/n0029_ag003_resnet152.json",
        device='cpu',
        half=False,
        batch_size=2,
        clients_folder="./clients",
        clients_url="test"
    ).run()