import os
import sieve
import numpy as np

from PIL import Image

@sieve.Model(
    name="image_style_transfer",
    python_version="3.8",
    gpu=True,
    python_packages=[
        "numpy==1.22.0",
        "Pillow==9.0.1",
        "protobuf==3.15.0",
        "six==1.12.0",
        "tensorboardX==1.8",
        "torch==1.8.1",
        "torchvision==0.9.1",
        "tqdm==4.35.0",
        "opencv-python==4.4.0.46",
        "imageio==2.9.0"
    ],
    run_commands=[
        "mkdir -p /root/.cache/style-transfer/AdaIN",
        "git clone https://github.com/naoto0804/pytorch-AdaIN /root/.cache/style-transfer/AdaIN",
        "mkdir -p /root/.cache/models/style-transfer/AdaIN/models",
        "wget -c 'https://drive.google.com/u/0/uc?id=1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr&export=download' -P /root/.cache/style-transfer/AdaIN/models", 
        "wget -c 'https://drive.google.com/u/0/uc?id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU&export=download' -P /root/.cache/style-transfer/AdaIN/models",
    ],
    iterator_input=True
)
class ImageStyleTransfer:
    def __setup__():
        import subprocess
        import cv2
        import uuid
        import tempfile
    def __predict__(content_img : sieve.Image, style_img : sieve.Image, alpha : str) -> sieve.Image:
        content_img, style_img, alpha = list(content_img)[0].array, list(style_img)[0].array, float(list(alpha)[0][0])

        temp_content = tempfile.NamedTemporaryFile(delete=True)
        temp_style = tempfile.NamedTemporaryFile(delete=True)

        cv2.imwrite(temp_content.name, content_img)
        cv2.imwrite(temp_style.name, style_img)

        test_path = "/root/.cache/style-transfer/AdaIN/pytorch-AdaIN/test.py"
        output_path = f"{uuid.uuid4()}.jpeg"

        subprocess.run(["python", test_path, "--content", temp_content.name, "--style", temp_style.name, "--alpha", alpha, "--output", output_path])
        return sieve.Image(path = output_path)

@sieve.workflow(name="image-neural-style-transfer", python_version="3.8")
def image_style_transfer_workflow(content_img : sieve.Image, style_img : sieve.Image, alpha : str) -> sieve.Image:
    output = ImageStyleTransfer()(content_img, style_img, alpha)
    return output