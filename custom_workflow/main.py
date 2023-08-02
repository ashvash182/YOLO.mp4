import os
import sieve
import cv2
import numpy as np

from PIL import Image

def get_mean_and_std(x: sieve.Image):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    return x_mean, x_std

@sieve.Model(name="color_transfer_model", python_version="3.8", iterator_input=True)
class ColorTransfer:
    def __setup__(self):
        return
    def __predict__(self, content_img : sieve.Image, style_img : sieve.Image) -> sieve.Image:
        content_img, style_img = list(content_img)[0], list(style_img)[0] 

        content_img, style_img = content_img.array, style_img.array
        
        s, t = cv2.cvtColor(content_img, cv2.COLOR_BGR2LAB), cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB)
        s_mean, s_std = get_mean_and_std(s)
        t_mean, t_std = get_mean_and_std(t)

        height, width, channel = s.shape
        for i in range(0,height):
            for j in range(0,width):
                for k in range(0,channel):
                    x = s[i,j,k]
                    x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                    # round or +0.5
                    x = round(x)
                    # boundary check
                    x = 0 if x<0 else x
                    x = 255 if x>255 else x
                    s[i,j,k] = x
            
        s = cv2.cvtColor(s,cv2.COLOR_LAB2RGB)
        return sieve.Image(array = np.array(s))

@sieve.workflow(name="image-color-transfer", python_version="3.8")
def color_transfer_workflow(content_img : sieve.Image, style_img : sieve.Image) -> sieve.Image:
    transfered = ColorTransfer()(content_img, style_img)
    return transfered
