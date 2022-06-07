import numpy as np
import PIL
from PIL import Image, ImageEnhance
import random


class DataAugmentorMNIST():
    def __init__(self):
        pass
    
    
    def augment_data(self, batch):
        augmented_images = np.empty((len(batch), 28,28))
        for i, image in enumerate(batch):
            #return either method["rotate", "brightness", "clean"]
            augment_method = self.select_random_augment_method()
            augmented_image = augment_method(image)
            augmented_images[i] = augmented_image
            
        return augmented_images
        
    
    def select_random_augment_method(self):
        available_methods = ["rotate", "brightness", "clean"]
        method_index = random.randint(0, len(available_methods)-1)
        
        selected_method = available_methods[method_index]
        
        if selected_method == "rotate":
            return self.rotate
        elif selected_method == "brightness":
            return self.brightness
        else:
            return self.clean 
        
        
    def get_raw_image(self, frame_index, focal_index, images):    
        raw_image = images[frame_index,:,:,focal_index] 
        return raw_image
    
    
    def rotate(self, raw_image):
        img = Image.fromarray(raw_image)
        degree = random.randint(5,5)
        img = img.rotate(degree)
        return np.asarray(img)
    
    
    def brightness(self, raw_image):
        img = Image.fromarray(raw_image)
        enhancer = ImageEnhance.Brightness(img)

        #factor = 1 #gives original image
        #factor = 1.1 #slighty brightens the image
        factor = random.uniform(1, 1.1)
        im_output = enhancer.enhance(factor)
        return np.asarray(im_output)
    
    
    def clean(self, raw_image):
        return raw_image
    
