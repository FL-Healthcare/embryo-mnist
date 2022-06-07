import numpy as np
import PIL   
from PIL import Image, ImageEnhance, ImageOps
import random


class DataAugmentor():
    def __init__(self):
        self.augment_lvl = "Normal"
        
        
    def set_augment_lvl(self, lvl):
        self.augment_lvl = lvl
        
    
    def augment_data(self, images_dataframe):
        images = images_dataframe['images']
        available_focals = images.shape[0]      # 3
        available_frames = images_dataframe['times']      # Either (-1,0,1)
        
        focal, frame = self.get_random_frame_and_focal(available_focals, available_frames)
        raw_image = self.get_raw_image(frame, focal, images)
        
        #return either method["rotate", "brightness", "horizontal-flip", "clean"]
        
        if self.augment_lvl == "Normal":
            augment_method = self.select_random_augment_method()
            augmented_image = augment_method(raw_image)
            return augmented_image
        elif self.augment_lvl == "Aggresive":
            augment_method = self.select_random_augment_method()
            augmented_image = augment_method(raw_image)
            augmented_image_addition = augment_method(augmented_image) 
            return augmented_image_addition
         
        
    def get_random_frame_index(self, frames):
        if len(frames) == 3:
            random_frame_index = random.randrange(0, 3, 1)
            return random_frame_index
        elif len(frames) == 2:
            random_frame_index = random.randrange(0, 2, 1)
            return random_frame_index
        else:
            return 0

       
    def get_random_frame_and_focal(self, available_focals, available_frames):
        random_focal_index = random.randint(0, available_focals-1)
        random_frame_index = self.get_random_frame_index(available_frames)
        return random_focal_index, random_frame_index

    
    def select_random_augment_method(self):
        available_methods = ["rotate", "brightness", "horizontal-flip", "clean", "invert"]
        method_index = random.randint(0, len(available_methods)-1)
        
        selected_method = available_methods[method_index]
        
        if selected_method == "rotate":
            return self.rotate
        elif selected_method == "brightness":
            return self.brightness
        elif selected_method == "invert":
            return self.invert
        elif "horizontal-flip":
            return self.horizontal_flip
        else:
            return self.clean 
        
        
    def get_raw_image(self, frame_index, focal_index, images):    
        raw_image = images[frame_index,:,:,focal_index] 
        return raw_image
    
    
    def rotate(self, raw_image):
        img = Image.fromarray(raw_image)
        degree = random.randint(-10,10)
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
    
    
    def invert(self, raw_image):
        img = Image.fromarray(raw_image)
        inverted_image = PIL.ImageOps.invert(img)
        return np.asarray(inverted_image)
        
    
    def clean(self, raw_image):
        return raw_image
    
    
    def horizontal_flip(self, raw_image):
        img = Image.fromarray(raw_image)
        img_output = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return  np.asarray(img_output)    