import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import apply_affine_transform
import random, os

class Augmentation():
    def __init__(self, img_path) -> None:
        self.img_path = img_path
        self.check_path()
        pass
    def noisy(self, image, mean = 0.03, capacity = random.randint(1000,500000)):
        row,col,ch= image.shape
        sigma = capacity**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch)).reshape(row,col,ch)
        return gauss + image

    def move(self, img, x, y):
        move_matrix = np.float32([[1, 0, x], [0, 1, y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv.warpAffine(img, move_matrix, dimensions)

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        return result
    
    def load_image(self):
        return cv.cvtColor(cv.imread(self.img_path), cv.COLOR_BGR2RGB)
    
    def check_path(self):
        if not os.path.exists('./new_images/'): os.mkdir('./new_images/')  
    
    def augmentation(self):
            img_rgb = self.load_image()
            af_img = apply_affine_transform(img_rgb, theta=random.randint(-20,20), row_axis=0, col_axis=1, channel_axis=2)
            gn_img = self.noisy(af_img)
            blurImg = cv.blur(gn_img,(random.randint(1,100),random.randint(1,100)))
            moved = self.move(blurImg, random.randint(1,100), random.randint(1,100))
            return self.rotate_image(moved, random.randint(1,80))

    def data_multiply(self, copy_number):
        for i in range (copy_number):
            filename = f'./new_images/{str(i)}.jpg'
            cv.imwrite(filename, self.augmentation())



if __name__ =='__main__':
    Augmentation(img_path='70.jpg').data_multiply(20)