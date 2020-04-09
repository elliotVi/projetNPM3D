# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:35:35 2020

@author: antoi
"""
import numpy as np

class Partition:
    
    def __init__(self, image):
        # The boolean matrix that indicates which pixels to partition
        self.image = image
        
        # Gives a different class to each pixel by default
        pixels = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j]:
                    pixels[i,j] = i*image.shape[1] + j
        self.pixels = pixels
        
        # Creates a class for each pixel by default
        classes = {}
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j]:
                    classes[i*image.shape[1] + j] = [ (i,j) ]
        self.classes = classes
        
    def get_key(self, pixel):
        if self.image[pixel]:
            return self.pixels[pixel]
    
    def merge_classes(self, pixel1, pixel2):
        # Takes the coordinates of two pixels as tuples, and merges two classes by adding the elements of the smaller one to the other
        key1, key2 = self.get_key(pixel1), self.get_key(pixel2)
        if(key1!=key2):
            class1, class2 = key1, key2

            if len(self.classes[key1]) < len(self.classes[key2]):
                class1, class2 = key2, key1
            
            # Updating the lists in the dictionary
            self.classes[class1] += self.classes[class2]
            del self.classes[class2]
            
            # Updating the array
            self.pixels[self.pixels==class2] = class1
        
    def get_class(self, pixel):
        # Takes pixel coordinates as a tuple and returns the list of all pixels in the same class
        key = self.pixels[pixel]
        return self.classes[key]
    
    def get_all_classes(self):
        return list(self.classes.values())
    
    
if __name__ == '__main__':
    nrow, ncol = 10, 10
    image = np.zeros((nrow, ncol))
    partition = Partition(image)
    print("Pixels: ", partition.pixels)
    print("Classes: ", partition.classes)
    print(partition.get_all_classes())
    
    for i in range(nrow):
        for j in range(1, ncol):
            partition.merge_classes((i,0), (i,j))
    print("Pixels: ", partition.pixels)
    print("Classes: ", partition.classes)
    print(partition.get_all_classes())
    
    zones = list(partition.get_all_classes())
    argmax = 0
    for i in range(0, len(zones)):
        if len(zones[i]) > len(zones[argmax]):
            argmax = i
    print(zones[argmax])