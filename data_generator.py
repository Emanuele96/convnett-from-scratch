import numpy as np
import random
import math        
from matplotlib import pyplot as plt

class Data_Generator():

    def __init__(self, n_size, categories, number_of_categories, pictures_per_category, train_val_test_percent, centered_percent, noise_factor, soft_start):
        random.seed(42)
        np.random.seed(42)
        self.n_size = n_size
        #numpy array of shape = (number_of_categories, number_of_categories, 1)  example [[1 0 0 0]
        #                                                                                 [0 1 0 0]
        #                                                                                 [0 0 1 0]
        #                                                                                 [0 0 0 1]]
        # can use identity matrix of size number_of_categories 
        self.categories = np.expand_dims(np.identity(categories, dtype=int),1)
        self.number_of_categories = number_of_categories
        self.pictures_per_category = pictures_per_category
        self.centered_percent = centered_percent
        self.total_images = self.pictures_per_category * self.number_of_categories
        self.train_size = math.ceil( train_val_test_percent[0] * self.total_images)
        self.valid_size = math.ceil(train_val_test_percent[1] * self.total_images)
        self.test_size = math.ceil(train_val_test_percent[2] * self.total_images)
        self.noise_factor = noise_factor/2
        self.soft_start = soft_start
        # Get an array of degrees for the circle calculation, size 3 times of picture size for scale(enough point to look nice, in relation to size)
        self.theta = np.linspace(0, 2 * np.pi, self.n_size**2)

    def generate_dataset(self):
        #shape of images and labels to match (tot, 1, n*n) and (tot, 1, cat), so that array of row vectors, to be fed in the network
        images = np.zeros(shape=(self.total_images, 1, self.n_size* self.n_size))
        labels = np.zeros(shape=(self.total_images, 1,  self.number_of_categories))
        counter = 0
        for category in self.categories:
            for picture in range(self.pictures_per_category):
                if (category == self.categories[0]).all():
                    images[counter] += self.apply_noise(self.generate_random_horizontal_bar_image(True), self.generate_noise(True))
                    labels[counter] = self.assign_label(category)
                    counter += 1
                elif (category == self.categories[1]).all():
                    images[counter] +=self.apply_noise(self.generate_random_vertical_bar_image(True), self.generate_noise(True))
                    labels[counter] = self.assign_label(category)
                    counter += 1
                elif (category == self.categories[2]).all():
                    images[counter] += self.apply_noise(self.generate_random_circle_image(True), self.generate_noise(True))
                    labels[counter] = self.assign_label(category)
                    counter += 1
                elif (category == self.categories[3]).all:
                    images[counter] += self.apply_noise(self.generate_random_rectangle_image(True), self.generate_noise(True))
                    labels[counter] = self.assign_label(category)
                    counter += 1
        #Generate a numpy array of evenly spaced indexes, then shuffle
        indexes = np.arange(self.total_images)
        np.random.shuffle(indexes)
        stop_index_validate = self.train_size + self.valid_size
        #start_index_test = stop_index_validate + self.valid_size
        x_train, y_train = images[indexes[:self.train_size]] , labels[indexes[:self.train_size]]
        x_validate , y_validate = images[indexes[self.train_size: stop_index_validate]] , labels[indexes[self.train_size: stop_index_validate]]
        x_test , y_test = images[indexes[stop_index_validate:]] , labels[indexes[stop_index_validate:]]
        
        return x_train, y_train, x_validate, y_validate, x_test, y_test

    def assign_label(self, category):
        if self.soft_start:
            return abs(category - 0.1)
        else:
            return category

    def generate_random_vertical_bar_image(self, flat):
        a= np.zeros((self.n_size , self.n_size))
        step = random.randint(2, self.n_size/2)
        for i in range(1, self.n_size):
            if i % step == 0:
                a[:, i] = 1
        if flat:
            return a.reshape(-1)
        return a

    def generate_random_horizontal_bar_image(self, flat):
        a= np.zeros((self.n_size , self.n_size))
        step = random.randint(2, self.n_size/2)
        for i in range(1, self.n_size):
            if i % step == 0:
                a[i, :] = 1
        if flat:
            return a.reshape(-1)
        return a

    def generate_random_circle_image(self, flat):
        # Begin with a blank canvas
        a= np.zeros((self.n_size , self.n_size))  
        # find the center of the image
        center = math.floor(self.n_size/2)    
        # decide for a random radius. Max size is half ot the image width/height
        r = random.randint(int(math.sqrt(self.n_size)), center)
        # Set the origo to 0,0 if with probability centered_percent, else find a random origo in relation to radius and picture size
        if random.random() <= self.centered_percent:
            o = (center, center)
        else:
            o = (random.randint(r, self.n_size - r), random.randint(r, self.n_size - r))
        # Calculate the x and y coordinate, stack it to get a 2 dimentional array of coordinates [[x1,y1], [x2,y2], ... , [xn*3, yn*3]]
        x = r * np.cos(self.theta) + o[0]
        y = r * np.sin(self.theta) + o[1]
        coordinate = np.column_stack((x, y))
        # For each point of the circle, toggle the pixel on the canvas to "draw" the circle
        for point in coordinate:
            x_point = min(math.floor(point[0]), self.n_size - 1)
            y_point = min(math.floor(point[1]), self.n_size - 1)
            a[x_point, y_point] = 1
        # Flatten the 2D array to 1D if required
        if flat:
            return a.reshape(-1)
        return a

    def generate_random_rectangle_image(self, flat):
        a =  np.zeros((self.n_size , self.n_size))
        x = random.randint(0, self.n_size - 1)
        y = random.randint(0, self.n_size - 1)
        if x < self.n_size/2:
            width = random.randint(x + 1, self.n_size - x - 1)
        else:
            width = - random.randint(2, x - 1)
        if y < self.n_size/2:
            height = random.randint( y + 1, self.n_size - y - 1)
        else:
            height = - random.randint(2, y - 1)
        
        x_step = int(width / abs(width))
        y_step =  int(height/abs(height))
        for i in range(0, width + x_step, x_step):
            a[y, x + i] = 1
            a[y + height, x + i] = 1
        for i in range(0, height + y_step, y_step):
            a[y + i, x] = 1
            a[y + i,x + width] = 1

        if flat:
            return a.reshape(-1)
        return a

    def generate_noise(self, flat):
        a = np.random.rand(self.n_size, self.n_size)
        noise = np.where(a <self.noise_factor, 1, 0)
        if flat:
            return noise.reshape(-1)
        return noise

    def apply_noise(self, data, noise):
        return np.subtract(data, noise)**2

    def show_picture(self, data):
        plt.imshow(data, interpolation="nearest")
        plt.gca().invert_yaxis()
        plt.show()
    
