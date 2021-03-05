import model
import data_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from progress.bar import IncrementalBar
import json
import argparse

# Tools 
def read_config_from_json(filename):
    with open('configs/' + filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open('configs/' + filename, 'w') as fp:
        json.dump(cfg, fp)

def animate(i):
    plt.clf()
    if i == len(x_train):
        print( '.', end ='' ) 
        bar.finish()
    im = plt.imshow(np.reshape(x_train[i], (cfg["n_size"],cfg["n_size"])),interpolation="nearest")
    bar.next()
    return im



# Parse config file of choice
parser = argparse.ArgumentParser("Deep Learning Project 1")
parser.add_argument('--config', default="conv2d.json", type=str, help="Select configuration file to load")
args = parser.parse_args()
cfg = read_config_from_json(args.config)
# Generate dataset
data_generator = data_generator.Data_Generator(cfg["n_size"], cfg["categories"], cfg["categories"], cfg["pic_per_categories"], cfg["train_val_test_percent"], cfg["center_image_prob"], cfg["noise_percent"], cfg["soft_start"], cfg["flatten_dataset"])
x_train, y_train, x_validate, y_validate, x_test, y_test = data_generator.generate_dataset()
# Run Program
if __name__ == "__main__":

    if cfg["show_pictures_on_start"]:
        bar = IncrementalBar('X_train input', max=len(x_train))
        fig = plt.figure( )
        anim = animation.FuncAnimation(fig, animate, interval  = cfg["animation_speed"])
        plt.show()
    # Construct the model
    m1 = model.Model(cfg, input_shape = x_train[0].shape)
    input_shape = x_train[0].shape
    input_nodes = None
    # Add hidden layers
    for layer in cfg["layers"]:
        input_shape, input_nodes = m1.add_layer(layer, input_shape, input_nodes, cfg["debug"])
    # Add Softmax if required
    if cfg["use_softmax"]:
        m1.add_softmax()  
    print(m1)
    #Train the model
    if cfg["train_on_start"]:
        losses, validation_errors = m1.train(x_train, y_train, x_validate, y_validate, cfg["epochs"], cfg["batch_size"])
        time = np.linspace(0, len(losses), num=len(losses))
        time_validate = np.linspace(0, len(losses), num=cfg["epochs"])
        # test trained model on test data
        test_loss = m1.test(x_test, y_test)
        test_losses = np.linspace(test_loss, test_loss, num = 10)
        time_test = np.linspace(len(losses) + 1 , len(losses) + len(test_losses) + 1, num=len(test_losses))
        plt.plot(time, losses)
        plt.plot(time_validate, validation_errors)
        plt.plot(time_test, test_losses)
        plt.legend(["train", "validate", "test"], loc ="upper right") 
        plt.show()
    if cfg["visualize_kernels"]:
        m1.visualize_kernels()
        

    


