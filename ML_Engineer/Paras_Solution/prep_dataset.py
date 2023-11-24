import os
import cv2
import glob
import tqdm
import argparse
import config
import numpy as np


def prepare_dataset(path: str) -> None:
    """Cocominitrain_25k download link: https://ln5.sync.com/dl/0324da1d0/rmi7abjx-2dj4ktii-d9jcwgc5-s7fwwrb7
    Repository: https://github.com/giddyyupp/coco-minitrain
    """
    # Make train/test folders for Sobel dataset
    cwd = os.getcwd()
    custom_data_path = os.path.join(cwd, "custom_data")
    
    # Make custom data path
    if not os.path.exists(custom_data_path): 
        os.mkdir(custom_data_path) 
    
    # Make train/test paths
    train_path = os.path.join(custom_data_path, "train")
    if not os.path.exists(train_path): 
        os.mkdir(train_path)
    test_path = os.path.join(custom_data_path, "test") 
    if not os.path.exists(test_path): 
        os.mkdir(test_path)

    # Make input/output folders
    train_input_path = os.path.join(train_path, "input") 
    if not os.path.exists(train_input_path): 
        os.mkdir(train_input_path)
    
    train_output_path = os.path.join(train_path, "output")
    if not os.path.exists(train_output_path): 
        os.mkdir(train_output_path)
    
    test_input_path = os.path.join(test_path, "input")
    if not os.path.exists(test_input_path): 
        os.mkdir(test_input_path)
    
    test_output_path = os.path.join(test_path, "output")
    if not os.path.exists(test_output_path): 
        os.mkdir(test_output_path)

    input, output = [], []
    data = glob.glob(os.path.join(path, "images", "train2017/*"))
    for file in data[:config.DATA_SIZE]:
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (250,250))
        input.append(img)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_gt = np.sqrt(np.square(gx), np.square(gy))
        sobel_gt /= np.max(sobel_gt) # normalize
        output.append(sobel_gt)


    # Save train and test images
    # Train test split 90:10
    train_input, train_output = input[:int(config.DATA_SIZE*0.9)], output[:int(config.DATA_SIZE*0.9)] 
    test_input, test_output = input[int(config.DATA_SIZE*0.9):], output[int(config.DATA_SIZE*0.9):] 
    for i in range(len(train_input)):
        cv2.imwrite(os.path.join(train_input_path, f"img_{i}.png"), train_input[i])
        cv2.imwrite(os.path.join(train_output_path, f"sobel_{i}.png"), train_output[i]*255)  
    
    for i in range(len(test_input)):
        cv2.imwrite(os.path.join(test_input_path, f"img_{i}.png"), test_input[i])
        cv2.imwrite(os.path.join(test_output_path, f"sobel_{i}.png"), test_output[i]*255)  


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Dataset Path')
    parser.add_argument("--path",
                        type=str,
                        default=os.path.join(os.getcwd(), "coco_minitrain_25k"), 
                        help="Original dataset path")
    
    args = parser.parse_args()
    prepare_dataset(args.path)