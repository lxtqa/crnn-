from demo import *
import os
import sys
from tqdm import tqdm
def recognition(path):
    transformer, model, converter = load_model()

    max_confi = - sys.maxsize - 1
    img_name = ""
    imglist = os.listdir(path)
    for img in tqdm(imglist):
        confi, bconfi, raw_pred, sim_pred = demo(path+"/"+img, transformer, model, converter)
        if sim_pred == "jfe":
        # if img == "113_5939.jpg":
        #     print("")
        #     print(confi)
        #     print(bconfi)
        #     print(sim_pred)
        #     print("")

        # if img == "113_4102.jpg":
        #     print("")
        #     print(confi)
        #     print(bconfi)
        #     print(sim_pred)
        #     print("")
            if confi > max_confi:
                img_name = img
                max_confi = confi
    return img_name, max_confi
if __name__ == "__main__":
    path = "../jfe"
    img,confi = recognition(path)
    print(img)
    print(confi)
    
