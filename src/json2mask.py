import numpy as np
import cv2
import json
import os
import tqdm

from constant import JSON_DIR, MASK_DIR




json_list = os.listdir(JSON_DIR)

iterator_example=range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass

for json_name in tqdm.tqdm(json_list):
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)

    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")
    
    for obj in json_dict["objects"]:
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]),
                                color=100)
            
    cv2.imwrite(mask_path, mask.astype(np.uint8))