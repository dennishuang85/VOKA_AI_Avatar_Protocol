import torch, os, sys
sys.path.append(".")
import logging
logging.basicConfig(level=logging.WARNING)
from StylizedModel.IO.Load import load
from StylizedModel.IO.Save import save
from StylizedModel.ModelUtils import find_eye_rims, find_mouth
from utils import walk_file
import json
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    obj_list = walk_file("./data/models/normalized")
    obj_list.sort()
    for name, path in obj_list:
        
        (_filepath, _tempfilename) = os.path.split(path)
        (_filename, _extension) = os.path.splitext(_tempfilename)

        info = None
        info_file_path = _filepath + '/' + _filename + '.json'
        if os.path.exists(info_file_path):
            with open(info_file_path, 'r') as file:
                info = json.load(file)
        
        if info is not None and "hole_name" in info and 'LeftEyeRim' in info["hole_name"].values() and 'RightEyeRim' in info["hole_name"].values():
            continue
        
        print("Dealing with %s" % name)
        obj = load(path, device = device)
        
        if 'Face' in obj.components:
            if 'LeftEyeRim' not in obj.holes or 'RightEyeRim' not in obj.holes:
                find_eye_rims(obj)
            if 'Mouth' not in obj.holes:
                find_mouth(obj)
        
        save(obj, "./data/models/normalized/" + name)