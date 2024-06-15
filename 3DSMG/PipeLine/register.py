import sys
sys.path.append(".")
from torch import tensor, save
from Algorithms.Register import register
from StylizedModel.IO.Load import load as model_load
from StylizedModel.IO.Save import save as model_save
from tqdm import tqdm
from copy import deepcopy
from utils import walk_file
from StylizedModel.HeadMesh import HeadMesh
import os

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_path = 'data/models/template/template.fbx'

src_mesh = model_load(src_path ,device)

trg_file_list = walk_file('data/models/normalized/')
trg_file_list.sort()
trg_file_list = trg_file_list[39:]

loop_trg = tqdm(trg_file_list)
loop_iter = tqdm(range(4000))
for trg_name, trg_file in loop_trg:
    try:
        trg_mesh = model_load(trg_file, device)
        spatial_trans = register(src_mesh, trg_mesh, None, loop_iter)
        save(spatial_trans, 'data/spatialTrans/' + trg_name + '.pt')
        
        (filepath, tempfilename) = os.path.split(trg_file)
        model_save(trg_mesh, 'data/models/aligned/' + tempfilename)
        _src_mesh = model_load(src_path, device)
        
        with torch.no_grad():
            _src_mesh.transform_with_model(spatial_trans)
        model_save(_src_mesh, 'data/models/registered/' + trg_name + '.fbx')
    except Exception as e:
        print(repr(e))