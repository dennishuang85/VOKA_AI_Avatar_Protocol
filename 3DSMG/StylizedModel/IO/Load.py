from fbx import FbxImporter
from StylizedModel.HeadMesh import HeadMesh
from StylizedModel.ModelUtils import get_components, get_holes, find_eye_rims, find_mouth
import logging, torch, os, json
logger = logging.getLogger()
from StylizedModel.IO.FBXHandler import FBXHandler
from torch import tensor, zeros, ones, cat
    
def load_fbx(file_path: str, device: torch.device):
    
    obj = HeadMesh(device = device)
    
    handler = FBXHandler(file_path)
    
    mtrl_map, textures = handler.get_materials()
        
    verts, faces, uvs, mtrl_map = handler.get_mesh(mtrl_map)
    shapes = handler.get_shapes()
    
    obj.verts = tensor(verts).to(device)
    obj.faces = tensor(faces, dtype = torch.long).to(device)
    obj.uvs = tensor(uvs).to(device)
    obj.shapes = tensor(shapes).to(device)
    obj.handler = handler
    obj.mtrl_map = mtrl_map
    obj.textures = textures

    return obj

def load_obj(file_path: str, device: torch.device):
    from pytorch3d.io import load_obj
    verts, faces, aux = load_obj(file_path, device = device)
    obj = HeadMesh(verts, faces.verts_idx, device = device)
    return obj

def load_info(obj, file_path):
    
    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)

    info_file_path = filepath + '/' + filename + '.json'
    if os.path.exists(info_file_path):
        with open(info_file_path, 'r') as file:
            obj.info = json.load(file)

def pre_process(obj: HeadMesh, file_path: str):
    
    components = get_components(obj)
    
    logger.debug("Num of components is %d" % len(components))
    
    if obj.info is None or "components_name" not in obj.info:
        obj.components = {}
        if obj.info == None:
            obj.info = {}
        obj.info["components_name"] = {}
        for idx in components:
            obj.info["components_name"][str(idx)] = "Face"
            if "Face" not in obj.components:
                obj.components["Face"] = components[idx]
            else:
                obj.components["Face"] += components[idx]
    else:
        obj.components = {}
        for cp_idx in sorted(obj.info["components_name"]):
            if obj.info["components_name"][cp_idx] not in obj.components:
                obj.components[obj.info["components_name"][cp_idx]] = components.pop(int(cp_idx))
            else:
                obj.components[obj.info["components_name"][cp_idx]] += components.pop(int(cp_idx))
    
    if "Face" not in obj.components:
        logger.warning("No component Face in current object %s " % str)
    else:
        obj.holes = get_holes(obj)
        if obj.info is not None and "hole_name" in obj.info:
            for cp_idx in obj.info["hole_name"]:
                if obj.info["hole_name"][cp_idx] not in obj.holes:
                    obj.holes[obj.info["hole_name"][cp_idx]] = obj.holes.pop(int(cp_idx))
                else:
                    obj.holes[obj.info["hole_name"][cp_idx]] += obj.holes.pop(int(cp_idx))

def load(file_path: str, device: torch.device) -> HeadMesh:
    """_summary_

    Args:
        file_name (_type_): _description_
        device (_type_): _description_
    """
    logger.info("Loading file: %s" % file_path)
    
    if file_path.endswith('.fbx'):
        obj = load_fbx(file_path, device)
    elif file_path.endswith('.obj'):
        obj = load_obj(file_path, device)
    else:
        logger.warning("Can't load such file type: %s" % file_path)
    
    load_info(obj, file_path)
    
    pre_process(obj, file_path)
    
    return obj
    
    