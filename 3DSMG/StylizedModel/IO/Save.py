from numpy import str0
from StylizedModel.HeadMesh import HeadMesh
import logging, os, json
from torch import device
import pytorch3d

logger = logging.getLogger()

def save_fbx(obj: HeadMesh, file_path):
    obj.update_scene()
    assert obj.handler is not None
    from fbx import FbxExporter
    exporter = FbxExporter.Create(obj.handler.fbx_manager, "")
    if exporter.Initialize(file_path, -1, obj.handler.fbx_manager.GetIOSettings()) == 0:
        raise Exception("Could not save fbx")
    exporter.Export(obj.handler.scene)
    exporter.Destroy()

def save_obj(obj: HeadMesh, file_path: str):
    pytorch3d.io.save_obj(file_path, obj.vert_cords()[...,:3], obj.get_faces())

def save_info(obj: HeadMesh, file_path: str):
    
    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)

    info_file_path = filepath + '/' + filename + '.json'
    
    if obj.info is not None:
        with open(info_file_path, 'w+') as file:
            json.dump(obj.info, file)

def save(obj: HeadMesh, file_path: str) -> HeadMesh:
    
    logger.info("Saving file: %s" % file_path)
    
    if file_path.endswith('.fbx'):
        save_fbx(obj, file_path)
    elif file_path.endswith('.obj'):
        save_obj(obj, file_path)
    else:
        logger.warning("Can't save such file type: %s" % file_path)
    
    save_info(obj, file_path)
    