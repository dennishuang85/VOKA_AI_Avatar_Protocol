import os
from torch import Tensor, cat, ones

def walk_file(path, legal_type = ['.obj', '.fbx']):
    objs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_type = os.path.splitext(file)
            if file_type in legal_type:
                objs.append([file, os.path.join(root, file)])
    return objs


def homo(x: Tensor) -> Tensor:
    """
    Convert a coordinates to homogeneous coordinates [N, 3] -> [N, 4, 1]

    Args:
        x (Tensor): [N, 3]

    Returns:
        Tensor: [N, 4, 1]
    """
    return cat((x, ones(x.shape[:-1] + tuple([1])).to(x.device)), -1).unsqueeze(-1)
