from logging import Logger
from StylizedModel.BaseMesh import BaseMesh
from torch import Tensor, cat
from typing import Optional

class CompMesh(BaseMesh):
    def __init__(self, verts: Tensor, faces: Tensor, father, cfaces: Tensor = None, device = None, **kwargs):
        BaseMesh.__init__(self, verts, faces, device)
        self.father = father
        self.cfaces = cfaces # with shape [M, 1] type int
    
    def vert_cords(self):
        return self.father.vert_cords()[self.verts]
    
    def __add__(self, other):
        """ Merge two mesh in simple way

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self is None:
            return other
        if other is None:
            return self
        assert self.father == other.father
        new_faces = other.faces + self.verts.shape[0]
        self.cfaces = cat((self.cfaces, other.cfaces))
        self.faces = cat((self.faces, new_faces))
        
        self.verts = cat((self.verts, other.verts))
        return self
    
    def face_idx(self):
        return self.cfaces
