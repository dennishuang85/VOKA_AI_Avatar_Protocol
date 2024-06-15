from torch import Tensor, tensor, cat, zeros, matmul
from typing import Dict, Optional
from pytorch3d.structures.meshes import Meshes
from utils import homo

class BaseMesh(object):
    def __init__(self, verts: Tensor, faces: Tensor, device = None, **kwargs):
        """Base class for our model. All coordinates are stored in homogeneous form
        Args:
            verts (Optional[tensor], optional): _description_. Defaults to None.
            faces (Optional[tensor], optional): _description_. Defaults to None.
        """
        
        self.verts = verts
        self.faces = faces
        self.device = device
        
    def get_verts(self) -> Tensor:
        """
        Looking for coordinates? Use vert_cords instead of this function.
        This function for get origin data.
        Returns:
            _type_: _description_
        """
        return self.verts
    
    def vert_cords(self) -> Tensor:
        """_summary_

        Returns:
            Tensor: with shape [N, 4]
        """
        return self.verts
    
    def get_faces(self) -> Tensor:
        """ 

        Returns:
            Tensor: with shape [M, 3]
        """
        return self.faces
    
    def get_edge_table(self) -> Dict:
        table = {}
        edges= self.get_edges().tolist()
        for u, v in edges:
            if u not in table:
                table[u] = []
            table[u].append(v)
        return table
    
    def get_edges(self) -> Tensor:
        """ 
        Get a [*, 2] tensor representing the edges of this mesh.

        Returns:
            Tensor: _description_
        """
        edges = set()
        for face in self.faces.tolist():
            edges.add((face[0], face[1]))
            edges.add((face[0], face[2]))
            edges.add((face[1], face[2]))
            edges.add((face[1], face[0]))
            edges.add((face[2], face[0]))
            edges.add((face[2], face[1]))
        return tensor(list(edges)).to(self.device)

    def get_adjacency_matrix(self) -> Tensor:
        """
        Get a [N, N] matrix representing if two vertices are connected
        
        Returns:
            Tensor: with dtype bool.
        """
        edges = self.get_edges()
        N = self.verts.shape[0]
        adjacency_matrix: Tensor = zeros((N, N)).to(self.device)
        adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
        adjacency_matrix[edges[:, 1], edges[:, 0]] = 1
        return adjacency_matrix.bool()
    
    def to_p3d(self) -> Meshes:
        return Meshes([self.vert_cords()[..., :3]], [self.faces])
    
    def transform(self, tm):
        """[transform verts]

        Args:
            tm ([tensor]): [a 3 * 4 transform matrix]
        """
        if type(tm) != Tensor:
            tm = tensor(tm)
        if tm.device != self.device:
            tm.to(self.device)
            
        # Use unsqueeze to broadcast manually to avoid bug of pytorch
        tm = tm.unsqueeze(0) # [1, 3, 4]

        # After matmul [1, 3, 4], [N, 4, 1] -> [N, 3, 1] and squeeze into [N, 3]
        self.verts = matmul(tm, homo(self.verts)).squeeze(-1)