from asyncio.log import logger
from StylizedModel.BaseMesh import BaseMesh
from torch import Tensor, eye, tensor, cat, ones, matmul
from typing import Dict, Optional, List
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import join_meshes_as_scene
import plotly.graph_objects as go
import torch, json, os
from PIL import Image
import numpy as np
from utils import homo

from StylizedModel.CompMesh import CompMesh

class HeadMesh(BaseMesh):
    
    def __init__(self, verts = None, faces = None, device = None, **kwargs):
        BaseMesh.__init__(self, verts, faces, device)
        self.info: Dict = {}
        self.components: Dict = {}
        self.holes: Dict = {}
        self.shapes: Optional[Tensor] = None
        self.mtrl_map: List = None
        self.textures = None
        self.handler = None
        self.uvs = None
        self.faces_by_mtl = None
        
        self.maps = None
        self.faces_uvs = None
        self.verts_uvs = None
        self.group_verts = None
        self.group_faces = None
        # self.get_add_info()
        '''
        '''
    def get_cp_name(self, idx):
        if self.info is None:
            return None
        if 'component_name' not in self.info:
            return None
        if str(idx) not in self.info['component_name']:
            return None
        return self.info['component_name'][str(idx)]
    
    def get_hl_name(self, idx):
        if self.info is None:
            return None
        if 'hole_name' not in self.info:
            return None
        if str(idx) not in self.info['hole_name']:
            return None
        return self.info['hole_name'][str(idx)]
    
    def transform(self, tm):
        """[transform verts]

        Args:
            tm ([tensor]): [a 3 * 4 transform matrix]
        """
        if type(tm) != Tensor:
            tm = tensor(tm)
        if tm.device != self.device:
            tm.to(self.device)
        
        assert tm.shape[0] == 3 and tm.shape[1] == 4
         
        # Use unsqueeze to broadcast manually to avoid bug of pytorch
        tm = tm.unsqueeze(0) # [1, 3, 4]
        # After matmul [1, 3, 4], [N, 4, 1] -> [N, 4, 1] and squeeze into [N, 3]
        self.verts = matmul(tm, homo(self.verts)).squeeze(-1)
        
        if self.shapes is not None:
            tm = tm.unsqueeze(0) # [1, 1, 4, 4]
            # Do same thing for shapes.
            # After matmul [1, 1, 3, 4], [S, N, 4, 1] -> [S, N, 3, 1] and squeeze into [S, N, 3]
            self.shapes = matmul(tm, homo(self.shapes)).squeeze(-1)


    def normalize(self):
        center = (self.verts.max(dim = 0).values + self.verts.min(dim = 0).values) / 2
        
        matrix = eye(4).to(self.device) # [4, 4]
        matrix[0:3, 3] = -center[:3]

        self.transform(matrix[:3])
        
        scale = self.verts.pow(2.0).sum(-1).max().pow(0.5)
        matrix = eye(4).to(self.device) / scale
        
        self.transform(matrix[:3])
                      
    def to_textured_p3d(self, gened_texture = None) -> Meshes:

        # tex_maps is a dictionary of {material name: texture image}.
        # Take the first image:
        tex = None
        if self.handler is not None and self.mtrl_map is not None:
            
            assert len(self.mtrl_map) == self.faces.shape[0]
            
            M = self.faces.shape[0]
            
            if self.faces_by_mtl == None:
                self.faces_by_mtl = []
                for i in range(max(self.mtrl_map) + 1):
                    self.faces_by_mtl.append([[], []])
                for i in range(M):
                    self.faces_by_mtl[self.mtrl_map[i]][0].append(self.faces[i].cpu().numpy())
                    self.faces_by_mtl[self.mtrl_map[i]][1].append(self.uvs[i].cpu().numpy())
                for i in range(len(self.faces_by_mtl)):
                    self.faces_by_mtl[i][0] = tensor(np.array(self.faces_by_mtl[i][0])).to(self.device)
                    self.faces_by_mtl[i][1] = tensor(np.array(self.faces_by_mtl[i][1])).to(self.device)

            group_verts = []
            group_faces = []
            maps = []
            verts_uvs = []
            faces_uvs = []
            
            for i in range(len(self.faces_by_mtl)):
                group_verts.append(self.verts[..., :3])
                group_faces.append(self.faces_by_mtl[i][0])
                M = self.faces_by_mtl[i][1].shape[0]
                verts_uvs.append(self.faces_by_mtl[i][1].view(M * 3, 2))
                faces_uvs.append(torch.arange(M * 3).to(self.device).view((M, 3)))

                texture_image = np.array(Image.open(self.textures[i]))
                maps.append(tensor(texture_image, dtype = torch.float).to(self.device) / 256)
            # Create a textures object
            self.maps = maps
            self.verts_uvs = verts_uvs
            self.faces_uvs = faces_uvs
            self.group_verts = group_verts
            self.group_faces = group_faces
        if gened_texture is not None:
            self.maps[0] = gened_texture
        
        tex = TexturesUV(verts_uvs=self.verts_uvs, faces_uvs=self.faces_uvs, maps=self.maps)
        
        return join_meshes_as_scene(Meshes(verts = self.group_verts, faces = self.group_faces, textures = tex))
    
    def show(self):
        """
        Visualize the mesh based on plotly.
        """
        figure = go.FigureWidget()
        colors = ['#B0E0E6', '#E3CF57', '#FF9912', '#4169E1',
                  '#FF6100', '#FFD700', '#00FF00', '#FF6100',
                  '#3D9140', '#D2691E', '#FCE6C9', '#FFDEAD']
        
        for idx in self.components:
            cpt: CompMesh = self.components[idx]
            cx, cy, cz = cpt.vert_cords().clone().detach().cpu().squeeze().unbind(-1)
            ii, ij, ik = cpt.get_faces().clone().detach().cpu().squeeze().unbind(-1)
            from random import choice
            figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, 
                              color = choice(colors), opacity = 0.5, name = idx)
        for idx in self.holes:
            face_mesh: CompMesh = self.components['Face']
            hole: BaseMesh = self.holes[idx]
            cx, cy, cz = hole.vert_cords().clone().detach().cpu().squeeze().unbind(1)
            figure.add_scatter3d(x = cx, y = cy, z = cz, mode = 'markers',
                                marker = dict(size = 4, opacity = 0.5), name = idx)
            
        figure.show()
        
    def __add__(self, other):
        raise Exception("You should not do that.")
    
    def update_scene(self):
        from fbx import FbxVector4
        root_node = self.handler.scene.GetRootNode()
        model_node = root_node.GetChild(0)
        mesh = model_node.GetMesh()

        for i in range(mesh.GetShapeCount()):
            verts = self.shapes[i]
            for j in range(len(mesh.GetShape(0, i, 0).GetControlPoints())):
                mesh.GetShape(0, i, 0).SetControlPointAt(FbxVector4(verts[j][0].item(), verts[j][1].item(), verts[j][2].item()), j)
        
        for j in range(len(mesh.GetControlPoints())):
            mesh.SetControlPointAt(FbxVector4(self.verts[j][0].item(), self.verts[j][1].item(), self.verts[j][2].item()), j)

    def transform_with_model(self, model: torch.nn.Module):
        model.to(self.device)
        self.verts = matmul(model(self.verts), homo(self.verts)).squeeze()

        for i in range(len(self.shapes)):
            self.shapes[i] = matmul(model(self.shapes[i]), homo(self.shapes[i])).squeeze()
