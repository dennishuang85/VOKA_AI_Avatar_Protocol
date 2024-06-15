
from fbx import FbxManager, FbxImporter, FbxIOSettings, FbxScene, FbxExporter, FbxVector4, FbxLayerElement, FbxVector2
from typing import List, Tuple

class FBXHandler():
    def __init__(self, file_path):
        self.fbx_manager = FbxManager.Create() # create memory manager for FBX SDK
        io_settings = FbxIOSettings.Create(self.fbx_manager, "") # create IO settings
        from fbx import EXP_FBX_EMBEDDED
        io_settings.SetBoolProp(EXP_FBX_EMBEDDED, True) 
        self.fbx_manager.SetIOSettings(io_settings)
        importer = FbxImporter.Create(self.fbx_manager, "")
        status = importer.Initialize(file_path, -1, self.fbx_manager.GetIOSettings())
        if status == 0:
            raise Exception("Import failed.")
        self.scene = FbxScene.Create(self.fbx_manager, "Base Scene")
        importer.Import(self.scene) # Import the contents of the file into the scene.
        importer.Destroy() # Destroy the importer
        
        self.num_materials = 0
    
    def get_mesh(self, mtrl_map) -> Tuple[List, List, List]:
        root_node = self.scene.GetRootNode()
        model = root_node.GetChild(0)
        mesh = model.GetMesh()
        
        verts = []
        for point in mesh.GetControlPoints():
            verts.append((point[0], point[1], point[2]))
        
        faces = []
        UVs = []
        
        layer = mesh.GetLayer(0)
        uv = layer.GetUVs()
        new_mtrl_map = []
        assert uv.GetMappingMode() == FbxLayerElement.eByPolygonVertex
        
        for i in range(mesh.GetPolygonCount()):
            sz = mesh.GetPolygonSize(i)
            for j in range(sz - 2):
                faces.append((mesh.GetPolygonVertex(i, 0), mesh.GetPolygonVertex(i, j + 1), mesh.GetPolygonVertex(i, j + 2)))
                new_uv = []
                for k in [0, j + 1, j + 2]:
                    vec = FbxVector2(0, 0)
                    mesh.GetPolygonVertexUV(i, k, uv.GetName(), vec)
                    new_uv.append((vec[0], vec[1]))
                UVs.append(new_uv)
                if len(mtrl_map) == 1:
                    new_mtrl_map.append(0)
                else:
                    new_mtrl_map.append(mtrl_map[i])
        
        return verts, faces, UVs, new_mtrl_map
    
    def get_shapes(self) -> List:
        root_node = self.scene.GetRootNode()
        model = root_node.GetChild(0)
        mesh = model.GetMesh()
        
        shapes = []
        for i in range(mesh.GetShapeCount()):
            shape = []
            for point in mesh.GetShape(0, i, 0).GetControlPoints():
                shape.append((point[0], point[1], point[2]))
            shapes.append(shape)
        
        return shapes
    
    def get_materials(self) -> Tuple[List[int], List[str]]:
        
        root_node = self.scene.GetRootNode()
        model = root_node.GetChild(0)
        mesh = model.GetMesh()
        layer = mesh.GetLayer(0)
        
        layer_material = layer.GetMaterials()
        
        assert layer_material.GetMappingMode() == FbxLayerElement.eByPolygon \
            or layer_material.GetMappingMode() == FbxLayerElement.eAllSame
        
        mtrl_map = [idx for idx in layer_material.GetIndexArray()]
        
        textures = []
        for i in range(model.GetMaterialCount()):
            texture = model.GetMaterial(i).Diffuse.GetSrcObject(0)
            textures.append(texture.GetFileName())
        
        return mtrl_map, textures