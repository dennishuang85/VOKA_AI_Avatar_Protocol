from torch import Tensor, tensor, zeros
from StylizedModel.CompMesh import CompMesh
from StylizedModel.HeadMesh import HeadMesh
from StylizedModel.BaseMesh import BaseMesh
from typing import List, Tuple, Dict
import torch, logging
from sys import setrecursionlimit
setrecursionlimit(5000)

logger = logging.getLogger()

def split(obj: HeadMesh):
    visited = {}
    table = obj.get_edge_table()
    result = []
    N = obj.verts.shape[0]
    for i in range(N):
        if i in visited and visited[i] == True:
            continue
        result.append(dfs(i, visited, table))
    return result

def add_edge(u, v, table):
    if u not in table:
        table[u] = []
    table[u].append(v)
        
def dfs(u, visited, table):
    visited[u], ret = True, [u]
    if u in table:
        for v in table[u]:
            if v in visited and visited[v] == True:
                continue
            ret += dfs(v, visited, table)
    return ret

def get_edge_state(faces: Tensor) -> Dict[Tuple[int, int], int]:
    edge_set = {}
    def add_edge(u, v):
        if v < u:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set[(u, v)] = 0
        edge_set[(u, v)] += 1
    for v0, v1, v2 in faces.tolist():
        add_edge(v0, v1)
        add_edge(v1, v2)
        add_edge(v2, v0)
    return edge_set

def get_bad_edges(faces: Tensor) -> List[Tuple[int, int]]:
    edge_set = get_edge_state(faces)
    bad_edges = []
    for u, v in edge_set:
        if edge_set[(u, v)] != 2:
            bad_edges.append((u, v))
    return bad_edges

def get_is_boundary(obj: BaseMesh) -> Tensor:
    is_boundary = zeros(obj.get_verts().shape[0])
    for u, v in get_bad_edges(obj.get_faces()):
        assert type(u) == int and type(v) == int
        is_boundary[u], is_boundary[v] = 1, 1
    return is_boundary.to(obj.device)

def get_holes(obj: HeadMesh):
    if 'Face' not in obj.components:
        return {}
    face_mesh:BaseMesh = obj.components['Face']
    is_boundary = get_is_boundary(obj.components['Face'])
    N = face_mesh.verts.shape[0]

    visited = {}
    table = {}
    
    bad_edges = get_bad_edges(face_mesh.get_faces())
    for u, v in bad_edges:
        add_edge(u, v, table)
        add_edge(v, u, table)
    
    holes = {}
    for i in range(N):
        if is_boundary[i] != 1 or i in visited:
            continue
        new_verts = dfs(i, visited, table)
        new_faces = []
        for u, v in bad_edges:
            if u in new_verts and v in new_verts:
                new_faces.append([new_verts.index(u), new_verts.index(u), new_verts.index(v)])
                
        hole_name = len(holes)
        holes[hole_name] = CompMesh(tensor(new_verts, device = obj.device, dtype = torch.long),
                                    tensor(new_faces, device = obj.device, dtype = torch.long),
                                    device = obj.device, father = face_mesh)
    return holes

def get_components(obj: HeadMesh):
    components = {}
    faces = obj.get_faces().tolist() # with shape [M, 3]
    
    for i, cp_verts in enumerate(split(obj)):
        cp_verts.sort()
        new_idx = {}
        for j, idx in enumerate(cp_verts):
            new_idx[idx] = j
        
        cp_faces = []
        cp_cfaces = []
        
        for face_idx, face in enumerate(faces):
            for j in range(3):
                if face[j] in cp_verts:
                    cp_faces.append([new_idx[k] for k in face])
                    cp_cfaces.append(face_idx)
                    break
            
        cp_mesh = CompMesh(verts = tensor(cp_verts, dtype = torch.long).to(obj.device), 
                           faces = tensor(cp_faces, dtype = torch.long).to(obj.device),
                           cfaces = tensor(cp_cfaces, dtype = torch.long).to(obj.device),
                           device = obj.device, father = obj)
        
        components[i] = cp_mesh
    return components

def find_eye_rims(obj: HeadMesh):
    LeftEyeRim = None
    RightEyeRim = None
    face_mesh:BaseMesh = obj.components['Face']
    face_mesh_verts = face_mesh.vert_cords()
    for hole_id in obj.holes:
        min_bound = obj.holes[hole_id].vert_cords().min(dim = 0).values
        max_bound = obj.holes[hole_id].vert_cords().max(dim = 0).values
        if obj.holes[hole_id].vert_cords().shape[0] >= 10\
        and 0.5 > min_bound[1] > -0.6 \
        and -0 < max_bound[2] < 0.8\
        and 0.05 < max_bound[1] - min_bound[1] < 0.5\
        and max_bound[2] - min_bound[2] < 0.3\
        and 0.3 < max_bound[0] - min_bound[0]\
        and max_bound[2] - min_bound[2] < max_bound[0] - min_bound[0]:
            if min_bound[0] > 0:
                if LeftEyeRim is not None:
                    logger.warning("Several left eye %d %d" % (LeftEyeRim, hole_id))
                    return
                LeftEyeRim = hole_id
            elif max_bound[0] < 0:
                if RightEyeRim is not None:
                    logger.warning("Several right eye %d %d" % (RightEyeRim, hole_id))
                    return
                RightEyeRim = hole_id
    if LeftEyeRim == None or RightEyeRim == None:
        logger.warning("Cant find eye")
        return
    
    if "hole_name" not in obj.info:
        obj.info["hole_name"] = {}
        
    obj.info["hole_name"][str(LeftEyeRim)] = "LeftEyeRim"
    obj.info["hole_name"][str(RightEyeRim)] = "RightEyeRim"
    
    obj.holes["LeftEyeRim"] = obj.holes.pop(LeftEyeRim)
    obj.holes["RightEyeRim"] = obj.holes.pop(RightEyeRim)

def find_mouth(obj):
    Mouth = None
    face_mesh:BaseMesh = obj.components['Face']
    for hole_id in obj.holes:
        min_bound = obj.holes[hole_id].vert_cords().min(dim = 0).values
        max_bound = obj.holes[hole_id].vert_cords().max(dim = 0).values
        if min_bound[1] > -0.8 and max_bound[1] < -0\
        and min_bound[2] > -0 and max_bound[2] < 0.9\
        and max_bound[1] - min_bound[1] < 0.1\
        and max_bound[2] - min_bound[2] < 0.2\
        and 0.15 < max_bound[0] - min_bound[0]\
        and max_bound[2] - min_bound[2] < max_bound[0] - min_bound[0]:
            if Mouth is not None:
                print("Several mouth %d %d" % (Mouth, hole_id))
                return
            Mouth = hole_id
    if Mouth == None:
        print("Cant find mouth")
        return
    
    if "hole_name" not in obj.info:
        obj.info["hole_name"] = {}
    obj.info["hole_name"][str(Mouth)] = "Mouth"
    obj.holes["Mouth"] = obj.holes.pop(Mouth)