import os
import torch
from torch import rand, tensor, matmul, Tensor, ones, cat
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance,
    point_mesh_edge_distance,
)
from StylizedModel.BaseMesh import BaseMesh
from StylizedModel.HeadMesh import HeadMesh
from StylizedModel.ModelUtils import get_is_boundary
from tqdm.notebook import tqdm

from NNModels.SpatialNet import SpatialNet

import plotly.graph_objects as go
from ipywidgets import widgets
from utils import homo

class FittingConfigs:
    def __init__(self) -> None:
        self.Niter = 4000
        self.w_distance = 80000.0
        self.w_edge = 1
        self.w_normal = 0.01
        self.w_laplacian = 1
        self.plot_period = 500
        self.w_landmarks = 0.5
        self.w_sym = 1000.0
        self.BETA = 10.0
        self.GAMA = 2.0
        self.w_neck_stable = 10.0
        self.AGGREGATE_ITER_NUM = 5

c = FittingConfigs()

def point_to_edge_distance(meshes: Meshes, pcls: Pointclouds):
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for edges
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()
    max_segms = meshes.num_edges_per_mesh().max().item()

    # point to edge distance: shape (P,)
    from pytorch3d.loss.point_mesh_distance import point_edge_distance
    point_to_edge = point_edge_distance(
        points, points_first_idx, segms, segms_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i), )
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_edge = point_to_edge * weights_p
    point_dist = point_to_edge.sum() / N

    return point_dist
'''
# Visualization
def plot_mesh(figure, mesh, color = 'lightblue'):
    cx, cy, cz = mesh.verts_packed().clone().detach().cpu().squeeze().unbind(1)
    ii, ij, ik = mesh.faces_packed().clone().detach().cpu().squeeze().unbind(1)
    figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, color = color, opacity = 0.5)

def compare_meshs(src_mesh, trg_mesh, title=""):
    figure = go.FigureWidget()
    plot_mesh(figure, src_mesh, color = 'lightpink')
    plot_mesh(figure, trg_mesh, color = 'lightblue')
    figure.update_layout(width=800, height=600)
    figure.show(renderer="vscode")
'''

# Hyper parameters
def get_stiffness(ii, device):
    s = [12, 5, -2, -7, -7]
    idx = ii * (len(s) - 1) // c.Niter
    l = c.Niter * idx // (len(s) - 1)
    r = c.Niter * (idx + 1) // (len(s) - 1)
    w = ((r - ii) * s[idx] + (ii - l) * s[idx + 1]) / (r - l)
    return tensor(w).exp().to(device)

def get_landmarkness(ii, device):
    s = [1, 2, 3, 3]
    idx = ii * (len(s) - 1) // c.Niter
    l = c.Niter * idx // (len(s) - 1)
    r = c.Niter * (idx + 1) // (len(s) - 1)
    w = ((r - ii) * s[idx] + (ii - l) * s[idx + 1]) / (r - l)
    return tensor(w).exp().to(device)

def init_register(src_mesh, trg_mesh, device):
    trg_mesh.transform(tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.3]],
                               requires_grad = False).to(device))
    
    scale = tensor(1.0).to(device).requires_grad_()
    offset = tensor([0.0, 0.0, 0.0]).to(device).requires_grad_()
    optimizer = torch.optim.Adam([scale, offset], lr = 1e-4)
    
    for i in range(500):
        optimizer.zero_grad()
        new_trg_vers = scale * trg_mesh.components['Face'].vert_cords() + offset
        loss = point_mesh_face_distance(src_mesh.to_p3d(), Pointclouds([new_trg_vers]))
        loss.backward()
        optimizer.step()
    scale.requires_grad = False
    offset.requires_grad = False
    trg_mesh.transform(cat((scale * torch.eye(3).to(device), offset.unsqueeze(1)), 1))

def register(src_head: HeadMesh, trg_head: HeadMesh, vision = None, loop = None):
    assert 'Face' in src_head.components and 'Face' in trg_head.components
    assert 'LeftEyeRim' in src_head.holes and 'LeftEyeRim' in trg_head.holes
    assert 'RightEyeRim' in src_head.holes and 'RightEyeRim' in trg_head.holes
    assert src_head.device == trg_head.device
    
    device = src_head.device
    
    init_register(src_head, trg_head, device)
    
    src_face_mesh = BaseMesh(src_head.components['Face'].vert_cords(), src_head.components['Face'].faces, device = device)
    trg_face_mesh = BaseMesh(trg_head.components['Face'].vert_cords(), trg_head.components['Face'].faces, device = device)
    
    if vision is not None:
        vision.init_mesh(src_face_mesh, trg_face_mesh)

    src_verts = src_face_mesh.verts
    trg_verts = trg_face_mesh.verts
    src_faces = src_face_mesh.faces
    trg_faces = trg_face_mesh.faces

    adjacency_matrix = src_face_mesh.get_adjacency_matrix()
    adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(dim = 1, keepdims = True)
    is_boundary = get_is_boundary(trg_face_mesh)

    model = SpatialNet().to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    
    base_trans = tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]).to(device)

    for i in range(50):
        optimizer.zero_grad()
        loss = (model(torch.rand(300, 3).to(device) * 2 - 1) - base_trans).pow(2.0).sum()
        loss.backward()
        optimizer.step()

    if loop is None:
        loop = tqdm(range(c.Niter))
    for i in loop:

        optimizer.zero_grad()

        trans_mat = model(src_verts) # [*, 3] -> [*, 3, 4]
        
        # After matmul [1, 3, 4], [N, 4, 1] -> [N, 3, 1] and squeeze into [N, 3]
        new_verts = matmul(trans_mat, homo(src_verts)).squeeze() # [N, 3]

        # Calculate useful new verts
        pair_distance = (new_verts.unsqueeze(1) - trg_verts.unsqueeze(0)).pow(2.0).sum(dim = 2)
        argmin_distance = torch.argmin(pair_distance, dim = 1)
        
        if i < c.Niter * 0.25:
            good_boundary = matmul(adjacency_matrix, is_boundary[argmin_distance])
            
        useful_new_verts = new_verts[good_boundary < 1]

        loss_deform = tensor(0).to(device)

        num_rand = 1000
        rand_verts = torch.rand(num_rand, 3).to(device) * 2.4 - 1.2
        rand_transmat = model(rand_verts)
        
        diff = (rand_transmat.unsqueeze(1) - rand_transmat).pow(2.0).mean((-1, -2))
        rand_dis = (rand_verts.unsqueeze(1) - rand_verts).pow(2.0).sum(-1).maximum(tensor(1e-4).to(device))
        loss_deform = loss_deform + 0.2 * (diff / rand_dis).mean()

        iter_trans_mat = trans_mat.clone()
        for aggi in range(c.AGGREGATE_ITER_NUM):
            iter_trans_mat = (matmul(adjacency_matrix, iter_trans_mat.permute(1, 2, 0).unsqueeze(3))).squeeze().permute(2, 0, 1)
            loss_deform = loss_deform + (trans_mat - iter_trans_mat).pow(2.0).mean() / c.AGGREGATE_ITER_NUM
        

        # print(trg_face.to_p3d(), Pointclouds([useful_new_verts]))
        loss_distance = point_mesh_face_distance(trg_face_mesh.to_p3d(), Pointclouds([useful_new_verts]))
        
        
        new_src_mesh = Meshes(verts = [new_verts], faces = [src_faces])
        loss_normal = mesh_normal_consistency(new_src_mesh)
        
        loss = tensor(0.0, device = device)
        for hole_name in ['LeftEyeRim', 'RightEyeRim', 'Mouth']:
            if hole_name in src_head.holes and hole_name in trg_head.holes:
                loss = loss + get_landmarkness(i, device) * point_mesh_edge_distance(
                Meshes(verts = [trg_head.holes[hole_name].vert_cords()],
                    faces = [trg_head.holes[hole_name].faces]), 
                    Pointclouds([new_verts[src_head.holes[hole_name].verts]]))
        
        
        # calculate final loss
        loss = loss + loss_distance * c.w_distance
        loss = loss + loss_deform * get_stiffness(i, device)
        loss = loss + loss_normal * c.w_normal
        
        rand_verts = torch.cat((src_verts, rand_verts))
        sym_rand_verts = rand_verts * tensor([-1,1,1]).to(device)
        rand_transmat, sym_rand_transmat = model(rand_verts), model(sym_rand_verts)

        loss = loss + c.w_sym * (rand_transmat - sym_rand_transmat).pow(2.0).mean()
        
        loss = loss + c.w_sym * (rand_transmat.matmul(homo(rand_verts)) - sym_rand_transmat.matmul(homo(sym_rand_verts)) * tensor([[-1],[1],[1]]).to(device)).pow(2.0).mean()

        neck_verts = src_head.holes['Neck'].vert_cords()

        neck_trans_mat = model(neck_verts)[..., :3] # [*, 3, 3]

        # loss = loss + 2 * (neck_trans_mat - neck_trans_mat.mean() * torch.eye(3).to(cur_device)).pow(2.0).sum()
        loss = loss + c.w_neck_stable * (neck_trans_mat * tensor([[0,1,1],[1,0,1],[1,1,0]]).to(device)).pow(2.0).sum()
        
        rand_verts = torch.rand(1000, 3).to(device) * 2 - 1
        rand_verts = rand_verts / rand_verts.pow(2.0).sum(dim = -1, keepdim=True).sqrt() * 1.2
        
        new_rand_verts = model(rand_verts).matmul(homo(rand_verts)).squeeze()

        loss = loss + 10.0 * (rand_verts - new_rand_verts).pow(2.0).mean()

        # Plot mesh
        loop.set_description(str(loss.item()))
        loss.backward()
        optimizer.step()
        
        if i % 300 == 0 and vision is not None:
            vision.update(new_verts, useful_new_verts)

    return model


