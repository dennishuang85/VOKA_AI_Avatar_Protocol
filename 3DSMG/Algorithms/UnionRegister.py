import os
import idna
from numpy import VisibleDeprecationWarning
import torch
from torch import rand, tensor, matmul, Tensor, ones, cat, nn, zeros
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
    point_mesh_face_distance,
    point_mesh_edge_distance,
)
from NNModels.Deformer import Deformer
from NNModels.Identifier import Identifier
from StylizedModel.BaseMesh import BaseMesh
from StylizedModel.HeadMesh import HeadMesh
from StylizedModel.ModelUtils import get_is_boundary
from tqdm.notebook import tqdm

from NNModels.SpatialNet import SpatialNet
from random import randint
from typing import List
import plotly.graph_objects as go
from ipywidgets import widgets
from utils import homo

class RegisterConfigs:
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
        self.identity_rank = 20
        self.num_rand_verts = 100
        self.w_stable = 10.0

c = RegisterConfigs()

class Dsource(nn.Module):
    def __init__(self, mesh: BaseMesh, holes, num_target: int):
        super().__init__()
        self.mesh = mesh
        self.device = mesh.device
        self.deformer = Deformer(c.identity_rank)
        self.identifier = Identifier(num_target, c.identity_rank)
        self.adjacency_matrix = mesh.get_adjacency_matrix()
        self.adjacency_matrix = self.adjacency_matrix / self.adjacency_matrix.sum(dim = 1, keepdims = True)
        self.holes = holes
    
    def cal_energy(self, identity):
        rand_verts = torch.rand(c.num_rand_verts, 3).to(self.device) * 2.4 - 1.2
        rand_transmat = self.deformer(rand_verts, identity)
        
        new_rand_verts = matmul(rand_transmat, homo(rand_verts)).squeeze()
        
        mid_verts = (rand_verts.unsqueeze(1) + rand_verts) / 2.0
        mid_trans = self.deformer(mid_verts, identity)
        # [N, N, 3, 4] matmul [N, N, 4, 1] -> [N, N, 3, 1] and squeeze to [N, N, 3]
        new_mid_verts = matmul(mid_trans, homo(mid_verts)).squeeze()
        
        # the square of distance between the two random verts
        rand_dis = (rand_verts.unsqueeze(1) - rand_verts).pow(2.0).sum(-1).maximum(tensor(1e-6).to(self.device))
        
        loss = (new_rand_verts.unsqueeze(1) + new_rand_verts - 2.0 * new_mid_verts) / rand_dis
        return c.w_energy * loss
    
    def cal_topolo(self, trans_mat):
        
        loss = Tensor(0.0).to(self.device)
        iter_trans_mat = trans_mat.clone()
        for aggi in range(c.AGGREGATE_ITER_NUM):
            iter_trans_mat = (matmul(self.adjacency_matrix, iter_trans_mat.permute(1, 2, 0).unsqueeze(3))).squeeze().permute(2, 0, 1)
            loss = loss + (trans_mat - iter_trans_mat).pow(2.0).mean() / c.AGGREGATE_ITER_NUM
        return c.w_topolo * loss
    
    def cal_symmet(self, identity):
        loss = Tensor(0.0).to(self.device)
        rand_verts = torch.cat((self.mesh.vert_cords(), rand_verts))
        sym_rand_verts = rand_verts * tensor([-1,1,1]).to(self.device)
        rand_transmat, sym_rand_transmat = self.deformer(rand_verts, identity), self.deformer(sym_rand_verts, identity)
        loss = loss + (rand_transmat - sym_rand_transmat).pow(2.0).mean()
        loss = loss + (rand_transmat.matmul(homo(rand_verts)) - sym_rand_transmat.matmul(homo(sym_rand_verts)) * tensor([[-1],[1],[1]]).to(self.device)).pow(2.0).mean()
        
        return c.w_sym * loss
    
    def cal_stable(self, identity):
        
        rand_verts = torch.rand(1000, 3).to(self.device) * 2 - 1
        rand_verts = rand_verts / rand_verts.pow(2.0).sum(dim = -1, keepdim=True).sqrt() * 1.2
        new_rand_verts = self.deformer(rand_verts).matmul(homo(rand_verts)).squeeze()
        loss = (rand_verts - new_rand_verts).pow(2.0).mean()
        return c.w_stable * loss
    
    def cal_loss(self, trans_mat, identity):
        return self.cal_energy(identity) + self.cal_topolo(trans_mat) + self.cal_symme(identity)
    
    def forward(self, id, x = None):
        if x == None:
            x = self.mesh.vert_cords()
        identity = self.identifier(id)
        trans_mat = self.deformer(x, identity)
        return trans_mat, self.cal_loss(trans_mat, identity)
    
    def to(self, device):
        super().to(device)
        self.device = device
        self.adjacency_matrix.to(device)
    
class Dtarget(nn.Module):
    def __init__(self, mesh: BaseMesh, holes, index: Tensor):
        super(Dtarget, self).__init__()
        self.device = mesh.device
        self.verts = mesh
        self.holes = holes
        self.scale = nn.Parameter(ones(1))
        self.offset = nn.Parameter(zeros(3))
        self.is_boundary = get_is_boundary(mesh) # with size [N]
        self.index = index
    
    def forward(self):
        return self.mesh.vert_cords() * self.scale + self.offset, torch.log(self.scale), self.cal_loss()
    
    def cal_loss(self):
        loss = torch.log(self.scale).pow(2.0)
        return loss
    def to(self, device):
        super(Dtarget, self).to(device)
        self.device = device

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

def cal_loss(src: Dsource, trg: Dtarget, epoch_cnt: int, vision, device):
    
        trans_mat, loss = src(trg.index)
        
        new_verts = matmul(trans_mat, homo(src.mesh.vert_cords())).squeeze() # [N, 3]

        # Calculate useful new verts
        pair_distance = (new_verts.unsqueeze(1) - trg.mesh.vert_cords().unsqueeze(0)).pow(2.0).sum(dim = -1)
        argmin_distance = torch.argmin(pair_distance, dim = 1)
        
        good_boundary = matmul(src.adjacency_matrix, trg.is_boundary[argmin_distance])
            
        useful_new_verts = new_verts[good_boundary < 1]

        loss_deform = tensor(0).to(device)

        # print(trg_face.to_p3d(), Pointclouds([useful_new_verts]))
        loss_distance = point_mesh_face_distance(trg.mesh.to_p3d(), Pointclouds([useful_new_verts]))
        
        new_src_mesh = Meshes(verts = [new_verts], faces = [src.mesh.faces])
        loss_normal = mesh_normal_consistency(new_src_mesh)
        
        loss = tensor(0.0, device = device)
        for hole_name in ['LeftEyeRim', 'RightEyeRim', 'Mouth']:
            if hole_name in src.holes and hole_name in trg.holes:
                loss = loss + get_landmarkness(epoch_cnt, device) * point_mesh_edge_distance(
                Meshes(verts = [trg.holes[hole_name].vert_cords()],
                    faces = [trg.holes[hole_name].faces]), 
                    Pointclouds([new_verts[src.holes[hole_name].verts]]))
        
        
        # calculate final loss
        loss = loss + loss_distance * c.w_distance
        loss = loss + loss_deform * get_stiffness(epoch_cnt, device)
        loss = loss + loss_normal * c.w_normal
        

        neck_verts = src.holes['Neck'].verts
        neck_trans_mat = trans_mat[neck_verts][..., :3] # [*, 3, 3]

        loss = loss + c.w_neck_stable * (neck_trans_mat * tensor([[0,1,1],[1,0,1],[1,1,0]]).to(device)).pow(2.0).sum()
        
        # Plot mesh
        
        if randint(0, 1000) == 0 and vision is not None:
            vision.update(new_verts, useful_new_verts)
        return loss


def register(src_head: HeadMesh, trg_heads: List[HeadMesh], vision = None, loop = None):
    for trg_head in trg_heads:
        assert src_head.device == trg_head.device
    
    device = src_head.device
    
    src_face_mesh = BaseMesh(src_head.components['Face'].vert_cords(), src_head.components['Face'].faces, device = device)
    src = Dsource(src_face_mesh, src_head.holes, len(trg_heads)).to(device)
    
    trgs = []
    for i, trg_head in enumerate(trg_heads):
        trg_face_mesh = BaseMesh(trg_head.components['Face'].vert_cords(), trg_head.components['Face'].faces, device = device)
        index = zeros(len(trg_heads)).to(device)
        index[i] = 1.0
        trgs.append(Dtarget(trg_face_mesh, trg_head.holes, index).to(device))
    
    if vision is not None:
        vision.init_mesh(src_face_mesh, trg_face_mesh)

    parameters = src.parameters()
    for trg in trgs:
        parameters += trg.parameters()
        
    optimizer = torch.optim.AdamW(parameters, lr = 1e-4)
    
    base_trans = tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]]).to(device)

    for i in range(100):
        loss = Tensor(0.0).to(device)
        for trg in len(trgs):
            cur_trans, _loss = src(trg.index, torch.rand(300, 3).to(device) * 2 - 1)
            loss += (cur_trans - base_trans).pow(2.0).sum()
            loss += _loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if loop is None:
        loop = tqdm(range(c.Niter))
    
    for i in loop:
        
        loss = Tensor(0.0).to(device)
        for trg in trgs:
            loss += cal_loss(src, trg, i, vision, device)

        loop.set_description(str(loss.item()))
        loss.backward()
        optimizer.step()

    return src.deformer


