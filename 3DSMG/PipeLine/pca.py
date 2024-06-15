import torch, os, sys
sys.path.append(".")

from StylizedModel.IO.Load import load as model_load
from utils import walk_file
from torch import cat, save, load, Tensor, svd, tensor
import cv2
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def PCA(X: Tensor, k):
    shape = X[0].shape
    print(X.shape, X.shape[:1] + tuple([X[0].numel()]))
    X = X.reshape(X.shape[:1] + tuple([X[0].numel()]))
    X_mean = X.mean(dim = 0)
    X = X - X_mean
    
    u, s, v = svd(X)
    
    X_mean = X_mean.reshape(shape)
    comp = v[..., :k].reshape(shape + tuple([k]))
    
    return X_mean, comp


src_path = 'data/models/template/template.fbx'
src_mesh = model_load(src_path ,device)

'''
trg_file_list = walk_file('data/models/registered/')

base, shapes = [], []

for trg_name, trg_file in trg_file_list:
    trg_mesh = model_load(trg_file, device)
    
    base.append(trg_mesh.verts.unsqueeze(0))
    shapes.append((trg_mesh.shapes - trg_mesh.verts).unsqueeze(0))
    
save(cat(base), 'tmp/base.pt')
save(cat(shapes), 'tmp/shapes.pt')
'''
'''

base = load('tmp/base.pt').to(device)
shapes = load('tmp/shapes.pt').to(device).permute((0, 2, 3, 1))

from Generator.GenBase import GenBase
from Generator.GenMesh import GenMesh

base, comp = PCA(base, 20)
x0 = GenBase(base, comp)

base, comp = PCA(shapes, 20)
x1 = GenBase(base, comp)

mesh_gen = GenMesh(x0, x1, tensor(src_mesh.info["landmarks"]).to(device))

save(mesh_gen, "data/generators/mesh_gen_rk20.pt")
'''
trg_file_list = walk_file('data/textures/', ['.png', '.jpg'])

images = []

for trg_name, trg_file in trg_file_list:
    
    image = cv2.imread(trg_file)
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    image = cv2.resize(image, (1024, 1024))
    images.append(tensor(image, dtype = torch.float).unsqueeze(0) / 256)
    print(images[-1].shape)

images = cat(images).to(device)
 
base, comp = PCA(images, 8)
from Generator.GenBase import GenBase
text_gen = GenBase(base, comp)
save(text_gen, "data/generators/text_gen_rk20.pt")