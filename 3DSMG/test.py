from StylizedModel.IO.Load import load
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mesh = load("data/template/template.fbx", device)

