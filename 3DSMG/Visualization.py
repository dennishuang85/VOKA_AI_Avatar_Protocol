import plotly.graph_objects as go

from StylizedModel.BaseMesh import BaseMesh

class Visualization:
    def __init__(self, figure):
        self.figure = figure
        self.src_mesh_plot = None
    
    def init_mesh(self, src_face: BaseMesh, trg_face: BaseMesh):
        
        if self.figure is None:
            return
        self.figure.data = []
        cx, cy, cz = src_face.verts[...,:3].clone().detach().cpu().squeeze().unbind(1)
        ii, ij, ik = src_face.faces.clone().detach().cpu().squeeze().unbind(1)
        self.figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, color = 'lightblue', opacity = 0.5)
        
        cx, cy, cz = trg_face.verts[...,:3].clone().detach().cpu().squeeze().unbind(1)
        ii, ij, ik = trg_face.faces.clone().detach().cpu().squeeze().unbind(1)
        self.figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, color = 'lightpink', opacity = 0.5)
        self.src_mesh_plot = self.figure.data[0]
    
    def update(self, new_verts, useful_new_verts):
        if self.figure is None:
            return
        cx, cy, cz = new_verts.clone().detach().cpu().squeeze().unbind(1)
        self.src_mesh_plot.x = cx
        self.src_mesh_plot.y = cy
        self.src_mesh_plot.z = cz
        self.figure.data = self.figure.data[0:2]
        cx, cy, cz = useful_new_verts.clone().detach().cpu().squeeze().unbind(1)
        self.figure.add_scatter3d(x = cx, y = cy, z = cz, mode = 'markers',
            marker = dict(size = 2, opacity = 0.5))

def show(figure, mesh, color = 'lightpink', w = None):
    vertices = mesh.get_vertices()
    cx = [vertex[0] for vertex in vertices]
    cy = [vertex[1] for vertex in vertices]
    cz = [vertex[2] for vertex in vertices]

    faces = mesh.get_faces()
    ii = [face[0] for face in faces]
    ij = [face[1] for face in faces]
    ik = [face[2] for face in faces]

    if w is not None:
        figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, color = color, opacity = 0.5, hovertext = w)
    else:
        figure.add_mesh3d(x = cx, y = cy, z = cz, i = ii, j = ij, k = ik, color = color, opacity = 0.5)

    if mesh.fixed_points is not None:
        fixed_points = mesh.fixed_points[1]
        fx = [point[0] for point in fixed_points]
        fy = [point[1] for point in fixed_points]
        fz = [point[2] for point in fixed_points]

        figure.add_scatter3d(x = fx, y = fy, z = fz, mode = 'markers')
    return cx, cy, cz, ii, ij, ik


def compare(source_mesh, target_mesh, w = None):
    figure = go.Figure()
    show(figure, source_mesh, color = 'lightpink', w = w)
    show(figure, target_mesh, color = 'lightblue')
    figure.show()


if __name__ == '__main__':
    pass
