import numpy as np
import matplotlib.pyplot as plt

class MetricGraph:
    """Graphe métrique pour la localisation de source"""
    
    def __init__(self):
        self.edges = []
        self.vertices = {}
        self.boundary_vertices = set()
        self.n_dof = 0
        self.vertex_positions = {}
        
    def add_edge(self, edge_id, v_start, v_end, length, a_coef, n_points):
        """Ajoute une arête au graphe"""
        edge = {
            'id': edge_id,
            'v_start': v_start,
            'v_end': v_end,
            'length': length,
            'a': a_coef,
            'n': n_points,
            'h': length / (n_points + 1),
            'dof_start': None,
            'dof_end': None
        }
        self.edges.append(edge)
        
        if v_start not in self.vertices:
            self.vertices[v_start] = {'edges': [], 'dof': None}
        if v_end not in self.vertices:
            self.vertices[v_end] = {'edges': [], 'dof': None}
            
        self.vertices[v_start]['edges'].append((edge_id, 'start'))
        self.vertices[v_end]['edges'].append((edge_id, 'end'))
    
    def set_vertex_position(self, v_id, x, y):
        self.vertex_positions[v_id] = (x, y)
        
    def set_boundary_vertices(self, boundary_list):
        self.boundary_vertices = set(boundary_list)
        
    def build_dof_map(self):
        dof_counter = 0
        
        for v_id in self.vertices:
            if v_id not in self.boundary_vertices:
                self.vertices[v_id]['dof'] = dof_counter
                dof_counter += 1
        
        for edge in self.edges:
            edge['dof_start'] = dof_counter
            dof_counter += edge['n']
            edge['dof_end'] = dof_counter
            
        self.n_dof = dof_counter
        print(f"Nombre total de DDL: {self.n_dof}")
        
    def get_vertex_dof(self, v_id):
        if v_id in self.boundary_vertices:
            return None
        return self.vertices[v_id]['dof']
    
    def get_edge_dofs(self, edge_id):
        edge = self.edges[edge_id]
        return list(range(edge['dof_start'], edge['dof_end']))
    
    def plot_graph(self, title="Graphe métrique 2D", vertex_labels=True, edge_labels=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for edge in self.edges:
            v_start = edge['v_start']
            v_end = edge['v_end']
            
            if v_start in self.vertex_positions and v_end in self.vertex_positions:
                x1, y1 = self.vertex_positions[v_start]
                x2, y2 = self.vertex_positions[v_end]
                
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)
                
                if edge_labels:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"E{edge['id']}", 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=9, ha='center')
        
        for v_id, pos in self.vertex_positions.items():
            x, y = pos
            if v_id in self.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=12, label='Bord' if v_id == list(self.boundary_vertices)[0] else '')
            else:
                ax.plot(x, y, 'go', markersize=12, label='Interne' if v_id == list(set(self.vertices.keys()) - self.boundary_vertices)[0] else '')
            
            if vertex_labels:
                ax.text(x, y + 0.15, v_id, fontsize=11, ha='center', fontweight='bold')
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
