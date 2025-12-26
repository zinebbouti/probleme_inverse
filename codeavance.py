
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

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


class SourceLocalization:
    """Résolution du problème de localisation de source avec méthode adjointe"""
    
    def __init__(self, graph):
        self.graph = graph
        self.u = None  # Solution du problème direct
        self.w = None  # Sensibilité
        self.p = None  # État adjoint
        
    def source_function(self, x, epsilon, intensity=1.0, width=0.05):
        """Fonction source gaussienne centrée en epsilon"""
        return intensity * np.exp(-((x - epsilon)**2) / (2 * width**2))
    
    # def source_function_mms(self, x, edge):
    #     """
    #     Source manufacturée pour la validation MMS
    #     """
    #     L = edge['length']
    #     a = edge['a']
    #     return a * (np.pi / L)**2 * np.sin(np.pi * x / L)
    
    def source_function_mms(self, x, edge):
        a = edge['a']
        L = edge['length']
        eid = edge['id']

        C = 1.0
        A1 = 0.0
        if eid == 0:
            A = A1
        elif eid == 1:
            A = 2.0 * C / L**2 - A1
        else:
            A = 0.0

        B = 1.0

        wpp = 2.0 * L**2 - 12.0 * L * x + 12.0 * x**2  # w''(x)
        return 2.0 * a * A - a * B * wpp




    
    def source_derivative(self, x, epsilon, intensity=1.0, width=0.05):
        """Dérivée de la source par rapport à epsilon"""
        gauss = np.exp(-((x - epsilon)**2) / (2 * width**2))
        return intensity * (x - epsilon) / width**2 * gauss
    
    def assemble_system(self, epsilon_dict=None, source_intensity=1.0):
        """Assemble le système linéaire A*u = g"""
        n = self.graph.n_dof
        A = lil_matrix((n, n))
        g = np.zeros(n)

        for edge in self.graph.edges:
            edge_id = edge['id']
            h = edge['h']
            a = edge['a']
            n_pts = edge['n']

            x = np.linspace(h, edge['length'] - h, n_pts)
            dofs = self.graph.get_edge_dofs(edge_id)

            # =========================
            # CHOIX DE LA SOURCE
            # =========================
            if self.validation_mode:
                g_local = self.source_function_mms(x, edge)
            elif epsilon_dict is not None and edge_id in epsilon_dict:
                epsilon = epsilon_dict[edge_id]
                g_local = self.source_function(x, epsilon, source_intensity)
            else:
                g_local = np.zeros(n_pts)

            stiffness = a / h

            for i, dof in enumerate(dofs):
                g[dof] += h * g_local[i]

                A[dof, dof] = 2 * stiffness

                if i > 0:
                    A[dof, dofs[i-1]] = -stiffness
                else:
                    v_start_dof = self.graph.get_vertex_dof(edge['v_start'])
                    if v_start_dof is not None:
                        A[dof, v_start_dof] = -stiffness

                if i + 1 < len(dofs):
                    A[dof, dofs[i+1]] = -stiffness
                else:
                    v_end_dof = self.graph.get_vertex_dof(edge['v_end'])
                    if v_end_dof is not None:
                        A[dof, v_end_dof] = -stiffness
        
        for v_id in self.graph.vertices:
            if v_id in self.graph.boundary_vertices:
                continue

            v_dof = self.graph.get_vertex_dof(v_id)
            incident_edges = self.graph.vertices[v_id]['edges']

            for edge_id, position in incident_edges:
                edge = self.graph.edges[edge_id]
                h = edge['h']
                a = edge['a']

                dofs = self.graph.get_edge_dofs(edge_id)
                npts = len(dofs)

                # Il faut au moins 2 points internes pour ordre 2 au nœud
                if npts >= 2:
                    if position == 'start':
                        u1 = dofs[0]   # point à h
                        u2 = dofs[1]   # point à 2h
                    else:
                        u1 = dofs[-1]  # point à L-h (adjacent au nœud)
                        u2 = dofs[-2]  # point à L-2h

                    # Kirchhoff ordre 2 :
                    # sum_e a * (-3 u_v + 4 u1 - u2)/(2h) = 0
                    # -> diag positive :
                    coeff = a / (2.0 * h)
                    A[v_dof, v_dof] += 3.0 * coeff
                    A[v_dof, u1]    += -4.0 * coeff
                    A[v_dof, u2]    += 1.0 * coeff

                else:
                    # Fallback ordre 1 si pas assez de points
                    u1 = dofs[0] if position == 'start' else dofs[-1]
                    coeff = a / h
                    A[v_dof, v_dof] += coeff
                    A[v_dof, u1]    += -coeff

        return A.tocsr(), g
    
    def assemble_sensitivity_rhs(self, epsilon_dict, edge_id_sens, source_intensity=1.0):
        """Assemble le second membre pour l'équation de sensibilité"""
        n = self.graph.n_dof
        g_sens = np.zeros(n)
        
        if edge_id_sens not in epsilon_dict:
            print(f"Warning: edge_id {edge_id_sens} not in epsilon_dict")
            return g_sens
        
        edge = self.graph.edges[edge_id_sens]
        h = edge['h']
        n_pts = edge['n']
        
        x = np.linspace(h, edge['length'] - h, n_pts)
        dofs = self.graph.get_edge_dofs(edge_id_sens)
        
        epsilon = epsilon_dict[edge_id_sens]
        g_prime = self.source_derivative(x, epsilon, source_intensity)
        
        for i, dof in enumerate(dofs):
            g_sens[dof] = g_prime[i]
        
        return g_sens
    
    def solve_direct(self, epsilon_dict=None, source_intensity=1.0):
        self.validation_mode = True
        A, g = self.assemble_system(epsilon_dict, source_intensity)
        self.u = spsolve(A, g)
        return self.u

    
    def solve_sensitivity(self, epsilon_dict, edge_id_sens, source_intensity=1.0):
        """Résout l'équation de sensibilité: A*w = dg/d(epsilon)"""
        A, _ = self.assemble_system(epsilon_dict, source_intensity)
        g_sens = self.assemble_sensitivity_rhs(epsilon_dict, edge_id_sens, source_intensity)
        self.w = spsolve(A, g_sens)
        return self.w
    
    def compute_sensitivity_all_edges(self, epsilon_dict, source_intensity=1.0):
        """Calcule la sensibilité pour toutes les arêtes"""
        sensitivities = {}
        
        print("\n" + "="*70)
        print("CALCUL DES SENSIBILITÉS SUR TOUTES LES ARÊTES")
        print("="*70)
        
        for edge_id in epsilon_dict.keys():
            print(f"\nCalcul de sensibilité pour arête {edge_id}...")
            w = self.solve_sensitivity(epsilon_dict, edge_id, source_intensity)
            sensitivities[edge_id] = w
        
        return sensitivities
    
    def compute_sensitivity_error_per_edge(self, epsilon_dict, source_intensity=1.0, delta=1e-6):
        """Valide les sensibilités par différences finies"""
        errors = {}
        u_base = self.solve_direct(epsilon_dict, source_intensity)

        for edge_id in epsilon_dict:
            w = self.solve_sensitivity(epsilon_dict, edge_id, source_intensity)

            eps_pert = epsilon_dict.copy()
            eps_pert[edge_id] += delta
            u_pert = self.solve_direct(eps_pert, source_intensity)
            w_fd = (u_pert - u_base) / delta

            dofs = self.graph.get_edge_dofs(edge_id)
            err = np.linalg.norm(w[dofs] - w_fd[dofs]) / np.linalg.norm(w_fd[dofs])
            errors[edge_id] = err

        return errors

    # ========================================================================
    # ÉQUATION ADJOINTE
    # ========================================================================
    
    def assemble_adjoint_rhs(self, u_data, flux_data=None, varpi=0.0):
        """
        Assemble le second membre pour l'équation adjointe
        
        L'équation adjointe est: A^T * p = -dJ/du
        
        IMPORTANT: Pour la fonctionnelle J = 1/2 ∫(u - u_data)² dx,
        on a dJ/du = ∫(u - u_data)·φ dx
        
        Après discrétisation par différences finies avec intégration rectangulaire:
        dJ/du_i = h·(u_i - u_data,i)
        
        Donc le second membre est: -h·(u - u_data)
        Le facteur h est ESSENTIEL pour la cohérence dimensionnelle !
        
        Parameters:
        -----------
        u_data : array
            Données de référence (même taille que self.u)
        flux_data : dict {vertex_id: flux_value}, optional
            Flux de référence aux sommets de bord
        varpi : float
            Poids de la contribution flux dans la fonctionnelle
        
        Returns:
        --------
        rhs : array
            Second membre -dJ/du
        """
        n = self.graph.n_dof
        rhs = np.zeros(n)
        
        if self.u is None:
            raise ValueError("Résolvez d'abord le problème direct (solve_direct)")
        
        # === Contribution volumique: -h·(u - u_data) ===
        # Le facteur h vient de l'intégration numérique !
        for edge in self.graph.edges:
            edge_id = edge['id']
            h = edge['h']  # ← FACTEUR CRUCIAL
            dofs = self.graph.get_edge_dofs(edge_id)
            
            # Terme volumique avec poids d'intégration
            rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])
        
        # === Contribution flux aux sommets de bord (optionnel) ===
        if varpi > 0 and flux_data is not None:
            for v_id in self.graph.boundary_vertices:
                for edge_id, pos in self.graph.vertices[v_id]['edges']:
                    edge = self.graph.edges[edge_id]
                    a = edge['a']
                    h = edge['h']
                    dofs = self.graph.get_edge_dofs(edge_id)
                    
                    # Calculer le flux numérique au sommet
                    if pos == 'start':
                        flux_num = a * (self.u[dofs[0]] - 0.0) / h
                        # Dérivée de J2 par rapport à u[dofs[0]]
                        rhs[dofs[0]] -= varpi * (flux_num - flux_data[v_id]) * (a / h)
                    else:  # pos == 'end'
                        flux_num = a * (0.0 - self.u[dofs[-1]]) / h
                        # Dérivée de J2 par rapport à u[dofs[-1]]
                        rhs[dofs[-1]] -= varpi * (flux_num - flux_data[v_id]) * (-a / h)
        
        return rhs
    
    def solve_adjoint(self, u_data, flux_data=None, varpi=0.0, epsilon_dict=None, source_intensity=1.0):
        """
        Résout l'équation adjointe: A^T * p = -dJ/du
        
        L'état adjoint p permet de calculer efficacement le gradient
        de la fonctionnelle J par rapport aux paramètres epsilon.
        
        Note: Comme A est symétrique (différences finies centrées),
        on a A^T = A, donc on résout: A * p = -dJ/du
        
        Parameters:
        -----------
        u_data : array
            Données de référence
        flux_data : dict, optional
            Flux de référence aux sommets de bord
        varpi : float
            Poids de la contribution flux
        epsilon_dict : dict, optional
            Positions des sources (pour assembler A si nécessaire)
        source_intensity : float
            Intensité des sources
        
        Returns:
        --------
        p : array
            État adjoint solution de A^T * p = -dJ/du
        """
        # Assembler la matrice A
        if epsilon_dict is not None:
            A, _ = self.assemble_system(epsilon_dict, source_intensity)
        else:
            if self.u is None:
                raise ValueError("Fournir epsilon_dict ou résoudre d'abord le problème direct")
            A, _ = self.assemble_system({}, source_intensity)
        
        # Assembler le second membre de l'équation adjointe
        rhs_adjoint = self.assemble_adjoint_rhs(u_data, flux_data, varpi)
        
        # Résoudre A^T * p = rhs (A est symétrique donc A^T = A)
        self.p = spsolve(A.T, rhs_adjoint)
        
        return self.p
    
    def compute_gradient_from_adjoint(self, epsilon_dict, edge_id, source_intensity=1.0):
        """
        Calcule le gradient dJ/d(epsilon) via l'état adjoint
        
        Formule: dJ/dε = -p^T * dg/dε
        
        où:
        - p est l'état adjoint (solution de A^T * p = -dJ/du)
        - dg/dε est la dérivée de la source par rapport à epsilon
        
        Cette méthode est beaucoup plus efficace que les différences finies
        car elle ne nécessite qu'une seule résolution adjointe pour obtenir
        le gradient par rapport à TOUS les paramètres.
        
        Parameters:
        -----------
        epsilon_dict : dict
            Positions actuelles des sources
        edge_id : int
            ID de l'arête pour laquelle calculer le gradient
        source_intensity : float
            Intensité de la source
        
        Returns:
        --------
        gradient : float
            dJ/d(epsilon) pour l'arête spécifiée
        """
        if self.p is None:
            raise ValueError("Résolvez d'abord l'équation adjointe (solve_adjoint)")
        
        # Assembler dg/dε
        dg_deps = self.assemble_sensitivity_rhs(epsilon_dict, edge_id, source_intensity)
        
        # Calculer le produit scalaire: gradient = -p^T * dg/dε
        gradient = -np.dot(self.p, dg_deps)
        
        return gradient
    
    def compute_gradient_all_edges(self, epsilon_dict, source_intensity=1.0):
        """
        Calcule le gradient pour toutes les arêtes avec sources
        
        Returns:
        --------
        gradients : dict {edge_id: gradient}
        """
        if self.p is None:
            raise ValueError("Résolvez d'abord l'équation adjointe (solve_adjoint)")
        
        gradients = {}
        for edge_id in epsilon_dict.keys():
            grad = self.compute_gradient_from_adjoint(epsilon_dict, edge_id, source_intensity)
            gradients[edge_id] = grad
        
        return gradients
    
    # ========================================================================
    # FONCTIONNELLE DE COÛT
    # ========================================================================
    
    def compute_cost_functional(self, u, u_data, flux_data, varpi=0.0):
        """
        Calcule la fonctionnelle de coût J(ε)
        
        J = J1 + J2
        où:
        J1 = 1/2 ∑_e ∫ (u - u_data)^2 dx
        J2 = varpi/2 ∑_v (flux_num - flux_data)^2
        """
        J1 = 0.0

        # Contribution volumique
        for edge in self.graph.edges:
            dofs = self.graph.get_edge_dofs(edge['id'])
            h = edge['h']
            diff = u[dofs] - u_data[dofs]
            J1 += 0.5 * np.sum(diff**2) * h

        # Contribution flux
        J2 = 0.0
        if varpi > 0 and flux_data is not None:
            for v_id in self.graph.boundary_vertices:
                for edge_id, pos in self.graph.vertices[v_id]['edges']:
                    edge = self.graph.edges[edge_id]
                    a = edge['a']
                    h = edge['h']
                    dofs = self.graph.get_edge_dofs(edge_id)

                    if pos == 'start':
                        dudn = (u[dofs[0]] - 0.0) / h
                    else:
                        dudn = (0.0 - u[dofs[-1]]) / h

                    J2 += 0.5 * varpi * (a * dudn - flux_data[v_id])**2

        return J1 + J2
    
    def compute_gradient_finite_diff(self, epsilon_dict, edge_id, u_data, flux_data,
                                     source_intensity, varpi=0.0, delta=1e-6):
        """
        Calcule le gradient par différences finies (pour validation)
        
        grad ≈ (J(ε + δ) - J(ε)) / δ
        """
        # Coût initial
        self.solve_direct(epsilon_dict, source_intensity)
        J1 = self.compute_cost_functional(self.u, u_data, flux_data, varpi)
        
        # Coût perturbé
        epsilon_perturbed = epsilon_dict.copy()
        epsilon_perturbed[edge_id] += delta
        self.solve_direct(epsilon_perturbed, source_intensity)
        J2 = self.compute_cost_functional(self.u, u_data, flux_data, varpi)
        
        grad_fd = (J2 - J1) / delta
        
        # Restaurer la solution originale
        self.solve_direct(epsilon_dict, source_intensity)
        
        return grad_fd
    
    # ========================================================================
    # VISUALISATION
    # ========================================================================
    
    def plot_solution_on_graph(self, epsilon_dict=None, title="Solution sur le graphe"):
        """Visualise la solution u sur tout le graphe 2D"""
        if self.u is None:
            print("Aucune solution à afficher.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        for edge in self.graph.edges:
            v_start = edge['v_start']
            v_end = edge['v_end']
            edge_id = edge['id']
            
            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]
                
                dofs = self.graph.get_edge_dofs(edge_id)
                u_edge = self.u[dofs]
                
                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)
                
                scatter = ax.scatter(x_interp, y_interp, c=u_edge, cmap='coolwarm', 
                                   s=80, vmin=self.u.min(), vmax=self.u.max(), zorder=2)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.3, zorder=1)
                
                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge['length']
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(x_source, y_source, 'y*', markersize=25, 
                           markeredgecolor='black', markeredgewidth=2, zorder=3,
                           label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=14, zorder=4, 
                       label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
            else:
                ax.plot(x, y, 'go', markersize=14, zorder=4,
                       label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
        plt.colorbar(scatter, ax=ax, label='Valeur de u')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=10, loc='best')
        plt.tight_layout()
        plt.show()
    
    def plot_adjoint_on_graph(self, epsilon_dict=None, title="État adjoint p"):
        """Visualise l'état adjoint p sur tout le graphe 2D"""
        if self.p is None:
            print("Aucun état adjoint à afficher.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        for edge in self.graph.edges:
            v_start = edge['v_start']
            v_end = edge['v_end']
            edge_id = edge['id']
            
            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]
                
                dofs = self.graph.get_edge_dofs(edge_id)
                p_edge = self.p[dofs]
                
                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)
                
                scatter = ax.scatter(x_interp, y_interp, c=p_edge, cmap='viridis', 
                                   s=80, vmin=self.p.min(), vmax=self.p.max(), zorder=2)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.3, zorder=1)
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=14, zorder=4)
            else:
                ax.plot(x, y, 'go', markersize=14, zorder=4)
        
        plt.colorbar(scatter, ax=ax, label='Valeur de p')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_sensitivity_comparison(self, sensitivities, epsilon_dict):
        """Compare les sensibilités sur toutes les arêtes"""
        n_edges = len(sensitivities)
        fig, axes = plt.subplots(1, n_edges, figsize=(6*n_edges, 5))
        
        if n_edges == 1:
            axes = [axes]
        
        for idx, (edge_id, w) in enumerate(sensitivities.items()):
            edge = self.graph.edges[edge_id]
            h = edge['h']
            n_pts = edge['n']
            x = np.linspace(h, edge['length'] - h, n_pts)
            dofs = self.graph.get_edge_dofs(edge_id)
            
            axes[idx].plot(x, w[dofs], 'b-o', linewidth=2, markersize=5)
            
            if edge_id in epsilon_dict:
                epsilon = epsilon_dict[edge_id]
                axes[idx].axvline(epsilon, color='orange', linestyle=':', linewidth=2,
                                label=f'Source ε={epsilon:.2f}')
            
            axes[idx].set_xlabel('Position x', fontsize=11)
            axes[idx].set_ylabel('∂u/∂ε', fontsize=11)
            axes[idx].set_title(f'Sensibilité - Arête {edge_id}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=9)
        
        plt.tight_layout()
        plt.show()


# ============= EXEMPLES D'UTILISATION =============

# def exact_solution_mms(x, edge):
#             return np.sin(np.pi * x / edge.length)
def exact_solution_mms(x, edge):
    L = edge['length']
    eid = edge['id']

    C = 1.0
    A1 = 0.0
    # Kirchhoff (2 arêtes, même a et même L) -> A2 = 2C/L^2 - A1
    if eid == 0:
        A = A1
    elif eid == 1:
        A = 2.0 * C / L**2 - A1
    else:
        A = 0.0

    B = 1.0  # amplitude du terme degré 4 (tu peux changer)

    return C * (1.0 - x / L) + A * x * (L - x) + B * (x**2) * ((L - x)**2)


    
def compute_errors_mms(graph, u_num):
    """
    Calcule les erreurs L1, L2, Linf sur tout le graphe métrique
    (uniquement sur les DDL d'arêtes)
    """
    L1 = 0.0
    L2 = 0.0
    Linf = 0.0

    for edge in graph.edges:
        edge_id = edge['id']
        L = edge['length']
        n = edge['n']
        h = L / (n + 1)

        dofs = graph.get_edge_dofs(edge_id)

        for i, dof in enumerate(dofs):
            x = (i + 1) * h
            u_exact = exact_solution_mms(x, edge)
            diff = abs(u_num[dof] - u_exact)

            L1 += h * diff
            L2 += h * diff**2
            Linf = max(Linf, diff)

    return L1, np.sqrt(L2), Linf

def plot_solution_all_edges(graph, u_num):
    import numpy as np
    import matplotlib.pyplot as plt

    for edge in graph.edges:
        edge_id = edge['id']
        L = edge['length']
        n = edge['n']
        h = L / (n + 1)

        dofs = graph.get_edge_dofs(edge_id)

        x = np.array([(i + 1) * h for i in range(n)])
        u_num_edge = np.array([u_num[dof] for dof in dofs])

        # Solution exacte MMS degré > 2
        u_exact = np.array([
            graph.solver.exact_solution_mms(xi, edge)
            if hasattr(graph, "solver") else np.sin(np.pi * xi / L)
            for xi in x
        ])

        plt.figure(figsize=(6, 4))
        plt.plot(x, u_exact, 'r-', lw=2, label="Solution exacte")
        plt.plot(x, u_num_edge, 'bo', ms=4, label="Solution numérique")

        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title(f"Comparaison sur l’arête {edge_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    

def create_2d_graph_example():
    """Crée un graphe 2D simple"""
    graph = MetricGraph()
    
    positions = {
        'v0': (0, 2),
        'v1': (2, 2),
        'v2': (2, 0),
        'v3': (0, 0),
        'v4': (1, 1),
    }
    
    for v_id, pos in positions.items():
        graph.set_vertex_position(v_id, pos[0], pos[1])
    
    graph.add_edge(0, 'v3', 'v2', length=2.0, a_coef=1.0, n_points=30)
    graph.add_edge(1, 'v3', 'v0', length=2.0, a_coef=1.0, n_points=30)
    graph.add_edge(2, 'v3', 'v4', length=np.sqrt(2), a_coef=1.0, n_points=20)
    
    graph.set_boundary_vertices(['v0', 'v1', 'v2'])
    
    return graph


def example_adjoint_validation():
    """
    Validation de l'équation adjointe par comparaison avec différences finies
    
    EXPLICATION DE LA MÉTHODE ADJOINTE:
    ====================================
    
    1. PROBLÈME D'OPTIMISATION:
       min J(ε) = 1/2 ∫(u(ε) - u_data)² dx
       
       où u(ε) est solution de: A·u = g(ε)
    
    2. CALCUL DU GRADIENT:
       
       a) MÉTHODE NAÏVE (différences finies):
          Pour N paramètres → N+1 résolutions du problème direct
          dJ/dε_i ≈ [J(ε + δe_i) - J(ε)] / δ
          COÛT: O(N) résolutions
       
       b) MÉTHODE ADJOINTE (smart):
          - Résoudre une fois: A·u = g(ε)
          - Résoudre une fois: A^T·p = -∂J/∂u
          - Calculer: dJ/dε = -p^T · ∂g/∂ε
          COÛT: O(1) résolutions (2 au total, indépendant de N!)
    
    3. POURQUOI ÇA MARCHE?
       
       Par la règle de dérivation en chaîne:
       dJ/dε = (∂J/∂u)^T · (du/dε)
       
       Or du/dε satisfait: A·(du/dε) = ∂g/∂ε
       
       En introduisant l'adjoint p tel que: A^T·p = -∂J/∂u
       On obtient par produit scalaire:
       dJ/dε = -p^T · ∂g/∂ε
       
       → Pas besoin de calculer du/dε explicitement !
    
    4. INTERPRÉTATION PHYSIQUE:
       - u : état direct (propagation de la source vers les mesures)
       - p : état adjoint (rétro-propagation de l'erreur)
       - p indique comment chaque point influence la fonctionnelle J
    """
    print("\n" + "="*70)
    print("VALIDATION DE L'ÉQUATION ADJOINTE")
    print("="*70)
    print("\nCette validation compare deux méthodes de calcul du gradient:")
    print("  • Méthode adjointe (efficace): 2 résolutions")
    print("  • Différences finies (coûteuse): N+1 résolutions")
    print("="*70)
    
    # Créer le graphe
    graph = create_2d_graph_example()
    graph.build_dof_map()
    
    # Positions des sources
    epsilon_dict = {0: 1.5, 1: 0.7}
    
    # Paramètres
    source_intensity = 10.0
    varpi = 0.0  # Pas de terme de flux pour simplifier
    
    print("\n1. Résolution du problème direct...")
    solver = SourceLocalization(graph)
    u = solver.solve_direct(epsilon_dict, source_intensity)
    
    # Créer des données synthétiques (légèrement bruitées)
    u_data = u + 0.01 * np.random.randn(len(u))
    flux_data = {v: 0.0 for v in graph.boundary_vertices}
    
    # Calculer la fonctionnelle
    J = solver.compute_cost_functional(u, u_data, flux_data, varpi)
    print(f"Fonctionnelle J = {J:.6e}")
    
    print("\n2. Résolution de l'équation adjointe...")
    p = solver.solve_adjoint(u_data, flux_data, varpi, epsilon_dict, source_intensity)
    
    # Visualiser l'état adjoint
    print("\n3. Visualisation de l'état adjoint...")
    solver.plot_adjoint_on_graph(epsilon_dict, title="État adjoint p(x)")
    
    print("\n4. Calcul des gradients via méthode adjointe...")
    gradients_adjoint = solver.compute_gradient_all_edges(epsilon_dict, source_intensity)
    
    print("\n5. Calcul des gradients par différences finies (validation)...")
    gradients_fd = {}
    for edge_id in epsilon_dict.keys():
        grad_fd = solver.compute_gradient_finite_diff(
            epsilon_dict, edge_id, u_data, flux_data, source_intensity, varpi
        )
        gradients_fd[edge_id] = grad_fd
    
    # Comparaison
    print("\n" + "="*70)
    print("COMPARAISON DES GRADIENTS")
    print("="*70)
    print("\nInterprétation:")
    print("  • Gradient positif → déplacer ε vers la droite AUGMENTE J")
    print("  • Gradient négatif → déplacer ε vers la droite DIMINUE J")
    print("  • |Gradient| grand → forte sensibilité de J à ε")
    print("-"*70)
    print(f"{'Arête':<10} {'Adjoint':<20} {'Diff. Finies':<20} {'Erreur Rel.':<15}")
    print("-"*70)
    
    for edge_id in epsilon_dict.keys():
        grad_adj = gradients_adjoint[edge_id]
        grad_fd = gradients_fd[edge_id]
        
        if abs(grad_fd) > 1e-10:
            err_rel = abs(grad_adj - grad_fd) / abs(grad_fd)
        else:
            err_rel = abs(grad_adj - grad_fd)
        
        print(f"{edge_id:<10} {grad_adj:<20.8e} {grad_fd:<20.8e} {err_rel:<15.2e}")
    
    print("="*70)
    
    # Vérification globale
    all_errors = [abs(gradients_adjoint[e] - gradients_fd[e]) / abs(gradients_fd[e]) 
                  for e in epsilon_dict.keys() if abs(gradients_fd[e]) > 1e-10]
    
    max_error = max(all_errors) if all_errors else 0.0
    
    print(f"\nErreur relative maximale: {max_error:.2e}")
    
    # Seuil de validation plus réaliste pour les différences finies
    if max_error < 1e-3:
        print("✓ VALIDATION RÉUSSIE! Les gradients adjoints sont corrects.")
        print("  (Erreur < 0.1% : excellente précision)")
    elif max_error < 1e-2:
        print("✓ VALIDATION ACCEPTABLE. Les gradients adjoints sont fiables.")
        print("  (Erreur < 1% : bonne précision)")
    else:
        print("⚠ Attention: erreurs importantes détectées.")
        print("  Vérifier l'implémentation ou réduire delta.")
    
    print("="*70)


def example_2d_sensitivity_study():
    """Étude de sensibilité complète sur un graphe 2D"""
    print("\n" + "="*70)
    print("ÉTUDE DE SENSIBILITÉ SUR GRAPHE 2D")
    print("="*70)
    
    graph = create_2d_graph_example()
    graph.build_dof_map()
    
    print("\n1. Visualisation de la structure du graphe...")
    graph.plot_graph(title="Structure du graphe métrique 2D")
    
    epsilon_dict = {0: 1.5, 1: 0.7}
    
    print("\n2. Résolution du problème direct...")
    solver = SourceLocalization(graph)
    u = solver.solve_direct(epsilon_dict, source_intensity=10.0)
    
    print("\n3. Visualisation de la solution sur le graphe 2D...")
    solver.plot_solution_on_graph(epsilon_dict, title="Solution u(x) sur le graphe 2D")
    
    print("\n4. Calcul des sensibilités...")
    sensitivities = solver.compute_sensitivity_all_edges(epsilon_dict, source_intensity=10.0)
    
    print("\n5. Comparaison des sensibilités...")
    solver.plot_sensitivity_comparison(sensitivities, epsilon_dict)
    
    print("\n" + "="*70)
    print("STATISTIQUES")
    print("="*70)
    print(f"Nombre d'arêtes avec sources: {len(epsilon_dict)}")
    print(f"Nombre total de DDL: {graph.n_dof}")
    print(f"\nNorme L2 de la solution: {np.linalg.norm(u):.4f}")
    
    for edge_id, w in sensitivities.items():
        print(f"Norme L2 de sensibilité (arête {edge_id}): {np.linalg.norm(w):.4f}")
    
    print("="*70)
    errors = solver.compute_sensitivity_error_per_edge(epsilon_dict, source_intensity=10.0)

    for e, err in errors.items():
        print(f"Erreur relative sensibilité pour arête {e} : {err:.2e}")


def example_validation_1d():
    """Validation MMS sur graphe 1D avec jonction - étude de convergence"""

    print("\n" + "="*70)
    print("VALIDATION MMS - CONVERGENCE")
    print("="*70)

    Ns = [10, 20, 40, 80, 160]

    errors_L1 = []
    errors_L2 = []
    errors_Linf = []
    hs = []

    for N in Ns:
        graph = MetricGraph()
        graph.add_edge(0, 'v1', 'v2', length=1.0, a_coef=1.0, n_points=N)
        graph.add_edge(1, 'v1', 'v3', length=1.0, a_coef=1.0, n_points=N)

        graph.set_vertex_position('v1', 0, 0)
        graph.set_vertex_position('v2', 1, 0)
        graph.set_vertex_position('v3', 0, 1)
        graph.set_boundary_vertices(['v2', 'v3'])
        graph.build_dof_map()

        solver = SourceLocalization(graph)
        solver.validation_mode = True

        u_num = solver.solve_direct()

        L1, L2, Linf = compute_errors_mms(graph, u_num)

        h = 1.0 / (N + 1)

        hs.append(h)
        errors_L1.append(L1)
        errors_L2.append(L2)
        errors_Linf.append(Linf)

        print(f"N={N:4d} | L1={L1:.3e} | L2={L2:.3e} | Linf={Linf:.3e}")

    # ============================
    # Calcul des ordres observés
    # ============================
    def compute_orders(errs):
        return [
            np.log(errs[i-1] / errs[i]) / np.log(2.0)
            for i in range(1, len(errs))
        ]

    orders_L1 = compute_orders(errors_L1)
    orders_L2 = compute_orders(errors_L2)
    orders_Linf = compute_orders(errors_Linf)

    print("\nOrdres observés (entre deux raffinements) :")
    for i in range(len(orders_L1)):
        print(f"N={Ns[i]}→{Ns[i+1]} | "
              f"L1={orders_L1[i]:.2f}, "
              f"L2={orders_L2[i]:.2f}, "
              f"Linf={orders_Linf[i]:.2f}")

    # ============================
    # Courbe de convergence
    # ============================
    plt.figure(figsize=(7, 5))
    plt.loglog(hs, errors_L2, 'o-', label=r"$\|e\|_{L^2}$")
    plt.loglog(
        hs,
        errors_L2[0] * (np.array(hs) / hs[0])**2,
        '--',
        label="Référence ordre 2"
    )

    plt.gca().invert_xaxis()
    plt.xlabel("h")
    plt.ylabel("Erreur")
    plt.title("Convergence MMS – norme $L^2$")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    plot_solution_all_edges(graph, u_num)



    
    # print("\n1. Résolution du problème direct...")
    # u1 = solver.solve_direct(epsilon_dict, source_intensity=10.0)
    
    # print("2. Résolution de l'équation de sensibilité...")
    # w_sensitivity = solver.solve_sensitivity(epsilon_dict, edge_id_sens=0, source_intensity=10.0)
    
    # print("3. Calcul de la sensibilité par différences finies...")
    # delta = 1e-6
    # epsilon_perturbed = {0: epsilon_dict[0] + delta}
    # u2 = solver.solve_direct(epsilon_perturbed, source_intensity=10.0)
    # w_fd = (u2 - u1) / delta
    
    # error = np.linalg.norm(w_sensitivity - w_fd) / np.linalg.norm(w_fd)
    # print(f"\n✓ Erreur relative: {error:.2e}")
    
    # if error < 1e-4:
    #     print("✓ VALIDATION RÉUSSIE!")
    # else:
    #     print("⚠ Erreur élevée, vérifier les paramètres")
    
    # print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ANALYSE COMPLÈTE SUR GRAPHE MÉTRIQUE 2D")
    print("="*70)
    print("\nCe code permet:")
    print("  1. Création de graphes métriques 2D")
    print("  2. Visualisation de la structure du graphe")
    print("  3. Résolution du problème direct")
    print("  4. Calcul de sensibilité")
    print("  5. Résolution de l'équation adjointe")
    print("  6. Calcul du gradient via méthode adjointe")
    print("  7. Validation par différences finies")
    print("="*70 + "\n")
    
    # Validation 1D rapide
    example_validation_1d()
    
    
    # Étude complète sur graphe 2D
    #example_2d_sensitivity_study()
    
    # Validation de l'équation adjointe
    #example_adjoint_validation()
    
    print("\n" + "="*70)
    print("✓ ANALYSE TERMINÉE")
    print("="*70)
