
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class SourceLocalization:
    """Résolution du problème de localisation de source avec méthode adjointe"""
    
    def __init__(self, graph):
        self.graph = graph
        self.u = None
        self.w = None
        self.p = None
        
    def source_function(self, x, epsilon, intensity=1.0, width=0.05):
        """Fonction source gaussienne centrée en epsilon"""
        return intensity * np.exp(-((x - epsilon)**2) / (2 * width**2))
    
    def source_derivative_epsilon(self, x, epsilon, intensity=1.0, width=0.05):
        """Dérivée de la source par rapport à epsilon: ∂g/∂ε"""
        gauss = np.exp(-((x - epsilon)**2) / (2 * width**2))
        return intensity * (x - epsilon) / width**2 * gauss
    
    def source_derivative_alpha(self, x, epsilon, width=0.05):
        """Dérivée de la source par rapport à alpha: ∂g/∂α = g/α"""
        return np.exp(-((x - epsilon)**2) / (2 * width**2))
    
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
            if epsilon_dict is not None and edge_id in epsilon_dict:
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
    
    def solve_direct(self, epsilon_dict=None, source_intensity=1.0):
        """Résout le problème direct: A*u = g"""
        A, g = self.assemble_system(epsilon_dict, source_intensity)
        self.u = spsolve(A, g)
        return self.u
    
    def assemble_sensitivity_rhs_alpha(self, epsilon_dict):
        """Assemble ∂g/∂α pour l'équation de sensibilité"""
        n = self.graph.n_dof
        dg_dalpha = np.zeros(n)
        
        for edge_id, epsilon in epsilon_dict.items():
            edge = self.graph.edges[edge_id]
            h = edge['h']
            n_pts = edge['n']
            
            x = np.linspace(h, edge['length'] - h, n_pts)
            dofs = self.graph.get_edge_dofs(edge_id)
            
            dg = self.source_derivative_alpha(x, epsilon)
            
            for i, dof in enumerate(dofs):
                dg_dalpha[dof] += h * dg[i]
        
        return dg_dalpha
    
    def solve_sensitivity_alpha(self, epsilon_dict, source_intensity=1.0):
        """Résout l'équation de sensibilité: A*w = ∂g/∂α"""
        A, _ = self.assemble_system(epsilon_dict, source_intensity)
        dg_dalpha = self.assemble_sensitivity_rhs_alpha(epsilon_dict)
        
        self.w = spsolve(A, dg_dalpha)
        return self.w
    
    def compute_cost_functional(self, u_data):
        """Calcule J = 1/2 ∫ (u - u_data)² dx"""
        if self.u is None:
            raise ValueError("Résoudre d'abord le problème direct")
        
        J = 0.0
        for edge in self.graph.edges:
            dofs = self.graph.get_edge_dofs(edge['id'])
            h = edge['h']
            diff = self.u[dofs] - u_data[dofs]
            J += 0.5 * h * np.sum(diff**2)
        
        return J
    
    def compute_gradient_finite_diff(self, epsilon_dict, u_data, source_intensity, delta=None):
        """MÉTHODE 1: Gradient par différences finies centrées"""
        if delta is None:
            delta = np.finfo(float).eps**(1/3)
        
        u_plus = self.solve_direct(epsilon_dict, source_intensity + delta)
        J_plus = 0.5 * sum(edge['h'] * np.sum((u_plus[self.graph.get_edge_dofs(edge['id'])] - 
                          u_data[self.graph.get_edge_dofs(edge['id'])])**2) 
                          for edge in self.graph.edges)
        
        u_minus = self.solve_direct(epsilon_dict, source_intensity - delta)
        J_minus = 0.5 * sum(edge['h'] * np.sum((u_minus[self.graph.get_edge_dofs(edge['id'])] - 
                           u_data[self.graph.get_edge_dofs(edge['id'])])**2) 
                           for edge in self.graph.edges)
        
        grad_fd = (J_plus - J_minus) / (2 * delta)
        self.solve_direct(epsilon_dict, source_intensity)
        
        return grad_fd
    
    def compute_gradient_sensitivity(self, epsilon_dict, u_data, source_intensity):
        """MÉTHODE 2: Gradient via sensibilité directe"""
        self.solve_direct(epsilon_dict, source_intensity)
        w = self.solve_sensitivity_alpha(epsilon_dict, source_intensity)
        
        grad_sens = 0.0
        for edge in self.graph.edges:
            dofs = self.graph.get_edge_dofs(edge['id'])
            h = edge['h']
            
            diff = self.u[dofs] - u_data[dofs]
            grad_sens += h * np.dot(diff, w[dofs])
        
        return grad_sens
    
    def assemble_adjoint_rhs(self, u_data):
        """Assemble le second membre pour l'équation adjointe"""
        n = self.graph.n_dof
        rhs = np.zeros(n)
        
        if self.u is None:
            raise ValueError("Résoudre d'abord le problème direct")
        
        for edge in self.graph.edges:
            edge_id = edge['id']
            h = edge['h']
            dofs = self.graph.get_edge_dofs(edge_id)
            
            rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])
        
        return rhs
    
    def solve_adjoint(self, epsilon_dict, u_data, source_intensity):
        """Résout l'équation adjointe: A^T * p = -∂J/∂u"""
        A, _ = self.assemble_system(epsilon_dict, source_intensity)
        rhs_adjoint = self.assemble_adjoint_rhs(u_data)
        
        self.p = spsolve(A.T, rhs_adjoint)
        return self.p
    
    def compute_gradient_adjoint(self, epsilon_dict, source_intensity):
        """MÉTHODE 3: Gradient via méthode adjointe"""
        if self.p is None:
            raise ValueError("Résoudre d'abord l'équation adjointe")
        
        dg_dalpha = self.assemble_sensitivity_rhs_alpha(epsilon_dict)
        grad_adj = -np.dot(self.p, dg_dalpha)
        
        return grad_adj
    
    def validate_gradient_three_methods(self, epsilon_dict, u_data, source_intensity, delta=None):
        """VALIDATION COMPLÈTE: Compare les 3 méthodes pour calculer dJ/dα"""
        if delta is None:
            delta = np.finfo(float).eps**(1/3)
        
        print(f"\n{'='*80}")
        print(f"VALIDATION GRADIENT dJ/dα - COMPARAISON DES 3 MÉTHODES")
        print(f"{'='*80}")
        print(f"Delta optimal (différences finies) = {delta:.3e}")
        print(f"Intensité source α = {source_intensity}")
        print(f"\nOBJECTIF: Erreur relative < 10⁻¹⁵ (précision machine)\n")
        
        print("Calcul méthode 1: Différences finies...")
        grad_fd = self.compute_gradient_finite_diff(epsilon_dict, u_data, source_intensity, delta)
        
        print("Calcul méthode 2: Sensibilité directe...")
        grad_sens = self.compute_gradient_sensitivity(epsilon_dict, u_data, source_intensity)
        
        print("Calcul méthode 3: Méthode adjointe...")
        self.solve_adjoint(epsilon_dict, u_data, source_intensity)
        grad_adj = self.compute_gradient_adjoint(epsilon_dict, source_intensity)
        
        print(f"\n{'-'*80}")
        print(f"{'MÉTHODE':<30} {'VALEUR dJ/dα':<20} {'ERREUR vs FD':<15} ")
        print(f"{'-'*80}")
        
        print(f"{'1. Différences finies':<30} {grad_fd:<20.12e} {'---':<15} ")
        
        err_sens = abs(grad_sens - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_sens - grad_fd)
        status_sens = "✓✓ EXCELLENT" if err_sens < 1e-14 else ("✓ BON" if err_sens < 1e-10 else "⚠ AMÉLIORER")
        print(f"{'2. Sensibilité directe':<30} {grad_sens:<20.12e} {err_sens:<15.3e} ")
        
        err_adj = abs(grad_adj - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_adj - grad_fd)
        status_adj = "✓✓ EXCELLENT" if err_adj < 1e-14 else ("✓ BON" if err_adj < 1e-10 else "⚠ AMÉLIORER")
        print(f"{'3. Méthode adjointe':<30} {grad_adj:<20.12e} {err_adj:<15.3e} ")
        
        err_sens_adj = abs(grad_sens - grad_adj) / abs(grad_adj) if abs(grad_adj) > 1e-16 else abs(grad_sens - grad_adj)
        print(f"\n{'-'*80}")
        print(f"Comparaison directe Sensibilité vs Adjointe:")
        print(f"  Erreur relative: {err_sens_adj:.3e}")
        status_final = "✓✓ EXCELLENT" if err_sens_adj < 1e-14 else ("✓ BON" if err_sens_adj < 1e-10 else "⚠ AMÉLIORER")
        print(f"  Status: {status_final}")
        
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ:")
        print(f"{'='*80}")
        max_err = max(err_sens, err_adj)
        print(f"Erreur maximale (vs différences finies): {max_err:.3e}")
        
        if max_err < 1e-14 and err_sens_adj < 1e-14:
            print(f"\n{'✓'*40}")
            print(f"✓✓ VALIDATION RÉUSSIE!")
            print(f"✓✓ Les 3 méthodes concordent à la précision machine (< 10⁻¹⁴)")
            print(f"{'✓'*40}")
            validation_passed = True
        else:
            validation_passed = False
        
        return {
            'grad_fd': grad_fd,
            'grad_sensitivity': grad_sens,
            'grad_adjoint': grad_adj,
            'error_sens_vs_fd': err_sens,
            'error_adj_vs_fd': err_adj,
            'error_sens_vs_adj': err_sens_adj,
            'max_error': max_err,
            'validation_passed': validation_passed,
            'delta': delta
        }
    
    # ========================================================================
    # VISUALISATIONS
    # ========================================================================
    
    def plot_solution_on_graph(self, epsilon_dict=None, title="Solution u sur le graphe"):
        """Visualise la solution u sur le graphe"""
        if self.u is None:
            print("Aucune solution à afficher.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
                                   s=100, vmin=self.u.min(), vmax=self.u.max(), zorder=2,
                                   edgecolors='black', linewidth=0.5)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge['length']
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(x_source, y_source, 'y*', markersize=30, 
                           markeredgecolor='black', markeredgewidth=2, zorder=3,
                           label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=16, zorder=4, 
                       label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
            else:
                ax.plot(x, y, 'go', markersize=16, zorder=4,
                       label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Valeur de u')
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=11, loc='best')
        plt.tight_layout()
        plt.show()
    
    def plot_sensitivity_on_graph(self, epsilon_dict=None, title="Sensibilité w = ∂u/∂α"):
        """Visualise la sensibilité w sur le graphe"""
        if self.w is None:
            print("Aucune sensibilité à afficher.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for edge in self.graph.edges:
            v_start = edge['v_start']
            v_end = edge['v_end']
            edge_id = edge['id']
            
            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]
                
                dofs = self.graph.get_edge_dofs(edge_id)
                w_edge = self.w[dofs]
                
                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)
                
                scatter = ax.scatter(x_interp, y_interp, c=w_edge, cmap='viridis', 
                                   s=100, vmin=self.w.min(), vmax=self.w.max(), zorder=2,
                                   edgecolors='black', linewidth=0.5)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge['length']
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(x_source, y_source, 'y*', markersize=30, 
                           markeredgecolor='black', markeredgewidth=2, zorder=3,
                           label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=16, zorder=4, 
                       label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
            else:
                ax.plot(x, y, 'go', markersize=16, zorder=4,
                       label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Valeur de w')
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=11, loc='best')
        plt.tight_layout()
        plt.show()
    
    def plot_adjoint_on_graph(self, epsilon_dict=None, title="État adjoint p"):
        """Visualise l'état adjoint p sur le graphe"""
        if self.p is None:
            print("Aucun état adjoint à afficher.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
                
                scatter = ax.scatter(x_interp, y_interp, c=p_edge, cmap='plasma', 
                                   s=100, vmin=self.p.min(), vmax=self.p.max(), zorder=2,
                                   edgecolors='black', linewidth=0.5)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge['length']
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(x_source, y_source, 'y*', markersize=30, 
                           markeredgecolor='black', markeredgewidth=2, zorder=3,
                           label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=16, zorder=4, 
                       label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
            else:
                ax.plot(x, y, 'go', markersize=16, zorder=4,
                       label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
        cbar = plt.colorbar(scatter, ax=ax, label='Valeur de p')
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=11, loc='best')
        plt.tight_layout()
        plt.show()
    
    def plot_all_results(self, epsilon_dict, u_data):
        """Affiche tous les résultats dans une grille 2x2"""
        fig = plt.figure(figsize=(16, 14))
        
        # Subplot 1: Solution u
        ax1 = plt.subplot(2, 2, 1)
        self._plot_on_axis(ax1, self.u, epsilon_dict, "Solution u", 'coolwarm')
        
        # Subplot 2: Sensibilité w
        ax2 = plt.subplot(2, 2, 2)
        if self.w is not None:
            self._plot_on_axis(ax2, self.w, epsilon_dict, "Sensibilité w = ∂u/∂α", 'viridis')
        else:
            ax2.text(0.5, 0.5, "Sensibilité non calculée", ha='center', va='center', fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        
        # Subplot 3: État adjoint p
        ax3 = plt.subplot(2, 2, 3)
        if self.p is not None:
            self._plot_on_axis(ax3, self.p, epsilon_dict, "État adjoint p", 'plasma')
        else:
            ax3.text(0.5, 0.5, "État adjoint non calculé", ha='center', va='center', fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # Subplot 4: Données observées
        ax4 = plt.subplot(2, 2, 4)
        self._plot_on_axis(ax4, u_data, epsilon_dict, "Données observées u_data", 'coolwarm')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_on_axis(self, ax, data, epsilon_dict, title, cmap):
        """Fonction auxiliaire pour tracer sur un axe donné"""
        if data is None:
            return
        
        for edge in self.graph.edges:
            v_start = edge['v_start']
            v_end = edge['v_end']
            edge_id = edge['id']
            
            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]
                
                dofs = self.graph.get_edge_dofs(edge_id)
                data_edge = data[dofs]
                
                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)
                
                scatter = ax.scatter(x_interp, y_interp, c=data_edge, cmap=cmap, 
                                   s=80, vmin=data.min(), vmax=data.max(), zorder=2,
                                   edgecolors='black', linewidth=0.5)
                
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.3, zorder=1)
                
                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge['length']
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(x_source, y_source, 'y*', markersize=20, 
                           markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        
        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, 'rs', markersize=12, zorder=4)
            else:
                ax.plot(x, y, 'go', markersize=12, zorder=4)
        
        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')







class validation : 
    def __init__(self, graph):
        self.graph = graph
        self.u = None  # Solution du problème direct
        self.w = None  # Sensibilité
        self.p = None  # État adjoint
        
   
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

    def assemble_system_val(self):
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
            g_local = self.source_function_mms(x, edge)

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
    
    def solve_direct_val(self):
        A, g = self.assemble_system_val()
        self.u = spsolve(A, g)
        return self.u