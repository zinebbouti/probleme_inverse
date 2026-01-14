
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import lil_matrix, csr_matrix
# from scipy.sparse.linalg import spsolve

# class SourceLocalization:
#     """Résolution du problème de localisation de source avec méthode adjointe"""
    
#     def __init__(self, graph):
#         self.graph = graph
#         self.u = None
#         self.w = None
#         self.p = None
        
#     def source_function(self, x, epsilon, intensity=1.0, width=0.05):
#         """Fonction source gaussienne centrée en epsilon"""
#         return intensity * np.exp(-((x - epsilon)**2) / (2 * width**2))
    
#     def source_derivative_epsilon(self, x, epsilon, intensity=1.0, width=0.05):
#         """Dérivée de la source par rapport à epsilon: ∂g/∂ε"""
#         gauss = np.exp(-((x - epsilon)**2) / (2 * width**2))
#         return intensity * (x - epsilon) / width**2 * gauss
    
#     def source_derivative_alpha(self, x, epsilon, width=0.05):
#         """Dérivée de la source par rapport à alpha: ∂g/∂α = g/α"""
#         return np.exp(-((x - epsilon)**2) / (2 * width**2))
    
#     def assemble_system(self, epsilon_dict=None, source_intensity=1.0):
#         """Assemble le système linéaire A*u = g"""
#         n = self.graph.n_dof
#         A = lil_matrix((n, n))
#         g = np.zeros(n)

#         for edge in self.graph.edges:
#             edge_id = edge['id']
#             h = edge['h']
#             a = edge['a']
#             n_pts = edge['n']

#             x = np.linspace(h, edge['length'] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)

#             # =========================
#             # CHOIX DE LA SOURCE
#             # =========================
#             if epsilon_dict is not None and edge_id in epsilon_dict:
#                 epsilon = epsilon_dict[edge_id]
#                 g_local = self.source_function(x, epsilon, source_intensity)
#             else:
#                 g_local = np.zeros(n_pts)

#             stiffness = a / h

#             for i, dof in enumerate(dofs):
#                 g[dof] += h * g_local[i]

#                 A[dof, dof] = 2 * stiffness

#                 if i > 0:
#                     A[dof, dofs[i-1]] = -stiffness
#                 else:
#                     v_start_dof = self.graph.get_vertex_dof(edge['v_start'])
#                     if v_start_dof is not None:
#                         A[dof, v_start_dof] = -stiffness

#                 if i + 1 < len(dofs):
#                     A[dof, dofs[i+1]] = -stiffness
#                 else:
#                     v_end_dof = self.graph.get_vertex_dof(edge['v_end'])
#                     if v_end_dof is not None:
#                         A[dof, v_end_dof] = -stiffness
        
#         for v_id in self.graph.vertices:
#             if v_id in self.graph.boundary_vertices:
#                 continue

#             v_dof = self.graph.get_vertex_dof(v_id)
#             incident_edges = self.graph.vertices[v_id]['edges']

#             for edge_id, position in incident_edges:
#                 edge = self.graph.edges[edge_id]
#                 h = edge['h']
#                 a = edge['a']

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 npts = len(dofs)

#                 # Il faut au moins 2 points internes pour ordre 2 au nœud
#                 if npts >= 2:
#                     if position == 'start':
#                         u1 = dofs[0]   # point à h
#                         u2 = dofs[1]   # point à 2h
#                     else:
#                         u1 = dofs[-1]  # point à L-h (adjacent au nœud)
#                         u2 = dofs[-2]  # point à L-2h

#                     # Kirchhoff ordre 2 :
#                     # sum_e a * (-3 u_v + 4 u1 - u2)/(2h) = 0
#                     # -> diag positive :
#                     coeff = a / (2.0 * h)
#                     A[v_dof, v_dof] += 3.0 * coeff
#                     A[v_dof, u1]    += -4.0 * coeff
#                     A[v_dof, u2]    += 1.0 * coeff

#                 else:
#                     # Fallback ordre 1 si pas assez de points
#                     u1 = dofs[0] if position == 'start' else dofs[-1]
#                     coeff = a / h
#                     A[v_dof, v_dof] += coeff
#                     A[v_dof, u1]    += -coeff

#         return A.tocsr(), g
    
#     def solve_direct(self, epsilon_dict=None, source_intensity=1.0):
#         """Résout le problème direct: A*u = g"""
#         A, g = self.assemble_system(epsilon_dict, source_intensity)
#         self.u = spsolve(A, g)
#         return self.u
    
#     def assemble_sensitivity_rhs_alpha(self, epsilon_dict):
#         """Assemble ∂g/∂α pour l'équation de sensibilité"""
#         n = self.graph.n_dof
#         dg_dalpha = np.zeros(n)
        
#         for edge_id, epsilon in epsilon_dict.items():
#             edge = self.graph.edges[edge_id]
#             h = edge['h']
#             n_pts = edge['n']
            
#             x = np.linspace(h, edge['length'] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)
            
#             dg = self.source_derivative_alpha(x, epsilon)
            
#             for i, dof in enumerate(dofs):
#                 dg_dalpha[dof] += h * dg[i]
        
#         return dg_dalpha
    
#     def solve_sensitivity_alpha(self, epsilon_dict, source_intensity=1.0):
#         """Résout l'équation de sensibilité: A*w = ∂g/∂α"""
#         A, _ = self.assemble_system(epsilon_dict, source_intensity)
#         dg_dalpha = self.assemble_sensitivity_rhs_alpha(epsilon_dict)
        
#         self.w = spsolve(A, dg_dalpha)
#         return self.w
    
#     def compute_cost_functional(self, u_data):
#         """Calcule J = 1/2 ∫ (u - u_data)² dx"""
#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")
        
#         J = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge['id'])
#             h = edge['h']
#             diff = self.u[dofs] - u_data[dofs]
#             J += 0.5 * h * np.sum(diff**2)
        
#         return J
    
#     def compute_gradient_finite_diff(self, epsilon_dict, u_data, source_intensity, delta=None):
#         """MÉTHODE 1: Gradient par différences finies centrées"""
#         if delta is None:
#             delta = np.finfo(float).eps**(1/3)
        
#         u_plus = self.solve_direct(epsilon_dict, source_intensity + delta)
#         J_plus = 0.5 * sum(edge['h'] * np.sum((u_plus[self.graph.get_edge_dofs(edge['id'])] - 
#                           u_data[self.graph.get_edge_dofs(edge['id'])])**2) 
#                           for edge in self.graph.edges)
        
#         u_minus = self.solve_direct(epsilon_dict, source_intensity - delta)
#         J_minus = 0.5 * sum(edge['h'] * np.sum((u_minus[self.graph.get_edge_dofs(edge['id'])] - 
#                            u_data[self.graph.get_edge_dofs(edge['id'])])**2) 
#                            for edge in self.graph.edges)
        
#         grad_fd = (J_plus - J_minus) / (2 * delta)
#         self.solve_direct(epsilon_dict, source_intensity)
        
#         return grad_fd
    
#     def compute_gradient_sensitivity(self, epsilon_dict, u_data, source_intensity):
#         """MÉTHODE 2: Gradient via sensibilité directe"""
#         self.solve_direct(epsilon_dict, source_intensity)
#         w = self.solve_sensitivity_alpha(epsilon_dict, source_intensity)
        
#         grad_sens = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge['id'])
#             h = edge['h']
            
#             diff = self.u[dofs] - u_data[dofs]
#             grad_sens += h * np.dot(diff, w[dofs])
        
#         return grad_sens
    
#     def assemble_adjoint_rhs(self, u_data):
#         """Assemble le second membre pour l'équation adjointe"""
#         n = self.graph.n_dof
#         rhs = np.zeros(n)
        
#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")
        
#         for edge in self.graph.edges:
#             edge_id = edge['id']
#             h = edge['h']
#             dofs = self.graph.get_edge_dofs(edge_id)
            
#             rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])
        
#         return rhs
    
#     def solve_adjoint(self, epsilon_dict, u_data, source_intensity):
#         """Résout l'équation adjointe: A^T * p = -∂J/∂u"""
#         A, _ = self.assemble_system(epsilon_dict, source_intensity)
#         rhs_adjoint = self.assemble_adjoint_rhs(u_data)
        
#         self.p = spsolve(A.T, rhs_adjoint)
#         return self.p
    
#     def compute_gradient_adjoint(self, epsilon_dict, source_intensity):
#         """MÉTHODE 3: Gradient via méthode adjointe"""
#         if self.p is None:
#             raise ValueError("Résoudre d'abord l'équation adjointe")
        
#         dg_dalpha = self.assemble_sensitivity_rhs_alpha(epsilon_dict)
#         grad_adj = -np.dot(self.p, dg_dalpha)
        
#         return grad_adj
    
#     def validate_gradient_three_methods(self, epsilon_dict, u_data, source_intensity, delta=None):
#         """VALIDATION COMPLÈTE: Compare les 3 méthodes pour calculer dJ/dα"""
#         if delta is None:
#             delta = np.finfo(float).eps**(1/3)
        
#         print(f"\n{'='*80}")
#         print(f"VALIDATION GRADIENT dJ/dα - COMPARAISON DES 3 MÉTHODES")
#         print(f"{'='*80}")
#         print(f"Delta optimal (différences finies) = {delta:.3e}")
#         print(f"Intensité source α = {source_intensity}")
#         print(f"\nOBJECTIF: Erreur relative < 10⁻¹⁵ (précision machine)\n")
        
#         print("Calcul méthode 1: Différences finies...")
#         grad_fd = self.compute_gradient_finite_diff(epsilon_dict, u_data, source_intensity, delta)
        
#         print("Calcul méthode 2: Sensibilité directe...")
#         grad_sens = self.compute_gradient_sensitivity(epsilon_dict, u_data, source_intensity)
        
#         print("Calcul méthode 3: Méthode adjointe...")
#         self.solve_adjoint(epsilon_dict, u_data, source_intensity)
#         grad_adj = self.compute_gradient_adjoint(epsilon_dict, source_intensity)
        
#         print(f"\n{'-'*80}")
#         print(f"{'MÉTHODE':<30} {'VALEUR dJ/dα':<20} {'ERREUR vs FD':<15} ")
#         print(f"{'-'*80}")
        
#         print(f"{'1. Différences finies':<30} {grad_fd:<20.12e} {'---':<15} ")
        
#         err_sens = abs(grad_sens - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_sens - grad_fd)
#         status_sens = "✓✓ EXCELLENT" if err_sens < 1e-14 else ("✓ BON" if err_sens < 1e-10 else "⚠ AMÉLIORER")
#         print(f"{'2. Sensibilité directe':<30} {grad_sens:<20.12e} {err_sens:<15.3e} ")
        
#         err_adj = abs(grad_adj - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_adj - grad_fd)
#         status_adj = "✓✓ EXCELLENT" if err_adj < 1e-14 else ("✓ BON" if err_adj < 1e-10 else "⚠ AMÉLIORER")
#         print(f"{'3. Méthode adjointe':<30} {grad_adj:<20.12e} {err_adj:<15.3e} ")
        
#         err_sens_adj = abs(grad_sens - grad_adj) / abs(grad_adj) if abs(grad_adj) > 1e-16 else abs(grad_sens - grad_adj)
#         print(f"\n{'-'*80}")
#         print(f"Comparaison directe Sensibilité vs Adjointe:")
#         print(f"  Erreur relative: {err_sens_adj:.3e}")
#         status_final = "✓✓ EXCELLENT" if err_sens_adj < 1e-14 else ("✓ BON" if err_sens_adj < 1e-10 else "⚠ AMÉLIORER")
#         print(f"  Status: {status_final}")
        
#         print(f"\n{'='*80}")
#         print(f"RÉSUMÉ:")
#         print(f"{'='*80}")
#         max_err = max(err_sens, err_adj)
#         print(f"Erreur maximale (vs différences finies): {max_err:.3e}")
        
#         if max_err < 1e-14 and err_sens_adj < 1e-14:
#             print(f"\n{'✓'*40}")
#             print(f"✓✓ VALIDATION RÉUSSIE!")
#             print(f"✓✓ Les 3 méthodes concordent à la précision machine (< 10⁻¹⁴)")
#             print(f"{'✓'*40}")
#             validation_passed = True
#         else:
#             validation_passed = False
        
#         return {
#             'grad_fd': grad_fd,
#             'grad_sensitivity': grad_sens,
#             'grad_adjoint': grad_adj,
#             'error_sens_vs_fd': err_sens,
#             'error_adj_vs_fd': err_adj,
#             'error_sens_vs_adj': err_sens_adj,
#             'max_error': max_err,
#             'validation_passed': validation_passed,
#             'delta': delta
#         }
    
#     # ========================================================================
#     # VISUALISATIONS
#     # ========================================================================
    
#     def plot_solution_on_graph(self, epsilon_dict=None, title="Solution u sur le graphe"):
#         """Visualise la solution u sur le graphe"""
#         if self.u is None:
#             print("Aucune solution à afficher.")
#             return
        
#         fig, ax = plt.subplots(figsize=(12, 10))
        
#         for edge in self.graph.edges:
#             v_start = edge['v_start']
#             v_end = edge['v_end']
#             edge_id = edge['id']
            
#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]
                
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 u_edge = self.u[dofs]
                
#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)
                
#                 scatter = ax.scatter(x_interp, y_interp, c=u_edge, cmap='coolwarm', 
#                                    s=100, vmin=self.u.min(), vmax=self.u.max(), zorder=2,
#                                    edgecolors='black', linewidth=0.5)
                
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge['length']
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(x_source, y_source, 'y*', markersize=30, 
#                            markeredgecolor='black', markeredgewidth=2, zorder=3,
#                            label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, 'rs', markersize=16, zorder=4, 
#                        label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
#             else:
#                 ax.plot(x, y, 'go', markersize=16, zorder=4,
#                        label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
#         cbar = plt.colorbar(scatter, ax=ax, label='Valeur de u')
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel('x', fontsize=13)
#         ax.set_ylabel('y', fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')
#         ax.legend(fontsize=11, loc='best')
#         plt.tight_layout()
#         plt.show()
    
#     def plot_sensitivity_on_graph(self, epsilon_dict=None, title="Sensibilité w = ∂u/∂α"):
#         """Visualise la sensibilité w sur le graphe"""
#         if self.w is None:
#             print("Aucune sensibilité à afficher.")
#             return
        
#         fig, ax = plt.subplots(figsize=(12, 10))
        
#         for edge in self.graph.edges:
#             v_start = edge['v_start']
#             v_end = edge['v_end']
#             edge_id = edge['id']
            
#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]
                
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 w_edge = self.w[dofs]
                
#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)
                
#                 scatter = ax.scatter(x_interp, y_interp, c=w_edge, cmap='viridis', 
#                                    s=100, vmin=self.w.min(), vmax=self.w.max(), zorder=2,
#                                    edgecolors='black', linewidth=0.5)
                
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge['length']
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(x_source, y_source, 'y*', markersize=30, 
#                            markeredgecolor='black', markeredgewidth=2, zorder=3,
#                            label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, 'rs', markersize=16, zorder=4, 
#                        label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
#             else:
#                 ax.plot(x, y, 'go', markersize=16, zorder=4,
#                        label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
#         cbar = plt.colorbar(scatter, ax=ax, label='Valeur de w')
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel('x', fontsize=13)
#         ax.set_ylabel('y', fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')
#         ax.legend(fontsize=11, loc='best')
#         plt.tight_layout()
#         plt.show()
    
#     def plot_adjoint_on_graph(self, epsilon_dict=None, title="État adjoint p"):
#         """Visualise l'état adjoint p sur le graphe"""
#         if self.p is None:
#             print("Aucun état adjoint à afficher.")
#             return
        
#         fig, ax = plt.subplots(figsize=(12, 10))
        
#         for edge in self.graph.edges:
#             v_start = edge['v_start']
#             v_end = edge['v_end']
#             edge_id = edge['id']
            
#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]
                
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 p_edge = self.p[dofs]
                
#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)
                
#                 scatter = ax.scatter(x_interp, y_interp, c=p_edge, cmap='plasma', 
#                                    s=100, vmin=self.p.min(), vmax=self.p.max(), zorder=2,
#                                    edgecolors='black', linewidth=0.5)
                
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge['length']
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(x_source, y_source, 'y*', markersize=30, 
#                            markeredgecolor='black', markeredgewidth=2, zorder=3,
#                            label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, 'rs', markersize=16, zorder=4, 
#                        label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
#             else:
#                 ax.plot(x, y, 'go', markersize=16, zorder=4,
#                        label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
#         cbar = plt.colorbar(scatter, ax=ax, label='Valeur de p')
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel('x', fontsize=13)
#         ax.set_ylabel('y', fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')
#         ax.legend(fontsize=11, loc='best')
#         plt.tight_layout()
#         plt.show()
    
#     def plot_all_results(self, epsilon_dict, u_data):
#         """Affiche tous les résultats dans une grille 2x2"""
#         fig = plt.figure(figsize=(16, 14))
        
#         # Subplot 1: Solution u
#         ax1 = plt.subplot(2, 2, 1)
#         self._plot_on_axis(ax1, self.u, epsilon_dict, "Solution u", 'coolwarm')
        
#         # Subplot 2: Sensibilité w
#         ax2 = plt.subplot(2, 2, 2)
#         if self.w is not None:
#             self._plot_on_axis(ax2, self.w, epsilon_dict, "Sensibilité w = ∂u/∂α", 'viridis')
#         else:
#             ax2.text(0.5, 0.5, "Sensibilité non calculée", ha='center', va='center', fontsize=14)
#             ax2.set_xlim(0, 1)
#             ax2.set_ylim(0, 1)
        
#         # Subplot 3: État adjoint p
#         ax3 = plt.subplot(2, 2, 3)
#         if self.p is not None:
#             self._plot_on_axis(ax3, self.p, epsilon_dict, "État adjoint p", 'plasma')
#         else:
#             ax3.text(0.5, 0.5, "État adjoint non calculé", ha='center', va='center', fontsize=14)
#             ax3.set_xlim(0, 1)
#             ax3.set_ylim(0, 1)
        
#         # Subplot 4: Données observées
#         ax4 = plt.subplot(2, 2, 4)
#         self._plot_on_axis(ax4, u_data, epsilon_dict, "Données observées u_data", 'coolwarm')
        
#         plt.tight_layout()
#         plt.show()
    
#     def _plot_on_axis(self, ax, data, epsilon_dict, title, cmap):
#         """Fonction auxiliaire pour tracer sur un axe donné"""
#         if data is None:
#             return
        
#         for edge in self.graph.edges:
#             v_start = edge['v_start']
#             v_end = edge['v_end']
#             edge_id = edge['id']
            
#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]
                
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 data_edge = data[dofs]
                
#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)
                
#                 scatter = ax.scatter(x_interp, y_interp, c=data_edge, cmap=cmap, 
#                                    s=80, vmin=data.min(), vmax=data.max(), zorder=2,
#                                    edgecolors='black', linewidth=0.5)
                
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.3, zorder=1)
                
#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge['length']
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(x_source, y_source, 'y*', markersize=20, 
#                            markeredgecolor='black', markeredgewidth=1.5, zorder=3)
        
#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, 'rs', markersize=12, zorder=4)
#             else:
#                 ax.plot(x, y, 'go', markersize=12, zorder=4)
        
#         plt.colorbar(scatter, ax=ax)
#         ax.set_xlabel('x', fontsize=11)
#         ax.set_ylabel('y', fontsize=11)
#         ax.set_title(title, fontsize=12, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')

#    # ============================================================
#     # COÛT + GRADIENT (α)
#     # ============================================================
#     def cost_and_gradient_alpha(self, epsilon_dict, u_data, alpha):
#         self.solve_direct(epsilon_dict, alpha)
#         J = self.compute_cost_functional(u_data)
#         self.solve_adjoint(epsilon_dict, u_data, alpha)
#         grad = self.compute_gradient_adjoint(epsilon_dict, alpha)
#         return J, grad


#     # ============================================================
#     # LINE SEARCH (Armijo)
#     # ============================================================
#     def line_search_alpha(
#         self,
#         epsilon_dict,
#         u_data,
#         alpha,
#         d,
#         J0,
#         g0,
#         c=1e-4,
#         step0=1.0
#     ):
#         slope0 = g0 * d
#         if slope0 >= 0:
#             d = -g0
#             slope0 = g0 * d

#         step = step0
#         for _ in range(20):
#             alpha_try = max(alpha + step * d, 0.0)
#             if alpha_try == alpha:
#                 return alpha

#             self.solve_direct(epsilon_dict, alpha_try)
#             J_try = self.compute_cost_functional(u_data)

#             if J_try <= J0 + c * step * slope0:
#                 return alpha_try

#             step *= 0.5

#         return alpha


#     # ============================================================
#     # GRADIENT CONJUGUÉ NON LINÉAIRE
#     # ============================================================
#     def conjugate_gradient_alpha(
#         self,
#         epsilon_dict,
#         u_data,
#         alpha_init,
#         max_iter=50,
#         tol=1e-8,
#         verbose=True
#     ):
#         alpha = float(alpha_init)
#         J, g = self.cost_and_gradient_alpha(epsilon_dict, u_data, alpha)
#         d = -g

#         if verbose:
#             print("\n" + "="*70)
#             print("INVERSION PAR GRADIENT CONJUGUÉ (α)")
#             print("="*70)
#             print(f"Init  | alpha = {alpha:.6f} | J = {J:.3e}")

#         for k in range(max_iter):
#             if abs(g) < tol:
#                 print("✓ Convergence atteinte")
#                 break

#             alpha_new = self.line_search_alpha(
#                 epsilon_dict, u_data, alpha, d, J, g
#             )

#             J_new, g_new = self.cost_and_gradient_alpha(
#                 epsilon_dict, u_data, alpha_new
#             )

#             if verbose:
#                 print(
#                     f"Iter {k+1:02d} | alpha = {alpha_new:.6f} | "
#                     f"J = {J_new:.3e} | |grad| = {abs(g_new):.3e}"
#                 )

#             beta = (g_new * g_new) / (g * g) if abs(g) > 1e-30 else 0.0
#             d = -g_new + beta * d

#             if g_new * d >= 0:
#                 d = -g_new

#             alpha, g, J = alpha_new, g_new, J_new

#         return alpha

############################################################


############################################################


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from scipy.sparse import lil_matrix, csr_matrix
# from scipy.sparse.linalg import spsolve

# class SourceLocalizationEpsilon:
#     """Localisation de source : optimisation de ε avec α=1 fixé"""
    
#     def __init__(self, graph):
#         self.graph = graph
#         self.u = None
#         self.w = None  # Sensibilité par rapport à ε
#         self.p = None  # État adjoint
#         self.alpha_fixed = 1.0  # Intensité fixée
        
#         # Historique pour l'optimisation
#         self.history = {
#             'epsilon': [],
#             'J': [],
#             'grad_norm': [],
#             'method': None
#         }
    
#     def source_function(self, x, epsilon, width=0.05):
#         """Fonction source gaussienne : g(x,ε) = α * exp(-((x-ε)²)/(2σ²))"""
#         return self.alpha_fixed * np.exp(-((x - epsilon)**2) / (2 * width**2))
    
#     def source_derivative_epsilon(self, x, epsilon, width=0.05):
#         """Dérivée de la source par rapport à ε: ∂g/∂ε"""
#         gauss = np.exp(-((x - epsilon)**2) / (2 * width**2))
#         return self.alpha_fixed * (x - epsilon) / width**2 * gauss
    
#     # ========================================================================
#     # ASSEMBLAGE DU SYSTÈME
#     # ========================================================================
    
#     def assemble_system(self, epsilon_dict):
#         """Assemble A*u = g(ε) pour une position de source donnée"""
#         n = self.graph.n_dof
#         A = lil_matrix((n, n))
#         g = np.zeros(n)

#         for edge in self.graph.edges:
#             edge_id = edge['id']
#             h = edge['h']
#             a = edge['a']
#             n_pts = edge['n']

#             x = np.linspace(h, edge['length'] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)

#             # Source sur l'arête si epsilon_dict le spécifie
#             if epsilon_dict is not None and edge_id in epsilon_dict:
#                 epsilon = epsilon_dict[edge_id]
#                 g_local = self.source_function(x, epsilon)
#             else:
#                 g_local = np.zeros(n_pts)

#             stiffness = a / h

#             for i, dof in enumerate(dofs):
#                 g[dof] += h * g_local[i]
#                 A[dof, dof] = 2 * stiffness

#                 if i > 0:
#                     A[dof, dofs[i-1]] = -stiffness
#                 else:
#                     v_start_dof = self.graph.get_vertex_dof(edge['v_start'])
#                     if v_start_dof is not None:
#                         A[dof, v_start_dof] = -stiffness

#                 if i + 1 < len(dofs):
#                     A[dof, dofs[i+1]] = -stiffness
#                 else:
#                     v_end_dof = self.graph.get_vertex_dof(edge['v_end'])
#                     if v_end_dof is not None:
#                         A[dof, v_end_dof] = -stiffness
        
#         # Conditions de Kirchhoff aux nœuds internes (ordre 2)
#         for v_id in self.graph.vertices:
#             if v_id in self.graph.boundary_vertices:
#                 continue

#             v_dof = self.graph.get_vertex_dof(v_id)
#             incident_edges = self.graph.vertices[v_id]['edges']

#             for edge_id, position in incident_edges:
#                 edge = self.graph.edges[edge_id]
#                 h = edge['h']
#                 a = edge['a']
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 npts = len(dofs)

#                 if npts >= 2:
#                     if position == 'start':
#                         u1, u2 = dofs[0], dofs[1]
#                     else:
#                         u1, u2 = dofs[-1], dofs[-2]

#                     coeff = a / (2.0 * h)
#                     A[v_dof, v_dof] += 3.0 * coeff
#                     A[v_dof, u1]    += -4.0 * coeff
#                     A[v_dof, u2]    += 1.0 * coeff
#                 else:
#                     u1 = dofs[0] if position == 'start' else dofs[-1]
#                     coeff = a / h
#                     A[v_dof, v_dof] += coeff
#                     A[v_dof, u1]    += -coeff

#         return A.tocsr(), g
    
#     # ========================================================================
#     # PROBLÈME DIRECT
#     # ========================================================================
    
#     def solve_direct(self, epsilon_dict):
#         """Résout le problème direct: A*u = g(ε)"""
#         A, g = self.assemble_system(epsilon_dict)
#         self.u = spsolve(A, g)
#         return self.u
    
#     # ========================================================================
#     # FONCTIONNELLE DE COÛT
#     # ========================================================================
    
#     def compute_cost_functional(self, u_data):
#         """Calcule J(ε) = 1/2 ∫ (u(ε) - u_data)² dx"""
#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")
        
#         J = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge['id'])
#             h = edge['h']
#             diff = self.u[dofs] - u_data[dofs]
#             J += 0.5 * h * np.sum(diff**2)
        
#         return J
    
#     # ========================================================================
#     # MÉTHODE 1 : DIFFÉRENCES FINIES
#     # ========================================================================
    
#     def compute_gradient_finite_diff(self, epsilon_dict, u_data, edge_id_source, delta=None):
#         """Gradient dJ/dε par différences finies centrées"""
#         if delta is None:
#             delta = np.finfo(float).eps**(1/3)  # δ optimal ≈ 6e-6
        
#         epsilon_current = epsilon_dict[edge_id_source]
        
#         # J(ε + δ)
#         epsilon_plus = epsilon_dict.copy()
#         epsilon_plus[edge_id_source] = epsilon_current + delta
#         u_plus = self.solve_direct(epsilon_plus)
#         J_plus = 0.5 * sum(edge['h'] * np.sum((u_plus[self.graph.get_edge_dofs(edge['id'])] -
#                           u_data[self.graph.get_edge_dofs(edge['id'])])**2)
#                           for edge in self.graph.edges)
        
#         # J(ε - δ)
#         epsilon_minus = epsilon_dict.copy()
#         epsilon_minus[edge_id_source] = epsilon_current - delta
#         u_minus = self.solve_direct(epsilon_minus)
#         J_minus = 0.5 * sum(edge['h'] * np.sum((u_minus[self.graph.get_edge_dofs(edge['id'])] -
#                            u_data[self.graph.get_edge_dofs(edge['id'])])**2)
#                            for edge in self.graph.edges)
        
#         grad_fd = (J_plus - J_minus) / (2 * delta)
        
#         # Restaurer la solution à ε courant
#         self.solve_direct(epsilon_dict)
        
#         return grad_fd
    
#     # ========================================================================
#     # MÉTHODE 2 : SENSIBILITÉ DIRECTE
#     # ========================================================================
    
#     def assemble_sensitivity_rhs_epsilon(self, epsilon_dict):
#         """Assemble ∂g/∂ε pour l'équation de sensibilité"""
#         n = self.graph.n_dof
#         dg_deps = np.zeros(n)
        
#         for edge_id, epsilon in epsilon_dict.items():
#             edge = self.graph.edges[edge_id]
#             h = edge['h']
#             n_pts = edge['n']
            
#             x = np.linspace(h, edge['length'] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)
            
#             dg = self.source_derivative_epsilon(x, epsilon)
            
#             for i, dof in enumerate(dofs):
#                 dg_deps[dof] += h * dg[i]
        
#         return dg_deps
    
#     def solve_sensitivity_epsilon(self, epsilon_dict):
#         """Résout l'équation de sensibilité: A*w = ∂g/∂ε"""
#         A, _ = self.assemble_system(epsilon_dict)
#         dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict)
        
#         self.w = spsolve(A, dg_deps)
#         return self.w
    
#     def compute_gradient_sensitivity(self, epsilon_dict, u_data):
#         """Gradient dJ/dε via sensibilité directe"""
#         self.solve_direct(epsilon_dict)
#         w = self.solve_sensitivity_epsilon(epsilon_dict)
        
#         grad_sens = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge['id'])
#             h = edge['h']
            
#             diff = self.u[dofs] - u_data[dofs]
#             grad_sens += h * np.dot(diff, w[dofs])
        
#         return grad_sens
    
#     # ========================================================================
#     # MÉTHODE 3 : MÉTHODE ADJOINTE
#     # ========================================================================
    
#     def assemble_adjoint_rhs(self, u_data):
#         """Assemble le second membre pour l'équation adjointe"""
#         n = self.graph.n_dof
#         rhs = np.zeros(n)
        
#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")
        
#         for edge in self.graph.edges:
#             edge_id = edge['id']
#             h = edge['h']
#             dofs = self.graph.get_edge_dofs(edge_id)
            
#             rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])
        
#         return rhs
    
#     def solve_adjoint(self, epsilon_dict, u_data):
#         """Résout l'équation adjointe: A^T * p = -∂J/∂u"""
#         A, _ = self.assemble_system(epsilon_dict)
#         rhs_adjoint = self.assemble_adjoint_rhs(u_data)
        
#         self.p = spsolve(A.T, rhs_adjoint)
#         return self.p
    
#     def compute_gradient_adjoint(self, epsilon_dict):
#         """Gradient dJ/dε via méthode adjointe"""
#         if self.p is None:
#             raise ValueError("Résoudre d'abord l'équation adjointe")
        
#         dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict)
#         grad_adj = -np.dot(self.p, dg_deps)
        
#         return grad_adj
    
#     # ========================================================================
#     # VALIDATION DES GRADIENTS
#     # ========================================================================
    
#     def validate_gradient_three_methods(self, epsilon_dict, u_data, edge_id_source, delta=None):
#         """Compare les 3 méthodes pour calculer dJ/dε"""
#         if delta is None:
#             delta = np.finfo(float).eps**(1/3)
        
#         print(f"\n{'='*80}")
#         print(f"VALIDATION GRADIENT dJ/dε - COMPARAISON DES 3 MÉTHODES")
#         print(f"{'='*80}")
#         print(f"Position source ε = {epsilon_dict[edge_id_source]:.4f}")
#         print(f"Intensité α = {self.alpha_fixed} (FIXÉE)")
#         print(f"Delta (différences finies) = {delta:.3e}\n")
        
#         # Calcul du coût initial
#         self.solve_direct(epsilon_dict)
#         J_current = self.compute_cost_functional(u_data)
#         print(f"Coût J(ε) = {J_current:.6e}\n")
        
#         print("Calcul méthode 1: Différences finies...")
#         grad_fd = self.compute_gradient_finite_diff(epsilon_dict, u_data, edge_id_source, delta)
        
#         print("Calcul méthode 2: Sensibilité directe...")
#         grad_sens = self.compute_gradient_sensitivity(epsilon_dict, u_data)
        
#         print("Calcul méthode 3: Méthode adjointe...")
#         self.solve_adjoint(epsilon_dict, u_data)
#         grad_adj = self.compute_gradient_adjoint(epsilon_dict)
        
#         print(f"\n{'-'*80}")
#         print(f"{'MÉTHODE':<30} {'VALEUR dJ/dε':<20} {'ERREUR vs FD':<15}")
#         print(f"{'-'*80}")
        
#         print(f"{'1. Différences finies':<30} {grad_fd:<20.12e} {'---':<15}")
        
#         err_sens = abs(grad_sens - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_sens - grad_fd)
#         print(f"{'2. Sensibilité directe':<30} {grad_sens:<20.12e} {err_sens:<15.3e}")
        
#         err_adj = abs(grad_adj - grad_fd) / abs(grad_fd) if abs(grad_fd) > 1e-16 else abs(grad_adj - grad_fd)
#         print(f"{'3. Méthode adjointe':<30} {grad_adj:<20.12e} {err_adj:<15.3e}")
        
#         err_sens_adj = abs(grad_sens - grad_adj) / abs(grad_adj) if abs(grad_adj) > 1e-16 else abs(grad_sens - grad_adj)
#         print(f"\n{'-'*80}")
#         print(f"Comparaison directe Sensibilité vs Adjointe:")
#         print(f"  Erreur relative: {err_sens_adj:.3e}")
        
#         # CORRECTION: max() au lieu de maerr_sens, err_adj)
#         max_err = max(err_sens, err_adj)
#         validation_passed = max_err < 1e-10 and err_sens_adj < 1e-14
        
#         print(f"\n{'='*80}")
#         if validation_passed:
#             print(f"✓✓ VALIDATION RÉUSSIE!")
#         else:
#             print(f"⚠ Validation partielle (erreur max: {max_err:.3e})")
#         print(f"{'='*80}\n")
        
#         return {
#             'J': J_current,
#             'grad_fd': grad_fd,
#             'grad_sensitivity': grad_sens,
#             'grad_adjoint': grad_adj,
#             'error_sens_vs_fd': err_sens,
#             'error_adj_vs_fd': err_adj,
#             'error_sens_vs_adj': err_sens_adj,
#             'validation_passed': validation_passed
#         }
    
#     # ========================================================================
#     # OPTIMISATION PAR GRADIENT CONJUGUÉ
#     # ========================================================================
    
#     def optimize_cg_adjoint(self, epsilon_init, edge_id_source, u_data,
#                             max_iter=50, tol=1e-6, bounds=None):
#         """Optimisation de ε par gradient conjugué (méthode adjointe)"""
#         print(f"\n{'='*80}")
#         print(f"OPTIMISATION PAR GRADIENT CONJUGUÉ (Méthode Adjointe)")
#         print(f"{'='*80}\n")
        
#         self.history = {'epsilon': [], 'J': [], 'grad_norm': [], 'method': 'CG-Adjoint'}
        
#         def objective_and_gradient(eps):
#             """Calcule J(ε) et dJ/dε"""
#             epsilon_dict = {edge_id_source: eps}
            
#             # Problème direct
#             self.solve_direct(epsilon_dict)
#             J = self.compute_cost_functional(u_data)
            
#             # Problème adjoint
#             self.solve_adjoint(epsilon_dict, u_data)
#             grad = self.compute_gradient_adjoint(epsilon_dict)
            
#             # Historique
#             self.history['epsilon'].append(eps)
#             self.history['J'].append(J)
#             self.history['grad_norm'].append(abs(grad))
            
#             print(f"Iter {len(self.history['J'])-1:3d} | ε = {eps:8.5f} | J = {J:.6e} | |∇J| = {abs(grad):.3e}")
            
#             return J, grad
        
#         # Optimisation avec scipy
#         if bounds is None:
#             edge = self.graph.edges[edge_id_source]
#             bounds = [(0.1, edge['length'] - 0.1)]  # Garder ε dans l'arête
        
#         result = minimize(
#             fun=lambda eps: objective_and_gradient(eps)[0],
#             x0=epsilon_init,
#             jac=lambda eps: objective_and_gradient(eps)[1],
#             method='L-BFGS-B',  # Quasi-Newton avec bornes
#             bounds=bounds,
#             options={'maxiter': max_iter, 'ftol': tol}
#         )
        
#         print(f"\n{'-'*80}")
#         print(f"RÉSULTAT OPTIMISATION:")
#         print(f"  Position optimale ε* = {result.x[0]:.6f}")
#         print(f"  Coût minimal J(ε*) = {result.fun:.6e}")
#         print(f"  Nombre d'itérations: {result.nit}")
#         print(f"  Convergence: {'OUI' if result.success else 'NON'}")
#         print(f"{'-'*80}\n")
        
#         return result
    
#     def optimize_cg_sensitivity(self, epsilon_init, edge_id_source, u_data,
#                                 max_iter=50, tol=1e-6, bounds=None):
#         """Optimisation de ε par gradient conjugué (méthode sensibilité)"""
#         print(f"\n{'='*80}")
#         print(f"OPTIMISATION PAR GRADIENT CONJUGUÉ (Méthode Sensibilité)")
#         print(f"{'='*80}\n")
        
#         self.history = {'epsilon': [], 'J': [], 'grad_norm': [], 'method': 'CG-Sensitivity'}
        
#         def objective_and_gradient(eps):
#             """Calcule J(ε) et dJ/dε"""
#             epsilon_dict = {edge_id_source: eps}
            
#             # Gradient par sensibilité
#             grad = self.compute_gradient_sensitivity(epsilon_dict, u_data)
#             J = self.compute_cost_functional(u_data)
            
#             # Historique
#             self.history['epsilon'].append(eps)
#             self.history['J'].append(J)
#             self.history['grad_norm'].append(abs(grad))
            
#             print(f"Iter {len(self.history['J'])-1:3d} | ε = {eps:8.5f} | J = {J:.6e} | |∇J| = {abs(grad):.3e}")
            
#             return J, grad
        
#         if bounds is None:
#             edge = self.graph.edges[edge_id_source]
#             bounds = [(0.1, edge['length'] - 0.1)]
        
#         result = minimize(
#             fun=lambda eps: objective_and_gradient(eps)[0],
#             x0=epsilon_init,
#             jac=lambda eps: objective_and_gradient(eps)[1],
#             method='L-BFGS-B',
#             bounds=bounds,
#             options={'maxiter': max_iter, 'ftol': tol}
#         )
        
#         print(f"\n{'-'*80}")
#         print(f"RÉSULTAT OPTIMISATION:")
#         print(f"  Position optimale ε* = {result.x[0]:.6f}")
#         print(f"  Coût minimal J(ε*) = {result.fun:.6e}")
#         print(f"  Nombre d'itérations: {result.nit}")
#         print(f"  Convergence: {'OUI' if result.success else 'NON'}")
#         print(f"{'-'*80}\n")
        
#         return result
    
#     # ========================================================================
#     # VISUALISATION
#     # ========================================================================
    
#     def plot_optimization_history(self):
#         """Affiche l'historique de convergence"""
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
#         # Position ε
#         axes[0].plot(self.history['epsilon'], 'o-', linewidth=2)
#         axes[0].set_xlabel('Itération', fontsize=11)
#         axes[0].set_ylabel('Position ε', fontsize=11)
#         axes[0].set_title('Convergence de ε', fontweight='bold')
#         axes[0].grid(True, alpha=0.3)
        
#         # Coût J
#         axes[1].semilogy(self.history['J'], 'o-', linewidth=2, color='red')
#         axes[1].set_xlabel('Itération', fontsize=11)
#         axes[1].set_ylabel('J(ε)', fontsize=11)
#         axes[1].set_title('Décroissance du coût', fontweight='bold')
#         axes[1].grid(True, alpha=0.3)
        
#         # Norme du gradient
#         axes[2].semilogy(self.history['grad_norm'], 'o-', linewidth=2, color='green')
#         axes[2].set_xlabel('Itération', fontsize=11)
#         axes[2].set_ylabel('|∇J|', fontsize=11)
#         axes[2].set_title('Norme du gradient', fontweight='bold')
#         axes[2].grid(True, alpha=0.3)
        
#         plt.suptitle(f'Optimisation par {self.history["method"]}',
#                      fontsize=14, fontweight='bold')
#         plt.tight_layout()
#         plt.show()

#     def plot_solution_on_graph(self, epsilon_dict, title="Solution u sur le graphe"):
#         """Visualise la solution u sur le graphe"""
#         if self.u is None:
#             print("Aucune solution à afficher. Résolvez d'abord le problème direct.")
#             return
        
#         fig, ax = plt.subplots(figsize=(12, 10))
        
#         for edge in self.graph.edges:
#             v_start = edge['v_start']
#             v_end = edge['v_end']
#             edge_id = edge['id']
            
#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]
                
#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 u_edge = self.u[dofs]
                
#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)
                
#                 scatter = ax.scatter(x_interp, y_interp, c=u_edge, cmap='coolwarm', 
#                                    s=100, vmin=self.u.min(), vmax=self.u.max(), zorder=2,
#                                    edgecolors='black', linewidth=0.5)
                
#                 ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.3, zorder=1)
                
#                 # Marquer la position de la source
#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge['length']
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(x_source, y_source, 'y*', markersize=30, 
#                            markeredgecolor='black', markeredgewidth=2, zorder=3,
#                            label='Source' if edge_id == list(epsilon_dict.keys())[0] else '')
        
#         # Tracer les nœuds
#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, 'rs', markersize=16, zorder=4, 
#                        label='Bord' if v_id == list(self.graph.boundary_vertices)[0] else '')
#             else:
#                 ax.plot(x, y, 'go', markersize=16, zorder=4,
#                        label='Interne' if v_id == list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)[0] else '')
        
#         cbar = plt.colorbar(scatter, ax=ax, label='Valeur de u')
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel('x', fontsize=13)
#         ax.set_ylabel('y', fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.axis('equal')
#         ax.legend(fontsize=11, loc='best')
#         plt.tight_layout()
#         plt.show()



##############################
##############################
##############################
##############################





import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class SourceLocalization:
    """Résolution du problème de localisation de source avec méthode adjointe"""

    def __init__(self, graph):
        self.graph = graph
        self.u = None
        self.w = None
        self.p = None

    # ========================================================================
    # SOURCE
    # ========================================================================

    def source_function(self, x, epsilon, intensity=1.0, width=0.05):
        """Fonction source gaussienne centrée en epsilon"""
        return intensity * np.exp(-((x - epsilon) ** 2) / (2 * width**2))

    def source_derivative_epsilon(self, x, epsilon, intensity=1.0, width=0.05):
        """Dérivée de la source par rapport à epsilon: ∂g/∂ε"""
        gauss = np.exp(-((x - epsilon) ** 2) / (2 * width**2))
        return intensity * (x - epsilon) / width**2 * gauss

    # ========================================================================
    # ASSEMBLAGE / DIRECT
    # ========================================================================

    def assemble_system(self, epsilon_dict=None, source_intensity=1.0, width=0.05):
        """Assemble le système linéaire A*u = g"""
        n = self.graph.n_dof
        A = lil_matrix((n, n))
        g = np.zeros(n)

        for edge in self.graph.edges:
            edge_id = edge["id"]
            h = edge["h"]
            a = edge["a"]
            n_pts = edge["n"]

            x = np.linspace(h, edge["length"] - h, n_pts)
            dofs = self.graph.get_edge_dofs(edge_id)

            # =========================
            # CHOIX DE LA SOURCE
            # =========================
            if epsilon_dict is not None and edge_id in epsilon_dict:
                epsilon = epsilon_dict[edge_id]
                g_local = self.source_function(x, epsilon, source_intensity, width=width)
            else:
                g_local = np.zeros(n_pts)

            stiffness = a / h

            for i, dof in enumerate(dofs):
                g[dof] += h * g_local[i]

                A[dof, dof] = 2 * stiffness

                if i > 0:
                    A[dof, dofs[i - 1]] = -stiffness
                else:
                    v_start_dof = self.graph.get_vertex_dof(edge["v_start"])
                    if v_start_dof is not None:
                        A[dof, v_start_dof] = -stiffness

                if i + 1 < len(dofs):
                    A[dof, dofs[i + 1]] = -stiffness
                else:
                    v_end_dof = self.graph.get_vertex_dof(edge["v_end"])
                    if v_end_dof is not None:
                        A[dof, v_end_dof] = -stiffness

        # Conditions de Kirchhoff aux nœuds internes (ordre 2 si possible)
        for v_id in self.graph.vertices:
            if v_id in self.graph.boundary_vertices:
                continue

            v_dof = self.graph.get_vertex_dof(v_id)
            incident_edges = self.graph.vertices[v_id]["edges"]

            for edge_id, position in incident_edges:
                edge = self.graph.edges[edge_id]
                h = edge["h"]
                a = edge["a"]

                dofs = self.graph.get_edge_dofs(edge_id)
                npts = len(dofs)

                if npts >= 2:
                    if position == "start":
                        u1 = dofs[0]
                        u2 = dofs[1]
                    else:
                        u1 = dofs[-1]
                        u2 = dofs[-2]

                    # sum_e a * (-3 u_v + 4 u1 - u2)/(2h) = 0
                    coeff = a / (2.0 * h)
                    A[v_dof, v_dof] += 3.0 * coeff
                    A[v_dof, u1] += -4.0 * coeff
                    A[v_dof, u2] += 1.0 * coeff
                else:
                    # Fallback ordre 1 si pas assez de points
                    u1 = dofs[0] if position == "start" else dofs[-1]
                    coeff = a / h
                    A[v_dof, v_dof] += coeff
                    A[v_dof, u1] += -coeff

        return A.tocsr(), g

    def solve_direct(self, epsilon_dict=None, source_intensity=1.0, width=0.05):
        """Résout le problème direct: A*u = g"""
        A, g = self.assemble_system(epsilon_dict, source_intensity, width=width)
        self.u = spsolve(A, g)
        return self.u

    # ========================================================================
    # COÛT
    # ========================================================================

    def compute_cost_functional(self, u_data):
        """Calcule J = 1/2 ∫ (u - u_data)² dx"""
        if self.u is None:
            raise ValueError("Résoudre d'abord le problème direct")

        J = 0.0
        for edge in self.graph.edges:
            dofs = self.graph.get_edge_dofs(edge["id"])
            h = edge["h"]
            diff = self.u[dofs] - u_data[dofs]
            J += 0.5 * h * np.sum(diff**2)

        return J

    # ========================================================================
    # SENSIBILITÉ wrt epsilon : A w = ∂g/∂ε
    # ========================================================================

    def assemble_sensitivity_rhs_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
        """Assemble ∂g/∂ε"""
        n = self.graph.n_dof
        dg_deps = np.zeros(n)

        for edge_id, epsilon in epsilon_dict.items():
            edge = self.graph.edges[edge_id]
            h = edge["h"]
            n_pts = edge["n"]

            x = np.linspace(h, edge["length"] - h, n_pts)
            dofs = self.graph.get_edge_dofs(edge_id)

            dg = self.source_derivative_epsilon(
                x, epsilon, intensity=source_intensity, width=width
            )

            for i, dof in enumerate(dofs):
                dg_deps[dof] += h * dg[i]

        return dg_deps

    def solve_sensitivity_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
        """Résout l'équation de sensibilité: A*w = ∂g/∂ε"""
        A, _ = self.assemble_system(epsilon_dict, source_intensity, width=width)
        dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict, source_intensity, width=width)

        self.w = spsolve(A, dg_deps)
        return self.w

    def compute_gradient_sensitivity_epsilon(self, epsilon_dict, u_data, source_intensity=1.0, width=0.05):
        """Gradient via sensibilité directe: dJ/dε"""
        self.solve_direct(epsilon_dict, source_intensity, width=width)
        w = self.solve_sensitivity_epsilon(epsilon_dict, source_intensity, width=width)

        grad = 0.0
        for edge in self.graph.edges:
            dofs = self.graph.get_edge_dofs(edge["id"])
            h = edge["h"]
            diff = self.u[dofs] - u_data[dofs]
            grad += h * np.dot(diff, w[dofs])

        return grad

    # ========================================================================
    # ADJOINT
    # ========================================================================

    def assemble_adjoint_rhs(self, u_data):
        """Assemble le second membre pour l'équation adjointe"""
        n = self.graph.n_dof
        rhs = np.zeros(n)

        if self.u is None:
            raise ValueError("Résoudre d'abord le problème direct")

        for edge in self.graph.edges:
            edge_id = edge["id"]
            h = edge["h"]
            dofs = self.graph.get_edge_dofs(edge_id)

            rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])

        return rhs

    def solve_adjoint(self, epsilon_dict, u_data, source_intensity=1.0, width=0.05):
        """Résout l'équation adjointe: A^T * p = -∂J/∂u"""
        A, _ = self.assemble_system(epsilon_dict, source_intensity, width=width)
        rhs_adjoint = self.assemble_adjoint_rhs(u_data)

        self.p = spsolve(A.T, rhs_adjoint)
        return self.p

    def compute_gradient_adjoint_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
        """Gradient via méthode adjointe: dJ/dε = - <p, ∂g/∂ε>"""
        if self.p is None:
            raise ValueError("Résoudre d'abord l'équation adjointe")

        dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict, source_intensity, width=width)
        grad_adj = -np.dot(self.p, dg_deps)
        return grad_adj

    # ========================================================================
    # DIFFÉRENCES FINIES wrt epsilon (alpha = pas DF)
    # ========================================================================

    def compute_gradient_fd_epsilon(
        self,
        edge_id,
        epsilon,
        u_data,
        source_intensity=1.0,
        width=0.05,
        alpha_fd=1e-6,
    ):
        """Gradient par différences finies centrées: dJ/dε ≈ (J(ε+α)-J(ε-α))/(2α)"""
        eps_plus = {edge_id: epsilon + alpha_fd}
        eps_minus = {edge_id: epsilon - alpha_fd}

        self.solve_direct(eps_plus, source_intensity, width=width)
        J_plus = self.compute_cost_functional(u_data)

        self.solve_direct(eps_minus, source_intensity, width=width)
        J_minus = self.compute_cost_functional(u_data)

        # remettre l'état au point courant
        self.solve_direct({edge_id: epsilon}, source_intensity, width=width)

        return (J_plus - J_minus) / (2 * alpha_fd)

    # ========================================================================
    # VALIDATION GRADIENTS (3 méthodes) wrt epsilon
    # ========================================================================

    def validate_gradient_three_methods_epsilon(
        self,
        edge_id,
        epsilon,
        u_data,
        source_intensity=1.0,
        width=0.05,
        alpha_fd=None,
    ):
        """Compare DF / Sensibilité / Adjoint pour dJ/dε"""
        if alpha_fd is None:
            alpha_fd = np.finfo(float).eps ** (1 / 3)

        epsilon_dict = {edge_id: float(epsilon)}

        print(f"\n{'='*80}")
        print("VALIDATION GRADIENT dJ/dε - COMPARAISON DES 3 MÉTHODES")
        print(f"{'='*80}")
        print(f"Pas DF alpha_fd = {alpha_fd:.3e}")
        print(f"Source intensity = {source_intensity}")
        print(f"epsilon = {epsilon:.6f}")
        print(f"\nOBJECTIF: Sensibilité ≈ Adjoint ; DF dépend de alpha_fd\n")

        print("Calcul méthode 1: Différences finies (ε)...")
        grad_fd = self.compute_gradient_fd_epsilon(
            edge_id, epsilon, u_data, source_intensity=source_intensity, width=width, alpha_fd=alpha_fd
        )

        print("Calcul méthode 2: Sensibilité directe (ε)...")
        grad_sens = self.compute_gradient_sensitivity_epsilon(
            epsilon_dict, u_data, source_intensity=source_intensity, width=width
        )

        print("Calcul méthode 3: Méthode adjointe (ε)...")
        self.solve_direct(epsilon_dict, source_intensity, width=width)
        self.solve_adjoint(epsilon_dict, u_data, source_intensity, width=width)
        grad_adj = self.compute_gradient_adjoint_epsilon(
            epsilon_dict, source_intensity=source_intensity, width=width
        )

        print(f"\n{'-'*80}")
        print(f"{'MÉTHODE':<30} {'VALEUR dJ/dε':<20} {'ERREUR vs FD':<15}")
        print(f"{'-'*80}")

        print(f"{'1. Différences finies':<30} {grad_fd:<20.12e} {'---':<15}")

        err_sens = (
            abs(grad_sens - grad_fd) / abs(grad_fd)
            if abs(grad_fd) > 1e-16
            else abs(grad_sens - grad_fd)
        )
        print(f"{'2. Sensibilité directe':<30} {grad_sens:<20.12e} {err_sens:<15.3e}")

        err_adj = (
            abs(grad_adj - grad_fd) / abs(grad_fd)
            if abs(grad_fd) > 1e-16
            else abs(grad_adj - grad_fd)
        )
        print(f"{'3. Méthode adjointe':<30} {grad_adj:<20.12e} {err_adj:<15.3e}")

        err_sens_adj = (
            abs(grad_sens - grad_adj) / abs(grad_adj)
            if abs(grad_adj) > 1e-16
            else abs(grad_sens - grad_adj)
        )

        print(f"\n{'-'*80}")
        print("Comparaison directe Sensibilité vs Adjointe:")
        print(f"  Erreur relative: {err_sens_adj:.3e}")

        print(f"\n{'='*80}")
        max_err = max(err_sens, err_adj)
        print(f"Erreur maximale (vs DF): {max_err:.3e}")

        return {
            "grad_fd": grad_fd,
            "grad_sensitivity": grad_sens,
            "grad_adjoint": grad_adj,
            "error_sens_vs_fd": err_sens,
            "error_adj_vs_fd": err_adj,
            "error_sens_vs_adj": err_sens_adj,
            "max_error": max_err,
            "alpha_fd": alpha_fd,
        }

    # ========================================================================
    # COÛT + GRADIENT wrt epsilon (pour l'optimisation)
    # ========================================================================

    def cost_and_gradient_epsilon_vector(
        self,
        epsilon_vec,
        edge_ids,
        u_data,
        source_intensity=1.0,
        width=0.05,
    ):
        """
        Calcule J et le gradient vectoriel ∇J wrt (ε0, ε1, ...)
        """
        epsilon_dict = {eid: epsilon_vec[i] for i, eid in enumerate(edge_ids)}

        # Problème direct
        self.solve_direct(epsilon_dict, source_intensity, width=width)
        J = self.compute_cost_functional(u_data)

        # Adjoint
        self.solve_adjoint(epsilon_dict, u_data, source_intensity, width=width)

        # Gradient vectoriel
        grad = np.zeros(len(edge_ids))
        for i, eid in enumerate(edge_ids):
            dg = self.assemble_sensitivity_rhs_epsilon(
                {eid: epsilon_vec[i]},
                source_intensity,
                width,
            )
            grad[i] = -np.dot(self.p, dg)

        return J, grad


    # ========================================================================
    # LINE SEARCH (Armijo) wrt epsilon
    # ========================================================================

    def line_search_epsilon(
        self,
        edge_id,
        epsilon,
        u_data,
        d,
        J0,
        g0,
        source_intensity=1.0,
        width=0.05,
        c=1e-4,
        step0=1.0,
    ):
        edge = self.graph.edges[edge_id]
        L = edge["length"]

        slope0 = g0 * d
        if slope0 >= 0:
            d = -g0
            slope0 = g0 * d

        step = step0
        for _ in range(20):
            eps_try = np.clip(epsilon + step * d, 0.0, L)

            J_try, _ = self.cost_and_gradient_epsilon(
                edge_id, eps_try, u_data, source_intensity=source_intensity, width=width
            )

            if J_try <= J0 + c * step * slope0:
                return eps_try

            step *= 0.5

        return epsilon

    # ========================================================================
    # GRADIENT CONJUGUÉ NON LINÉAIRE wrt epsilon
    # ========================================================================

    # def conjugate_gradient_epsilon(
    #     self,
    #     edge_id,
    #     u_data,
    #     epsilon_init,
    #     source_intensity=1.0,
    #     width=0.05,
    #     max_iter=50,
    #     tol=1e-8,
    #     verbose=True,
    # ):
    #     eps = float(epsilon_init)
    #     J, g = self.cost_and_gradient_epsilon(
    #         edge_id, eps, u_data, source_intensity=source_intensity, width=width
    #     )
    #     d = -g

    #     if verbose:
    #         print("\n" + "=" * 70)
    #         print("INVERSION PAR GRADIENT CONJUGUÉ (ε)")
    #         print("=" * 70)
    #         print(f"Init  | ε = {eps:.6f} | J = {J:.3e}")

    #     for k in range(max_iter):
    #         if abs(g) < tol:
    #             print("✓ Convergence atteinte")
    #             break

    #         eps_new = self.line_search_epsilon(
    #             edge_id,
    #             eps,
    #             u_data,
    #             d,
    #             J,
    #             g,
    #             source_intensity=source_intensity,
    #             width=width,
    #         )

    #         J_new, g_new = self.cost_and_gradient_epsilon(
    #             edge_id, eps_new, u_data, source_intensity=source_intensity, width=width
    #         )

    #         if verbose:
    #             print(
    #                 f"Iter {k+1:02d} | ε = {eps_new:.6f} | "
    #                 f"J = {J_new:.3e} | |grad| = {abs(g_new):.3e}"
    #             )

    #         beta = (g_new * g_new) / (g * g) if abs(g) > 1e-30 else 0.0
    #         d = -g_new + beta * d

    #         if g_new * d >= 0:
    #             d = -g_new

    #         eps, g, J = eps_new, g_new, J_new

    #     return eps

    # ========================================================================
# GRADIENT CONJUGUÉ SÉCURISÉ (adapté à optimisation alternée)
# ========================================================================

    def conjugate_gradient_epsilon_vector(
        self,
        epsilon_init,
        edge_ids,
        u_data,
        source_intensity=1.0,
        max_iter=100,
        tol=1e-8,
    ):
        eps = np.array(epsilon_init, dtype=float)

        J, g = self.cost_and_gradient_epsilon_vector(
            eps, edge_ids, u_data, source_intensity
        )
        d = -g

        for k in range(max_iter):
            if np.linalg.norm(g) < tol:
                break

            # Line search simple
            alpha = 1.0
            for _ in range(10):
                eps_try = eps + alpha * d
                for i, eid in enumerate(edge_ids):
                    L = self.graph.edges[eid]["length"]
                    eps_try[i] = np.clip(eps_try[i], 0.0, L)

                J_try, _ = self.cost_and_gradient_epsilon_vector(
                    eps_try, edge_ids, u_data, source_intensity
                )

                if J_try < J:
                    break
                alpha *= 0.5

            eps_new = eps + alpha * d
            J_new, g_new = self.cost_and_gradient_epsilon_vector(
                eps_new, edge_ids, u_data, source_intensity
            )

            beta = np.dot(g_new, g_new) / np.dot(g, g)
            d = -g_new + beta * d

            eps, g, J = eps_new, g_new, J_new

        return eps

    # ========================================================================
# GRADIENT PROJETÉ wrt epsilon (méthode de référence)
# ========================================================================

    def projected_gradient_epsilon_vector(
        self,
        epsilon_init,
        edge_ids,
        u_data,
        source_intensity=1.0,
        step=0.2,
        max_iter=200,
        tol=1e-8,
    ):
        eps = np.array(epsilon_init, dtype=float)

        for k in range(max_iter):
            J, g = self.cost_and_gradient_epsilon_vector(
                eps, edge_ids, u_data, source_intensity
            )

            if np.linalg.norm(g) < tol:
                break

            eps -= step * g

            # Projection
            for i, eid in enumerate(edge_ids):
                L = self.graph.edges[eid]["length"]
                eps[i] = np.clip(eps[i], 0.0, L)

        return eps

    # ========================================================================
    # VISUALISATIONS (inchangées)
    # ========================================================================

    def plot_solution_on_graph(self, epsilon_dict=None, title="Solution u sur le graphe"):
        """Visualise la solution u sur le graphe"""
        if self.u is None:
            print("Aucune solution à afficher.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        for edge in self.graph.edges:
            v_start = edge["v_start"]
            v_end = edge["v_end"]
            edge_id = edge["id"]

            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]

                dofs = self.graph.get_edge_dofs(edge_id)
                u_edge = self.u[dofs]

                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)

                scatter = ax.scatter(
                    x_interp,
                    y_interp,
                    c=u_edge,
                    cmap="coolwarm",
                    s=100,
                    vmin=self.u.min(),
                    vmax=self.u.max(),
                    zorder=2,
                    edgecolors="black",
                    linewidth=0.5,
                )

                ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge["length"]
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(
                        x_source,
                        y_source,
                        "y*",
                        markersize=30,
                        markeredgecolor="black",
                        markeredgewidth=2,
                        zorder=3,
                        label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
                    )

        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(
                    x,
                    y,
                    "rs",
                    markersize=16,
                    zorder=4,
                    label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
                )
            else:
                internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
                ax.plot(
                    x,
                    y,
                    "go",
                    markersize=16,
                    zorder=4,
                    label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
                )

        cbar = plt.colorbar(scatter, ax=ax, label="Valeur de u")
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=11, loc="best")
        plt.tight_layout()
        plt.show()

    def plot_sensitivity_on_graph(self, epsilon_dict=None, title="Sensibilité w = ∂u/∂ε"):
        """Visualise la sensibilité w sur le graphe"""
        if self.w is None:
            print("Aucune sensibilité à afficher.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        for edge in self.graph.edges:
            v_start = edge["v_start"]
            v_end = edge["v_end"]
            edge_id = edge["id"]

            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]

                dofs = self.graph.get_edge_dofs(edge_id)
                w_edge = self.w[dofs]

                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)

                scatter = ax.scatter(
                    x_interp,
                    y_interp,
                    c=w_edge,
                    cmap="viridis",
                    s=100,
                    vmin=self.w.min(),
                    vmax=self.w.max(),
                    zorder=2,
                    edgecolors="black",
                    linewidth=0.5,
                )

                ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge["length"]
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(
                        x_source,
                        y_source,
                        "y*",
                        markersize=30,
                        markeredgecolor="black",
                        markeredgewidth=2,
                        zorder=3,
                        label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
                    )

        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(
                    x,
                    y,
                    "rs",
                    markersize=16,
                    zorder=4,
                    label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
                )
            else:
                internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
                ax.plot(
                    x,
                    y,
                    "go",
                    markersize=16,
                    zorder=4,
                    label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
                )

        cbar = plt.colorbar(scatter, ax=ax, label="Valeur de w")
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=11, loc="best")
        plt.tight_layout()
        plt.show()

    def plot_adjoint_on_graph(self, epsilon_dict=None, title="État adjoint p"):
        """Visualise l'état adjoint p sur le graphe"""
        if self.p is None:
            print("Aucun état adjoint à afficher.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        for edge in self.graph.edges:
            v_start = edge["v_start"]
            v_end = edge["v_end"]
            edge_id = edge["id"]

            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]

                dofs = self.graph.get_edge_dofs(edge_id)
                p_edge = self.p[dofs]

                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)

                scatter = ax.scatter(
                    x_interp,
                    y_interp,
                    c=p_edge,
                    cmap="plasma",
                    s=100,
                    vmin=self.p.min(),
                    vmax=self.p.max(),
                    zorder=2,
                    edgecolors="black",
                    linewidth=0.5,
                )

                ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge["length"]
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(
                        x_source,
                        y_source,
                        "y*",
                        markersize=30,
                        markeredgecolor="black",
                        markeredgewidth=2,
                        zorder=3,
                        label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
                    )

        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(
                    x,
                    y,
                    "rs",
                    markersize=16,
                    zorder=4,
                    label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
                )
            else:
                internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
                ax.plot(
                    x,
                    y,
                    "go",
                    markersize=16,
                    zorder=4,
                    label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
                )

        cbar = plt.colorbar(scatter, ax=ax, label="Valeur de p")
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=11, loc="best")
        plt.tight_layout()
        plt.show()

    def plot_all_results(self, epsilon_dict, u_data):
        """Affiche tous les résultats dans une grille 2x2"""
        fig = plt.figure(figsize=(16, 14))

        ax1 = plt.subplot(2, 2, 1)
        self._plot_on_axis(ax1, self.u, epsilon_dict, "Solution u", "coolwarm")

        ax2 = plt.subplot(2, 2, 2)
        if self.w is not None:
            self._plot_on_axis(ax2, self.w, epsilon_dict, "Sensibilité w = ∂u/∂ε", "viridis")
        else:
            ax2.text(0.5, 0.5, "Sensibilité non calculée", ha="center", va="center", fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        ax3 = plt.subplot(2, 2, 3)
        if self.p is not None:
            self._plot_on_axis(ax3, self.p, epsilon_dict, "État adjoint p", "plasma")
        else:
            ax3.text(0.5, 0.5, "État adjoint non calculé", ha="center", va="center", fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

        ax4 = plt.subplot(2, 2, 4)
        self._plot_on_axis(ax4, u_data, epsilon_dict, "Données observées u_data", "coolwarm")

        plt.tight_layout()
        plt.show()

    def _plot_on_axis(self, ax, data, epsilon_dict, title, cmap):
        """Fonction auxiliaire pour tracer sur un axe donné"""
        if data is None:
            return

        for edge in self.graph.edges:
            v_start = edge["v_start"]
            v_end = edge["v_end"]
            edge_id = edge["id"]

            if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
                x1, y1 = self.graph.vertex_positions[v_start]
                x2, y2 = self.graph.vertex_positions[v_end]

                dofs = self.graph.get_edge_dofs(edge_id)
                data_edge = data[dofs]

                n_pts = len(dofs)
                t = np.linspace(0, 1, n_pts)
                x_interp = x1 + t * (x2 - x1)
                y_interp = y1 + t * (y2 - y1)

                scatter = ax.scatter(
                    x_interp,
                    y_interp,
                    c=data_edge,
                    cmap=cmap,
                    s=80,
                    vmin=data.min(),
                    vmax=data.max(),
                    zorder=2,
                    edgecolors="black",
                    linewidth=0.5,
                )

                ax.plot([x1, x2], [y1, y2], "k-", linewidth=1.5, alpha=0.3, zorder=1)

                if epsilon_dict is not None and edge_id in epsilon_dict:
                    epsilon = epsilon_dict[edge_id]
                    t_source = epsilon / edge["length"]
                    x_source = x1 + t_source * (x2 - x1)
                    y_source = y1 + t_source * (y2 - y1)
                    ax.plot(
                        x_source,
                        y_source,
                        "y*",
                        markersize=20,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        zorder=3,
                    )

        for v_id, pos in self.graph.vertex_positions.items():
            x, y = pos
            if v_id in self.graph.boundary_vertices:
                ax.plot(x, y, "rs", markersize=12, zorder=4)
            else:
                ax.plot(x, y, "go", markersize=12, zorder=4)

        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")


    


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


