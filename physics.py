# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve


# class SourceLocalization:
#     """Résolution du problème de localisation de source avec méthode adjointe"""

#     def __init__(self, graph):
#         self.graph = graph
#         self.u = None
#         self.w = None
#         self.p = None
        
#     # def uzawa_epsilon_vector(
#     #     self,
#     #     epsilon_init,
#     #     edge_ids,
#     #     u_data,
#     #     source_intensity=1.0,
#     #     width=0.05,
#     #     step=0.2,
#     #     rho=5.0,
#     #     max_iter=200,
#     #     tol=1e-8,
#     #     verbose=True,
#     # ):
#     #     """
#     #     Méthode d'Uzawa pour :
#     #         min J(epsilon)
#     #         s.c. sum(epsilon) <= 1
#     #             0 <= epsilon_e <= L_e
#     #     """

#     #     eps = np.array(epsilon_init, dtype=float)
#     #     lam = 0.0  # multiplicateur de Lagrange >= 0

#     #     # projection initiale
#     #     for i, eid in enumerate(edge_ids):
#     #         L = self.graph.edges[eid]["length"]
#     #         eps[i] = np.clip(eps[i], 0.0, L)

#     #     for k in range(max_iter):
#     #         # coût + gradient adjoint
#     #         J, gradJ = self.cost_and_gradient_epsilon_vector(
#     #             eps, edge_ids, u_data, source_intensity, width=width
#     #         )

#     #         # contrainte g(eps) = sum(eps) - 1
#     #         g = np.sum(eps) - 1.0

#     #         # gradient du Lagrangien
#     #         gradL = gradJ + lam * np.ones_like(eps)

#     #         # descente primal
#     #         eps_new = eps - step * gradL

#     #         # projection locale
#     #         for i, eid in enumerate(edge_ids):
#     #             L = self.graph.edges[eid]["length"]
#     #             eps_new[i] = np.clip(eps_new[i], 0.0, L)

#     #         # mise à jour duale
#     #         g_new = np.sum(eps_new) - 1.0
#     #         lam_new = max(0.0, lam + rho * g_new)

#     #         if verbose and k % 5 == 0:
#     #             print(
#     #                 f"Uzawa {k:03d} | J={J:.3e} | "
#     #                 f"sum(eps)={np.sum(eps):.4f} | λ={lam:.3e}"
#     #             )

#     #         # arrêt
#     #         if np.linalg.norm(eps_new - eps) < tol and abs(g_new) < tol:
#     #             if verbose:
#     #                 print("✓ Uzawa convergé")
#     #             eps, lam = eps_new, lam_new
#     #             break

#     #         eps, lam = eps_new, lam_new

#     #     return eps, lam
    

#     # ========================================================================
#     # SOURCE
#     # ========================================================================

#     def source_function(self, x, epsilon, intensity=1.0, width=0.05):
#         """Fonction source gaussienne centrée en epsilon"""
#         return intensity * np.exp(-((x - epsilon) ** 2) / (2 * width**2))

#     def source_derivative_epsilon(self, x, epsilon, intensity=1.0, width=0.05):
#         """Dérivée de la source par rapport à epsilon: ∂g/∂ε"""
#         gauss = np.exp(-((x - epsilon) ** 2) / (2 * width**2))
#         return intensity * (x - epsilon) / width**2 * gauss

#     # ========================================================================
#     # ASSEMBLAGE / DIRECT
#     # ========================================================================

#     def assemble_system(self, epsilon_dict=None, source_intensity=1.0, width=0.05):
#         """Assemble le système linéaire A*u = g"""
#         n = self.graph.n_dof
#         A = lil_matrix((n, n))
#         g = np.zeros(n)

#         for edge in self.graph.edges:
#             edge_id = edge["id"]
#             h = edge["h"]
#             a = edge["a"]
#             n_pts = edge["n"]

#             x = np.linspace(h, edge["length"] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)

#             # =========================
#             # CHOIX DE LA SOURCE
#             # =========================
#             if epsilon_dict is not None and edge_id in epsilon_dict:
#                 epsilon = epsilon_dict[edge_id]
#                 g_local = self.source_function(x, epsilon, source_intensity, width=width)
#             else:
#                 g_local = np.zeros(n_pts)

#             stiffness = a / h

#             for i, dof in enumerate(dofs):
#                 g[dof] += h * g_local[i]

#                 A[dof, dof] = 2 * stiffness

#                 if i > 0:
#                     A[dof, dofs[i - 1]] = -stiffness
#                 else:
#                     v_start_dof = self.graph.get_vertex_dof(edge["v_start"])
#                     if v_start_dof is not None:
#                         A[dof, v_start_dof] = -stiffness

#                 if i + 1 < len(dofs):
#                     A[dof, dofs[i + 1]] = -stiffness
#                 else:
#                     v_end_dof = self.graph.get_vertex_dof(edge["v_end"])
#                     if v_end_dof is not None:
#                         A[dof, v_end_dof] = -stiffness

#         # Conditions de Kirchhoff aux nœuds internes (ordre 2 si possible)
#         for v_id in self.graph.vertices:
#             if v_id in self.graph.boundary_vertices:
#                 continue

#             v_dof = self.graph.get_vertex_dof(v_id)
#             incident_edges = self.graph.vertices[v_id]["edges"]

#             for edge_id, position in incident_edges:
#                 edge = self.graph.edges[edge_id]
#                 h = edge["h"]
#                 a = edge["a"]

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 npts = len(dofs)

#                 if npts >= 2:
#                     if position == "start":
#                         u1 = dofs[0]
#                         u2 = dofs[1]
#                     else:
#                         u1 = dofs[-1]
#                         u2 = dofs[-2]

#                     # sum_e a * (-3 u_v + 4 u1 - u2)/(2h) = 0
#                     coeff = a / (2.0 * h)
#                     A[v_dof, v_dof] += 3.0 * coeff
#                     A[v_dof, u1] += -4.0 * coeff
#                     A[v_dof, u2] += 1.0 * coeff
#                 else:
#                     # Fallback ordre 1 si pas assez de points
#                     u1 = dofs[0] if position == "start" else dofs[-1]
#                     coeff = a / h
#                     A[v_dof, v_dof] += coeff
#                     A[v_dof, u1] += -coeff

#         return A.tocsr(), g

#     def solve_direct(self, epsilon_dict=None, source_intensity=1.0, width=0.05):
#         """Résout le problème direct: A*u = g"""
#         A, g = self.assemble_system(epsilon_dict, source_intensity, width=width)
#         self.u = spsolve(A, g)
#         return self.u

#     # ========================================================================
#     # COÛT
#     # ========================================================================

#     def compute_cost_functional(self, u_data):
#         """Calcule J = 1/2 ∫ (u - u_data)² dx"""
#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")

#         J = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge["id"])
#             h = edge["h"]
#             diff = self.u[dofs] - u_data[dofs]
#             J += 0.5 * h * np.sum(diff**2)

#         return J

#     # ========================================================================
#     # SENSIBILITÉ wrt epsilon : A w = ∂g/∂ε
#     # ========================================================================

#     def assemble_sensitivity_rhs_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
#         """Assemble ∂g/∂ε"""
#         n = self.graph.n_dof
#         dg_deps = np.zeros(n)

#         for edge_id, epsilon in epsilon_dict.items():
#             edge = self.graph.edges[edge_id]
#             h = edge["h"]
#             n_pts = edge["n"]

#             x = np.linspace(h, edge["length"] - h, n_pts)
#             dofs = self.graph.get_edge_dofs(edge_id)

#             dg = self.source_derivative_epsilon(
#                 x, epsilon, intensity=source_intensity, width=width
#             )

#             for i, dof in enumerate(dofs):
#                 dg_deps[dof] += h * dg[i]

#         return dg_deps

#     def solve_sensitivity_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
#         """Résout l'équation de sensibilité: A*w = ∂g/∂ε"""
#         A, _ = self.assemble_system(epsilon_dict, source_intensity, width=width)
#         dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict, source_intensity, width=width)

#         self.w = spsolve(A, dg_deps)
#         return self.w

#     def compute_gradient_sensitivity_epsilon(self, epsilon_dict, u_data, source_intensity=1.0, width=0.05):
#         """Gradient via sensibilité directe: dJ/dε"""
#         self.solve_direct(epsilon_dict, source_intensity, width=width)
#         w = self.solve_sensitivity_epsilon(epsilon_dict, source_intensity, width=width)

#         grad = 0.0
#         for edge in self.graph.edges:
#             dofs = self.graph.get_edge_dofs(edge["id"])
#             h = edge["h"]
#             diff = self.u[dofs] - u_data[dofs]
#             grad += h * np.dot(diff, w[dofs])

#         return grad

#     # ========================================================================
#     # ADJOINT
#     # ========================================================================

#     def assemble_adjoint_rhs(self, u_data):
#         """Assemble le second membre pour l'équation adjointe"""
#         n = self.graph.n_dof
#         rhs = np.zeros(n)

#         if self.u is None:
#             raise ValueError("Résoudre d'abord le problème direct")

#         for edge in self.graph.edges:
#             edge_id = edge["id"]
#             h = edge["h"]
#             dofs = self.graph.get_edge_dofs(edge_id)

#             rhs[dofs] = -h * (self.u[dofs] - u_data[dofs])

#         return rhs

#     def solve_adjoint(self, epsilon_dict, u_data, source_intensity=1.0, width=0.05):
#         """Résout l'équation adjointe: A^T * p = -∂J/∂u"""
#         A, _ = self.assemble_system(epsilon_dict, source_intensity, width=width)
#         rhs_adjoint = self.assemble_adjoint_rhs(u_data)

#         self.p = spsolve(A.T, rhs_adjoint)
#         return self.p

#     def compute_gradient_adjoint_epsilon(self, epsilon_dict, source_intensity=1.0, width=0.05):
#         """Gradient via méthode adjointe: dJ/dε = - <p, ∂g/∂ε>"""
#         if self.p is None:
#             raise ValueError("Résoudre d'abord l'équation adjointe")

#         dg_deps = self.assemble_sensitivity_rhs_epsilon(epsilon_dict, source_intensity, width=width)
#         grad_adj = -np.dot(self.p, dg_deps)
#         return grad_adj

#     # ========================================================================
#     # DIFFÉRENCES FINIES wrt epsilon (alpha = pas DF)
#     # ========================================================================

#     def compute_gradient_fd_epsilon(
#         self,
#         edge_id,
#         epsilon,
#         u_data,
#         source_intensity=1.0,
#         width=0.05,
#         alpha_fd=1e-6,
#     ):
#         """Gradient par différences finies centrées: dJ/dε ≈ (J(ε+α)-J(ε-α))/(2α)"""
#         eps_plus = {edge_id: epsilon + alpha_fd}
#         eps_minus = {edge_id: epsilon - alpha_fd}

#         self.solve_direct(eps_plus, source_intensity, width=width)
#         J_plus = self.compute_cost_functional(u_data)

#         self.solve_direct(eps_minus, source_intensity, width=width)
#         J_minus = self.compute_cost_functional(u_data)

#         # remettre l'état au point courant
#         self.solve_direct({edge_id: epsilon}, source_intensity, width=width)

#         return (J_plus - J_minus) / (2 * alpha_fd)

#     # ========================================================================
#     # VALIDATION GRADIENTS (3 méthodes) wrt epsilon
#     # ========================================================================

#     def validate_gradient_three_methods_epsilon(
#         self,
#         edge_id,
#         epsilon,
#         u_data,
#         source_intensity=1.0,
#         width=0.05,
#         alpha_fd=None,
#     ):
#         """Compare DF / Sensibilité / Adjoint pour dJ/dε"""
#         if alpha_fd is None:
#             alpha_fd = np.finfo(float).eps ** (1 / 3)

#         epsilon_dict = {edge_id: float(epsilon)}

#         print(f"\n{'='*80}")
#         print("VALIDATION GRADIENT dJ/dε - COMPARAISON DES 3 MÉTHODES")
#         print(f"{'='*80}")
#         print(f"Pas DF alpha_fd = {alpha_fd:.3e}")
#         print(f"Source intensity = {source_intensity}")
#         print(f"epsilon = {epsilon:.6f}")
#         print(f"\nOBJECTIF: Sensibilité ≈ Adjoint ; DF dépend de alpha_fd\n")

#         print("Calcul méthode 1: Différences finies (ε)...")
#         grad_fd = self.compute_gradient_fd_epsilon(
#             edge_id, epsilon, u_data, source_intensity=source_intensity, width=width, alpha_fd=alpha_fd
#         )

#         print("Calcul méthode 2: Sensibilité directe (ε)...")
#         grad_sens = self.compute_gradient_sensitivity_epsilon(
#             epsilon_dict, u_data, source_intensity=source_intensity, width=width
#         )

#         print("Calcul méthode 3: Méthode adjointe (ε)...")
#         self.solve_direct(epsilon_dict, source_intensity, width=width)
#         self.solve_adjoint(epsilon_dict, u_data, source_intensity, width=width)
#         grad_adj = self.compute_gradient_adjoint_epsilon(
#             epsilon_dict, source_intensity=source_intensity, width=width
#         )

#         print(f"\n{'-'*80}")
#         print(f"{'MÉTHODE':<30} {'VALEUR dJ/dε':<20} {'ERREUR vs FD':<15}")
#         print(f"{'-'*80}")

#         print(f"{'1. Différences finies':<30} {grad_fd:<20.12e} {'---':<15}")

#         err_sens = (
#             abs(grad_sens - grad_fd) / abs(grad_fd)
#             if abs(grad_fd) > 1e-16
#             else abs(grad_sens - grad_fd)
#         )
#         print(f"{'2. Sensibilité directe':<30} {grad_sens:<20.12e} {err_sens:<15.3e}")

#         err_adj = (
#             abs(grad_adj - grad_fd) / abs(grad_fd)
#             if abs(grad_fd) > 1e-16
#             else abs(grad_adj - grad_fd)
#         )
#         print(f"{'3. Méthode adjointe':<30} {grad_adj:<20.12e} {err_adj:<15.3e}")

#         err_sens_adj = (
#             abs(grad_sens - grad_adj) / abs(grad_adj)
#             if abs(grad_adj) > 1e-16
#             else abs(grad_sens - grad_adj)
#         )

#         print(f"\n{'-'*80}")
#         print("Comparaison directe Sensibilité vs Adjointe:")
#         print(f"  Erreur relative: {err_sens_adj:.3e}")

#         print(f"\n{'='*80}")
#         max_err = max(err_sens, err_adj)
#         print(f"Erreur maximale (vs DF): {max_err:.3e}")

#         return {
#             "grad_fd": grad_fd,
#             "grad_sensitivity": grad_sens,
#             "grad_adjoint": grad_adj,
#             "error_sens_vs_fd": err_sens,
#             "error_adj_vs_fd": err_adj,
#             "error_sens_vs_adj": err_sens_adj,
#             "max_error": max_err,
#             "alpha_fd": alpha_fd,
#         }

#     # ========================================================================
#     # COÛT + GRADIENT wrt epsilon (pour l'optimisation)
#     # ========================================================================

#     def cost_and_gradient_epsilon_vector(
#         self,
#         epsilon_vec,
#         edge_ids,
#         u_data,
#         source_intensity=1.0,
#         width=0.05,
#     ):
#         """
#         Calcule J et le gradient vectoriel ∇J wrt (ε0, ε1, ...)
#         """
#         epsilon_dict = {eid: epsilon_vec[i] for i, eid in enumerate(edge_ids)}

#         # Problème direct
#         self.solve_direct(epsilon_dict, source_intensity, width=width)
#         J = self.compute_cost_functional(u_data)

#         # Adjoint
#         self.solve_adjoint(epsilon_dict, u_data, source_intensity, width=width)

#         # Gradient vectoriel
#         grad = np.zeros(len(edge_ids))
#         for i, eid in enumerate(edge_ids):
#             dg = self.assemble_sensitivity_rhs_epsilon(
#                 {eid: epsilon_vec[i]},
#                 source_intensity,
#                 width,
#             )
#             grad[i] = -np.dot(self.p, dg)

#         return J, grad


#     # ========================================================================
#     # LINE SEARCH (Armijo) wrt epsilon
#     # ========================================================================

#     def line_search_epsilon(
#         self,
#         edge_id,
#         epsilon,
#         u_data,
#         d,
#         J0,
#         g0,
#         source_intensity=1.0,
#         width=0.05,
#         c=1e-4,
#         step0=1.0,
#     ):
#         edge = self.graph.edges[edge_id]
#         L = edge["length"]

#         slope0 = g0 * d
#         if slope0 >= 0:
#             d = -g0
#             slope0 = g0 * d

#         step = step0
#         for _ in range(20):
#             eps_try = np.clip(epsilon + step * d, 0.0, L)

#             J_try, _ = self.cost_and_gradient_epsilon(
#                 edge_id, eps_try, u_data, source_intensity=source_intensity, width=width
#             )

#             if J_try <= J0 + c * step * slope0:
#                 return eps_try

#             step *= 0.5

#         return epsilon

#     # ========================================================================
#     # GRADIENT CONJUGUÉ NON LINÉAIRE wrt epsilon
#     # ========================================================================

#     # def conjugate_gradient_epsilon(
#     #     self,
#     #     edge_id,
#     #     u_data,
#     #     epsilon_init,
#     #     source_intensity=1.0,
#     #     width=0.05,
#     #     max_iter=50,
#     #     tol=1e-8,
#     #     verbose=True,
#     # ):
#     #     eps = float(epsilon_init)
#     #     J, g = self.cost_and_gradient_epsilon(
#     #         edge_id, eps, u_data, source_intensity=source_intensity, width=width
#     #     )
#     #     d = -g

#     #     if verbose:
#     #         print("\n" + "=" * 70)
#     #         print("INVERSION PAR GRADIENT CONJUGUÉ (ε)")
#     #         print("=" * 70)
#     #         print(f"Init  | ε = {eps:.6f} | J = {J:.3e}")

#     #     for k in range(max_iter):
#     #         if abs(g) < tol:
#     #             print("✓ Convergence atteinte")
#     #             break

#     #         eps_new = self.line_search_epsilon(
#     #             edge_id,
#     #             eps,
#     #             u_data,
#     #             d,
#     #             J,
#     #             g,
#     #             source_intensity=source_intensity,
#     #             width=width,
#     #         )

#     #         J_new, g_new = self.cost_and_gradient_epsilon(
#     #             edge_id, eps_new, u_data, source_intensity=source_intensity, width=width
#     #         )

#     #         if verbose:
#     #             print(
#     #                 f"Iter {k+1:02d} | ε = {eps_new:.6f} | "
#     #                 f"J = {J_new:.3e} | |grad| = {abs(g_new):.3e}"
#     #             )

#     #         beta = (g_new * g_new) / (g * g) if abs(g) > 1e-30 else 0.0
#     #         d = -g_new + beta * d

#     #         if g_new * d >= 0:
#     #             d = -g_new

#     #         eps, g, J = eps_new, g_new, J_new

#     #     return eps

#     # ========================================================================
# # GRADIENT CONJUGUÉ SÉCURISÉ (adapté à optimisation alternée)
# # ========================================================================

#     def conjugate_gradient_epsilon_vector(
#         self,
#         epsilon_init,
#         edge_ids,
#         u_data,
#         source_intensity=1.0,
#         max_iter=100,
#         tol=1e-8,
#     ):
#         eps = np.array(epsilon_init, dtype=float)

#         J, g = self.cost_and_gradient_epsilon_vector(
#             eps, edge_ids, u_data, source_intensity
#         )
#         d = -g

#         for k in range(max_iter):
#             if np.linalg.norm(g) < tol:
#                 break

#             # Line search simple
#             alpha = 1.0
#             for _ in range(10):
#                 eps_try = eps + alpha * d
#                 for i, eid in enumerate(edge_ids):
#                     L = self.graph.edges[eid]["length"]
#                     eps_try[i] = np.clip(eps_try[i], 0.0, L)

#                 J_try, _ = self.cost_and_gradient_epsilon_vector(
#                     eps_try, edge_ids, u_data, source_intensity
#                 )

#                 if J_try < J:
#                     break
#                 alpha *= 0.5

#             eps_new = eps + alpha * d
#             J_new, g_new = self.cost_and_gradient_epsilon_vector(
#                 eps_new, edge_ids, u_data, source_intensity
#             )

#             beta = np.dot(g_new, g_new) / np.dot(g, g)
#             d = -g_new + beta * d

#             eps, g, J = eps_new, g_new, J_new

#         return eps

#     # ========================================================================
# # GRADIENT PROJETÉ wrt epsilon (méthode de référence)
# # ========================================================================
#     # ========================================================================
# # UZAWA – contrainte somme(epsilon) <= 1
# # ========================================================================


#     def projected_gradient_epsilon_vector(
#         self,
#         epsilon_init,
#         edge_ids,
#         u_data,
#         source_intensity=1.0,
#         step=0.2,
#         max_iter=200,
#         tol=1e-8,
#     ):
#         eps = np.array(epsilon_init, dtype=float)

#         for k in range(max_iter):
#             J, g = self.cost_and_gradient_epsilon_vector(
#                 eps, edge_ids, u_data, source_intensity
#             )

#             if np.linalg.norm(g) < tol:
#                 break

#             eps -= step * g

#             # Projection
#             for i, eid in enumerate(edge_ids):
#                 L = self.graph.edges[eid]["length"]
#                 eps[i] = np.clip(eps[i], 0.0, L)

#         return eps

#     # ========================================================================
#     # VISUALISATIONS (inchangées)
#     # ========================================================================

#     def plot_solution_on_graph(self, epsilon_dict=None, title="Solution u sur le graphe"):
#         """Visualise la solution u sur le graphe"""
#         if self.u is None:
#             print("Aucune solution à afficher.")
#             return

#         fig, ax = plt.subplots(figsize=(12, 10))

#         for edge in self.graph.edges:
#             v_start = edge["v_start"]
#             v_end = edge["v_end"]
#             edge_id = edge["id"]

#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 u_edge = self.u[dofs]

#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)

#                 scatter = ax.scatter(
#                     x_interp,
#                     y_interp,
#                     c=u_edge,
#                     cmap="coolwarm",
#                     s=100,
#                     vmin=self.u.min(),
#                     vmax=self.u.max(),
#                     zorder=2,
#                     edgecolors="black",
#                     linewidth=0.5,
#                 )

#                 ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge["length"]
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(
#                         x_source,
#                         y_source,
#                         "y*",
#                         markersize=30,
#                         markeredgecolor="black",
#                         markeredgewidth=2,
#                         zorder=3,
#                         label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
#                     )

#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(
#                     x,
#                     y,
#                     "rs",
#                     markersize=16,
#                     zorder=4,
#                     label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
#                 )
#             else:
#                 internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
#                 ax.plot(
#                     x,
#                     y,
#                     "go",
#                     markersize=16,
#                     zorder=4,
#                     label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
#                 )

#         cbar = plt.colorbar(scatter, ax=ax, label="Valeur de u")
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel("x", fontsize=13)
#         ax.set_ylabel("y", fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight="bold")
#         ax.grid(True, alpha=0.3)
#         ax.axis("equal")
#         ax.legend(fontsize=11, loc="best")
#         plt.tight_layout()
#         plt.show()

#     def plot_sensitivity_on_graph(self, epsilon_dict=None, title="Sensibilité w = ∂u/∂ε"):
#         """Visualise la sensibilité w sur le graphe"""
#         if self.w is None:
#             print("Aucune sensibilité à afficher.")
#             return

#         fig, ax = plt.subplots(figsize=(12, 10))

#         for edge in self.graph.edges:
#             v_start = edge["v_start"]
#             v_end = edge["v_end"]
#             edge_id = edge["id"]

#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 w_edge = self.w[dofs]

#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)

#                 scatter = ax.scatter(
#                     x_interp,
#                     y_interp,
#                     c=w_edge,
#                     cmap="viridis",
#                     s=100,
#                     vmin=self.w.min(),
#                     vmax=self.w.max(),
#                     zorder=2,
#                     edgecolors="black",
#                     linewidth=0.5,
#                 )

#                 ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge["length"]
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(
#                         x_source,
#                         y_source,
#                         "y*",
#                         markersize=30,
#                         markeredgecolor="black",
#                         markeredgewidth=2,
#                         zorder=3,
#                         label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
#                     )

#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(
#                     x,
#                     y,
#                     "rs",
#                     markersize=16,
#                     zorder=4,
#                     label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
#                 )
#             else:
#                 internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
#                 ax.plot(
#                     x,
#                     y,
#                     "go",
#                     markersize=16,
#                     zorder=4,
#                     label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
#                 )

#         cbar = plt.colorbar(scatter, ax=ax, label="Valeur de w")
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel("x", fontsize=13)
#         ax.set_ylabel("y", fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight="bold")
#         ax.grid(True, alpha=0.3)
#         ax.axis("equal")
#         ax.legend(fontsize=11, loc="best")
#         plt.tight_layout()
#         plt.show()

#     def plot_adjoint_on_graph(self, epsilon_dict=None, title="État adjoint p"):
#         """Visualise l'état adjoint p sur le graphe"""
#         if self.p is None:
#             print("Aucun état adjoint à afficher.")
#             return

#         fig, ax = plt.subplots(figsize=(12, 10))

#         for edge in self.graph.edges:
#             v_start = edge["v_start"]
#             v_end = edge["v_end"]
#             edge_id = edge["id"]

#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 p_edge = self.p[dofs]

#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)

#                 scatter = ax.scatter(
#                     x_interp,
#                     y_interp,
#                     c=p_edge,
#                     cmap="plasma",
#                     s=100,
#                     vmin=self.p.min(),
#                     vmax=self.p.max(),
#                     zorder=2,
#                     edgecolors="black",
#                     linewidth=0.5,
#                 )

#                 ax.plot([x1, x2], [y1, y2], "k-", linewidth=2, alpha=0.3, zorder=1)

#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge["length"]
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(
#                         x_source,
#                         y_source,
#                         "y*",
#                         markersize=30,
#                         markeredgecolor="black",
#                         markeredgewidth=2,
#                         zorder=3,
#                         label="Source" if edge_id == list(epsilon_dict.keys())[0] else "",
#                     )

#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(
#                     x,
#                     y,
#                     "rs",
#                     markersize=16,
#                     zorder=4,
#                     label="Bord" if v_id == list(self.graph.boundary_vertices)[0] else "",
#                 )
#             else:
#                 internal = list(set(self.graph.vertices.keys()) - self.graph.boundary_vertices)
#                 ax.plot(
#                     x,
#                     y,
#                     "go",
#                     markersize=16,
#                     zorder=4,
#                     label="Interne" if (len(internal) > 0 and v_id == internal[0]) else "",
#                 )

#         cbar = plt.colorbar(scatter, ax=ax, label="Valeur de p")
#         cbar.ax.tick_params(labelsize=10)
#         ax.set_xlabel("x", fontsize=13)
#         ax.set_ylabel("y", fontsize=13)
#         ax.set_title(title, fontsize=15, fontweight="bold")
#         ax.grid(True, alpha=0.3)
#         ax.axis("equal")
#         ax.legend(fontsize=11, loc="best")
#         plt.tight_layout()
#         plt.show()

#     def plot_all_results(self, epsilon_dict, u_data):
#         """Affiche tous les résultats dans une grille 2x2"""
#         fig = plt.figure(figsize=(16, 14))

#         ax1 = plt.subplot(2, 2, 1)
#         self._plot_on_axis(ax1, self.u, epsilon_dict, "Solution u", "coolwarm")

#         ax2 = plt.subplot(2, 2, 2)
#         if self.w is not None:
#             self._plot_on_axis(ax2, self.w, epsilon_dict, "Sensibilité w = ∂u/∂ε", "viridis")
#         else:
#             ax2.text(0.5, 0.5, "Sensibilité non calculée", ha="center", va="center", fontsize=14)
#             ax2.set_xlim(0, 1)
#             ax2.set_ylim(0, 1)

#         ax3 = plt.subplot(2, 2, 3)
#         if self.p is not None:
#             self._plot_on_axis(ax3, self.p, epsilon_dict, "État adjoint p", "plasma")
#         else:
#             ax3.text(0.5, 0.5, "État adjoint non calculé", ha="center", va="center", fontsize=14)
#             ax3.set_xlim(0, 1)
#             ax3.set_ylim(0, 1)

#         ax4 = plt.subplot(2, 2, 4)
#         self._plot_on_axis(ax4, u_data, epsilon_dict, "Données observées u_data", "coolwarm")

#         plt.tight_layout()
#         plt.show()

#     def _plot_on_axis(self, ax, data, epsilon_dict, title, cmap):
#         """Fonction auxiliaire pour tracer sur un axe donné"""
#         if data is None:
#             return

#         for edge in self.graph.edges:
#             v_start = edge["v_start"]
#             v_end = edge["v_end"]
#             edge_id = edge["id"]

#             if v_start in self.graph.vertex_positions and v_end in self.graph.vertex_positions:
#                 x1, y1 = self.graph.vertex_positions[v_start]
#                 x2, y2 = self.graph.vertex_positions[v_end]

#                 dofs = self.graph.get_edge_dofs(edge_id)
#                 data_edge = data[dofs]

#                 n_pts = len(dofs)
#                 t = np.linspace(0, 1, n_pts)
#                 x_interp = x1 + t * (x2 - x1)
#                 y_interp = y1 + t * (y2 - y1)

#                 scatter = ax.scatter(
#                     x_interp,
#                     y_interp,
#                     c=data_edge,
#                     cmap=cmap,
#                     s=80,
#                     vmin=data.min(),
#                     vmax=data.max(),
#                     zorder=2,
#                     edgecolors="black",
#                     linewidth=0.5,
#                 )

#                 ax.plot([x1, x2], [y1, y2], "k-", linewidth=1.5, alpha=0.3, zorder=1)

#                 if epsilon_dict is not None and edge_id in epsilon_dict:
#                     epsilon = epsilon_dict[edge_id]
#                     t_source = epsilon / edge["length"]
#                     x_source = x1 + t_source * (x2 - x1)
#                     y_source = y1 + t_source * (y2 - y1)
#                     ax.plot(
#                         x_source,
#                         y_source,
#                         "y*",
#                         markersize=20,
#                         markeredgecolor="black",
#                         markeredgewidth=1.5,
#                         zorder=3,
#                     )

#         for v_id, pos in self.graph.vertex_positions.items():
#             x, y = pos
#             if v_id in self.graph.boundary_vertices:
#                 ax.plot(x, y, "rs", markersize=12, zorder=4)
#             else:
#                 ax.plot(x, y, "go", markersize=12, zorder=4)

#         plt.colorbar(scatter, ax=ax)
#         ax.set_xlabel("x", fontsize=11)
#         ax.set_ylabel("y", fontsize=11)
#         ax.set_title(title, fontsize=12, fontweight="bold")
#         ax.grid(True, alpha=0.3)
#         ax.axis("equal")
    


    


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
    # UZAWA – CONTRAINTE DE SPARSITÉ L1 (VERSION CORRIGÉE)
    # ========================================================================
    
    def uzawa_epsilon_vector(
        self,
        epsilon_init,
        edge_ids,
        u_data,
        source_intensity=1.0,
        K_max=1.0,  # contrainte de sparsité L1
        width=0.05,
        step_init=0.5,
        rho=1.0,
        max_iter=500,
        tol=1e-10,
        verbose=True,
    ):
        """
        Méthode d'Uzawa avec contrainte de sparsité L1 + LINE SEARCH
        
            min J(epsilon)
            s.c. sum_e (epsilon_e / L_e) <= K_max
                 0 <= epsilon_e <= L_e
        
        Paramètres:
        -----------
        K_max : float
            Contrainte de sparsité (nombre max de sources normalisé)
            - K_max = 1.0 : au plus 1 source "pleine"
            - K_max = 2.0 : au plus 2 sources "pleines"
            - K_max = 0.5 : favorise sources partielles
        
        step_init : float
            Pas initial (adapté par line search)
        
        rho : float
            Paramètre de pénalisation augmentée
        """
        
        eps = np.array(epsilon_init, dtype=float)
        lam = 0.0  # multiplicateur de Lagrange >= 0
        
        # Longueurs des arêtes
        L = np.array([self.graph.edges[eid]["length"] for eid in edge_ids])
        
        # Projection initiale
        eps = np.clip(eps, 0.0, L)
        
        for k in range(max_iter):
            # ================================================================
            # 1. Calcul du coût et gradient de J
            # ================================================================
            J, gradJ = self.cost_and_gradient_epsilon_vector(
                eps, edge_ids, u_data, source_intensity, width=width
            )
            
            # ================================================================
            # 2. Contrainte de sparsité L1 normalisée
            # ================================================================
            # g(ε) = sum_e (ε_e / L_e) - K_max
            g = np.sum(eps / L) - K_max
            
            # Gradient de g : ∂g/∂ε_e = 1/L_e
            grad_g = 1.0 / L
            
            # ================================================================
            # 3. Gradient du Lagrangien augmenté
            # ================================================================
            # ∇L = ∇J + λ * ∇g + ρ * max(0, g) * ∇g
            if g > 0:
                # Contrainte violée
                gradL = gradJ + (lam + rho * g) * grad_g
            else:
                # Contrainte satisfaite
                gradL = gradJ + lam * grad_g
            
            # ================================================================
            # 4. LINE SEARCH + Descente de gradient projetée
            # ================================================================
            step = step_init
            
            for _ in range(15):
                eps_try = eps - step * gradL
                
                # Projection sur [0, L_e]
                eps_try = np.clip(eps_try, 0.0, L)
                
                # Calcul du nouveau coût
                J_try, _ = self.cost_and_gradient_epsilon_vector(
                    eps_try, edge_ids, u_data, source_intensity, width=width
                )
                
                # Condition d'Armijo
                if J_try < J - 1e-4 * step * np.dot(gradL, gradL):
                    eps_new = eps_try
                    break
                
                step *= 0.5
            else:
                # Si line search échoue, on garde le point
                eps_new = eps_try
            
            # ================================================================
            # 5. Mise à jour duale (multiplicateur de Lagrange)
            # ================================================================
            g_new = np.sum(eps_new / L) - K_max
            lam_new = max(0.0, lam + rho * g_new)
            
            # ================================================================
            # 6. Affichage et critère d'arrêt
            # ================================================================
            if verbose and k % 10 == 0:
                sparsity = np.sum(eps_new / L)
                print(
                    f"Uzawa {k:03d} | J={J:.3e} | "
                    f"sparsité={sparsity:.4f}/{K_max:.1f} | "
                    f"λ={lam:.3e} | step={step:.3e}"
                )
            
            # Critères d'arrêt
            delta_eps = np.linalg.norm(eps_new - eps)
            if delta_eps < tol and abs(g_new) < tol:
                if verbose:
                    print(f"\n✓ Uzawa convergé à l'itération {k}")
                    print(f"  - Changement ε : {delta_eps:.3e}")
                    print(f"  - Violation contrainte : {abs(g_new):.3e}")
                break
            
            eps, lam = eps_new, lam_new
        
        # ================================================================
        # Résultats finaux
        # ================================================================
        sparsity_final = np.sum(eps / L)
        
        if verbose:
            print("\n" + "="*70)
            print("RÉSULTATS UZAWA (avec line search)")
            print("="*70)
            print(f"Nombre d'itérations : {k+1}/{max_iter}")
            print(f"Sparsité L1 finale : {sparsity_final:.6f} (contrainte <= {K_max})")
            print(f"Multiplicateur final : λ = {lam:.3e}")
            print(f"\nPositions identifiées :")
            for i, eid in enumerate(edge_ids):
                print(f"  Arête {eid} : ε = {eps[i]:.6f} (ε̂ = {eps[i]/L[i]:.4f})")
        
        return eps, lam

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

                    coeff = a / (2.0 * h)
                    A[v_dof, v_dof] += 3.0 * coeff
                    A[v_dof, u1] += -4.0 * coeff
                    A[v_dof, u2] += 1.0 * coeff
                else:
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
    # DIFFÉRENCES FINIES wrt epsilon
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
        """Gradient par différences finies centrées"""
        eps_plus = {edge_id: epsilon + alpha_fd}
        eps_minus = {edge_id: epsilon - alpha_fd}

        self.solve_direct(eps_plus, source_intensity, width=width)
        J_plus = self.compute_cost_functional(u_data)

        self.solve_direct(eps_minus, source_intensity, width=width)
        J_minus = self.compute_cost_functional(u_data)

        self.solve_direct({edge_id: epsilon}, source_intensity, width=width)

        return (J_plus - J_minus) / (2 * alpha_fd)

    # ========================================================================
    # VALIDATION GRADIENTS
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

        grad_fd = self.compute_gradient_fd_epsilon(
            edge_id, epsilon, u_data, source_intensity=source_intensity, width=width, alpha_fd=alpha_fd
        )

        grad_sens = self.compute_gradient_sensitivity_epsilon(
            epsilon_dict, u_data, source_intensity=source_intensity, width=width
        )

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
        print(f"Erreur Sens/Adj : {err_sens_adj:.3e}")
        print(f"{'='*80}")

        return {
            "grad_fd": grad_fd,
            "grad_sensitivity": grad_sens,
            "grad_adjoint": grad_adj,
            "error_sens_vs_fd": err_sens,
            "error_adj_vs_fd": err_adj,
            "error_sens_vs_adj": err_sens_adj,
        }

    # ========================================================================
    # COÛT + GRADIENT VECTORIEL
    # ========================================================================

    def cost_and_gradient_epsilon_vector(
        self,
        epsilon_vec,
        edge_ids,
        u_data,
        source_intensity=1.0,
        width=0.05,
    ):
        """Calcule J et le gradient vectoriel ∇J wrt (ε0, ε1, ...)"""
        epsilon_dict = {eid: epsilon_vec[i] for i, eid in enumerate(edge_ids)}

        self.solve_direct(epsilon_dict, source_intensity, width=width)
        J = self.compute_cost_functional(u_data)

        self.solve_adjoint(epsilon_dict, u_data, source_intensity, width=width)

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
    # GRADIENT CONJUGUÉ VECTORIEL
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
    # GRADIENT PROJETÉ VECTORIEL
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

            for i, eid in enumerate(edge_ids):
                L = self.graph.edges[eid]["length"]
                eps[i] = np.clip(eps[i], 0.0, L)

        return eps

    # ========================================================================
    # VISUALISATIONS
    # ========================================================================

    def plot_all_results(self, epsilon_dict, u_data):
        """Affiche tous les résultats dans une grille 2x2"""
        fig = plt.figure(figsize=(16, 14))

        ax1 = plt.subplot(2, 2, 1)
        self._plot_on_axis(ax1, self.u, epsilon_dict, "Solution u", "coolwarm")

        ax2 = plt.subplot(2, 2, 2)
        if self.w is not None:
            self._plot_on_axis(ax2, self.w, epsilon_dict, "Sensibilité w = ∂u/∂ε", "viridis")

        ax3 = plt.subplot(2, 2, 3)
        if self.p is not None:
            self._plot_on_axis(ax3, self.p, epsilon_dict, "État adjoint p", "plasma")

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
