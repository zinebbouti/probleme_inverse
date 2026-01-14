# from graph_creation import * 
# from physics import *
# from test_case import * 
# import numpy as np
# import matplotlib.pyplot as plt


    
# def plot_solution_all_edges(graph, u_num):
#     for edge in graph.edges:
#         edge_id = edge['id']
#         L = edge['length']
#         n = edge['n']
#         h = L / (n + 1)

#         dofs = graph.get_edge_dofs(edge_id)

#         x = np.array([(i + 1) * h for i in range(n)])
#         u_num_edge = np.array([u_num[dof] for dof in dofs])

#         # Solution exacte MMS degré > 2
#         u_exact = np.array([
#             graph.solver.exact_solution_mms(xi, edge)
#             if hasattr(graph, "solver") else np.sin(np.pi * xi / L)
#             for xi in x
#         ])

#         plt.figure(figsize=(6, 4))
#         plt.plot(x, u_exact, 'r-', lw=2, label="Solution exacte")
#         plt.plot(x, u_num_edge, 'bo', ms=4, label="Solution numérique")

#         plt.xlabel("x")
#         plt.ylabel("u(x)")
#         plt.title(f"Comparaison sur l’arête {edge_id}")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()


# def create_2d_graph_example():
#     """Crée un graphe 2D simple"""
#     graph = MetricGraph()
    
#     positions = {
#         'v0': (0, 2),
#         'v1': (2, 2),
#         'v2': (2, 0),
#         'v3': (0, 0),
#         'v4': (1, 1),
#     }
    
#     for v_id, pos in positions.items():
#         graph.set_vertex_position(v_id, pos[0], pos[1])
    
#     graph.add_edge(0, 'v3', 'v2', length=2.0, a_coef=1.0, n_points=30)
#     graph.add_edge(1, 'v3', 'v0', length=2.0, a_coef=1.0, n_points=30)
#     graph.add_edge(2, 'v3', 'v4', length=np.sqrt(2), a_coef=1.0, n_points=20)
    
#     graph.set_boundary_vertices(['v0', 'v1', 'v2'])
    
#     return graph


# def example_validation_complete():
#     """
#     Cas test 2D complet :
#     - génération des données
#     - validation du gradient (FD / sensibilité / adjoint)
#     - inversion de l'intensité α par gradient conjugué
#     - visualisations finales
#     """

#     print("\n" + "="*80)
#     print("CAS TEST 2D – PROBLÈME DIRECT + ADJOINT + INVERSION (α)")
#     print("="*80)

#     # ============================================================
#     # 1. Création du graphe
#     # ============================================================
#     graph = create_2d_graph_example()
#     graph.build_dof_map()

#     print(f"\nNombre total de DDL: {graph.n_dof}")

#     print("\n>>> Structure du graphe métrique")
#     graph.plot_graph(title="Graphe métrique – cas test 2D")

#     # ============================================================
#     # 2. Définition de la vraie source
#     # ============================================================
#     epsilon_dict = {
#         0: 1.0,   # arête 0
#         2: 0.7,   # arête 2
#     }
#     alpha_exact = 10.0

#     # ============================================================
#     # 3. Création du solveur
#     # ============================================================
#     solver = SourceLocalization(graph)

#     # ============================================================
#     # 4. Génération des données observées
#     # ============================================================
#     print("\n>>> Génération des données observées")

#     u_exact = solver.solve_direct(epsilon_dict, alpha_exact)

#     noise_level = 0.01
#     np.random.seed(0)  # reproductibilité
#     u_data = u_exact + noise_level * np.random.randn(len(u_exact))

#     # ============================================================
#     # 5. VALIDATION DU GRADIENT
#     # ============================================================
#     print("\n" + "="*80)
#     print("VALIDATION DU GRADIENT dJ/dα")
#     print("="*80)

#     results = solver.validate_gradient_three_methods(
#         epsilon_dict=epsilon_dict,
#         u_data=u_data,
#         source_intensity=alpha_exact,
#         delta=None
#     )

#     if not results["validation_passed"]:
#         print("\n⚠ ATTENTION : validation du gradient non parfaite")
#     else:
#         print("\n✓ Validation du gradient réussie")

#     # ============================================================
#     # 6. INVERSION PAR GRADIENT CONJUGUÉ
#     # ============================================================
#     print("\n" + "="*80)
#     print("INVERSION DE L'INTENSITÉ DE LA SOURCE (α)")
#     print("="*80)

#     alpha_initial = 2.0

#     alpha_identified = solver.conjugate_gradient_alpha(
#         epsilon_dict=epsilon_dict,
#         u_data=u_data,
#         alpha_init=alpha_initial,
#         max_iter=30,
#         tol=1e-10,
#         verbose=True
#     )

#     print("\n" + "="*60)
#     print("RÉSULTAT FINAL")
#     print("="*60)
#     print(f"Alpha exact      : {alpha_exact:.6f}")
#     print(f"Alpha initial    : {alpha_initial:.6f}")
#     print(f"Alpha identifié  : {alpha_identified:.6f}")
#     print(f"Erreur relative  : "
#           f"{abs(alpha_identified - alpha_exact) / alpha_exact:.3e}")

#     # ============================================================
#     # 7. VISUALISATIONS FINALES (solution identifiée)
#     # ============================================================
#     print("\n>>> Visualisations finales")

#     solver.solve_direct(epsilon_dict, alpha_identified)

#     solver.plot_solution_on_graph(
#         epsilon_dict,
#         title="Solution u après inversion (α identifié)"
#     )

#     solver.solve_sensitivity_alpha(epsilon_dict, alpha_identified)
#     solver.plot_sensitivity_on_graph(
#         epsilon_dict,
#         title="Sensibilité w = ∂u/∂α (α identifié)"
#     )

#     solver.solve_adjoint(epsilon_dict, u_data, alpha_identified)
#     solver.plot_adjoint_on_graph(
#         epsilon_dict,
#         title="État adjoint p (α identifié)"
#     )

#     solver.plot_all_results(epsilon_dict, u_data)

#     return solver, graph, results




# def example_adjoint_validation():
#     """
#     Validation de l'équation adjointe par comparaison avec différences finies
    
#     EXPLICATION DE LA MÉTHODE ADJOINTE:
#     ====================================
    
#     1. PROBLÈME D'OPTIMISATION:
#        min J(ε) = 1/2 ∫(u(ε) - u_data)² dx
       
#        où u(ε) est solution de: A·u = g(ε)
    
#     2. CALCUL DU GRADIENT:
       
#        a) MÉTHODE NAÏVE (différences finies):
#           Pour N paramètres → N+1 résolutions du problème direct
#           dJ/dε_i ≈ [J(ε + δe_i) - J(ε)] / δ
#           COÛT: O(N) résolutions
       
#        b) MÉTHODE ADJOINTE (smart):
#           - Résoudre une fois: A·u = g(ε)
#           - Résoudre une fois: A^T·p = -∂J/∂u
#           - Calculer: dJ/dε = -p^T · ∂g/∂ε
#           COÛT: O(1) résolutions (2 au total, indépendant de N!)
    
#     3. POURQUOI ÇA MARCHE?
       
#        Par la règle de dérivation en chaîne:
#        dJ/dε = (∂J/∂u)^T · (du/dε)
       
#        Or du/dε satisfait: A·(du/dε) = ∂g/∂ε
       
#        En introduisant l'adjoint p tel que: A^T·p = -∂J/∂u
#        On obtient par produit scalaire:
#        dJ/dε = -p^T · ∂g/∂ε
       
#        → Pas besoin de calculer du/dε explicitement !
    
#     4. INTERPRÉTATION PHYSIQUE:
#        - u : état direct (propagation de la source vers les mesures)
#        - p : état adjoint (rétro-propagation de l'erreur)
#        - p indique comment chaque point influence la fonctionnelle J
#     """
#     print("\n" + "="*70)
#     print("VALIDATION DE L'ÉQUATION ADJOINTE")
#     print("="*70)
#     print("\nCette validation compare deux méthodes de calcul du gradient:")
#     print("  • Méthode adjointe (efficace): 2 résolutions")
#     print("  • Différences finies (coûteuse): N+1 résolutions")
#     print("="*70)
    
#     # Créer le graphe
#     graph = create_2d_graph_example()
#     graph.build_dof_map()
    
#     # Positions des sources
#     epsilon_dict = {0: 1.5, 1: 0.7}
    
#     # Paramètres
#     source_intensity = 10.0
#     varpi = 0.0  # Pas de terme de flux pour simplifier
    
#     print("\n1. Résolution du problème direct...")
#     solver = SourceLocalization(graph)
#     u = solver.solve_direct(epsilon_dict, source_intensity)
    
#     # Créer des données synthétiques (légèrement bruitées)
#     u_data = u + 0.01 * np.random.randn(len(u))
#     flux_data = {v: 0.0 for v in graph.boundary_vertices}
    
#     # Calculer la fonctionnelle
#     J = solver.compute_cost_functional(u, u_data, flux_data, varpi)
#     print(f"Fonctionnelle J = {J:.6e}")
    
#     print("\n2. Résolution de l'équation adjointe...")
#     p = solver.solve_adjoint(u_data, flux_data, varpi, epsilon_dict, source_intensity)
    
#     # Visualiser l'état adjoint
#     print("\n3. Visualisation de l'état adjoint...")
#     solver.plot_adjoint_on_graph(epsilon_dict, title="État adjoint p(x)")
    
#     print("\n4. Calcul des gradients via méthode adjointe...")
#     gradients_adjoint = solver.compute_gradient_all_edges(epsilon_dict, source_intensity)
    
#     print("\n5. Calcul des gradients par différences finies (validation)...")
#     gradients_fd = {}
#     for edge_id in epsilon_dict.keys():
#         grad_fd = solver.compute_gradient_finite_diff(
#             epsilon_dict, edge_id, u_data, flux_data, source_intensity, varpi
#         )
#         gradients_fd[edge_id] = grad_fd
    
#     # Comparaison
#     print("\n" + "="*70)
#     print("COMPARAISON DES GRADIENTS")
#     print("="*70)
#     print("\nInterprétation:")
#     print("  • Gradient positif → déplacer ε vers la droite AUGMENTE J")
#     print("  • Gradient négatif → déplacer ε vers la droite DIMINUE J")
#     print("  • |Gradient| grand → forte sensibilité de J à ε")
#     print("-"*70)
#     print(f"{'Arête':<10} {'Adjoint':<20} {'Diff. Finies':<20} {'Erreur Rel.':<15}")
#     print("-"*70)
    
#     for edge_id in epsilon_dict.keys():
#         grad_adj = gradients_adjoint[edge_id]
#         grad_fd = gradients_fd[edge_id]
        
#         if abs(grad_fd) > 1e-10:
#             err_rel = abs(grad_adj - grad_fd) / abs(grad_fd)
#         else:
#             err_rel = abs(grad_adj - grad_fd)
        
#         print(f"{edge_id:<10} {grad_adj:<20.8e} {grad_fd:<20.8e} {err_rel:<15.2e}")
    
#     print("="*70)
    
#     # Vérification globale
#     all_errors = [abs(gradients_adjoint[e] - gradients_fd[e]) / abs(gradients_fd[e]) 
#                   for e in epsilon_dict.keys() if abs(gradients_fd[e]) > 1e-10]
    
#     max_error = max(all_errors) if all_errors else 0.0
    
#     print(f"\nErreur relative maximale: {max_error:.2e}")
    
#     # Seuil de validation plus réaliste pour les différences finies
#     if max_error < 1e-3:
#         print("✓ VALIDATION RÉUSSIE! Les gradients adjoints sont corrects.")
#         print("  (Erreur < 0.1% : excellente précision)")
#     elif max_error < 1e-2:
#         print("✓ VALIDATION ACCEPTABLE. Les gradients adjoints sont fiables.")
#         print("  (Erreur < 1% : bonne précision)")
#     else:
#         print("⚠ Attention: erreurs importantes détectées.")
#         print("  Vérifier l'implémentation ou réduire delta.")
    
#     print("="*70)

# def exact_solution_mms(x, edge):
#     L = edge['length']
#     eid = edge['id']

#     C = 1.0
#     A1 = 0.0
#     # Kirchhoff (2 arêtes, même a et même L) -> A2 = 2C/L^2 - A1
#     if eid == 0:
#         A = A1
#     elif eid == 1:
#         A = 2.0 * C / L**2 - A1
#     else:
#         A = 0.0

#     B = 1.0  # amplitude du terme degré 4 (tu peux changer)

#     return C * (1.0 - x / L) + A * x * (L - x) + B * (x**2) * ((L - x)**2)
    
# def compute_errors_mms(graph, u_num):
#     """
#     Calcule les erreurs L1, L2, Linf sur tout le graphe métrique
#     (uniquement sur les DDL d'arêtes)
#     """
#     L1 = 0.0
#     L2 = 0.0
#     Linf = 0.0

#     for edge in graph.edges:
#         edge_id = edge['id']
#         L = edge['length']
#         n = edge['n']
#         h = L / (n + 1)

#         dofs = graph.get_edge_dofs(edge_id)

#         for i, dof in enumerate(dofs):
#             x = (i + 1) * h
#             u_exact = exact_solution_mms(x, edge)
#             diff = abs(u_num[dof] - u_exact)

#             L1 += h * diff
#             L2 += h * diff**2
#             Linf = max(Linf, diff)

#     return L1, np.sqrt(L2), Linf

# def validation_DF():

#     print("\n" + "="*70)
#     print("VALIDATION MMS - CONVERGENCE")
#     print("="*70)

#     Ns = [10, 20, 40, 80, 160]

#     errors_L1 = []
#     errors_L2 = []
#     errors_Linf = []
#     hs = []

#     for N in Ns:
#         graph = MetricGraph()
#         graph.add_edge(0, 'v1', 'v2', length=1.0, a_coef=1.0, n_points=N)
#         graph.add_edge(1, 'v1', 'v3', length=1.0, a_coef=1.0, n_points=N)

#         graph.set_vertex_position('v1', 0, 0)
#         graph.set_vertex_position('v2', 1, 0)
#         graph.set_vertex_position('v3', 0, 1)
#         graph.set_boundary_vertices(['v2', 'v3'])
#         graph.build_dof_map()

#         solver = validation(graph)

#         u_num = solver.solve_direct_val()

#         L1, L2, Linf = compute_errors_mms(graph, u_num)

#         h = 1.0 / (N + 1)

#         hs.append(h)
#         errors_L1.append(L1)
#         errors_L2.append(L2)
#         errors_Linf.append(Linf)

#         print(f"N={N:4d} | L1={L1:.3e} | L2={L2:.3e} | Linf={Linf:.3e}")

#     # ============================
#     # Calcul des ordres observés
#     # ============================
#     def compute_orders(errs):
#         return [
#             np.log(errs[i-1] / errs[i]) / np.log(2.0)
#             for i in range(1, len(errs))
#         ]

#     orders_L1 = compute_orders(errors_L1)
#     orders_L2 = compute_orders(errors_L2)
#     orders_Linf = compute_orders(errors_Linf)

#     print("\nOrdres observés (entre deux raffinements) :")
#     for i in range(len(orders_L1)):
#         print(f"N={Ns[i]}→{Ns[i+1]} | "
#               f"L1={orders_L1[i]:.2f}, "
#               f"L2={orders_L2[i]:.2f}, "
#               f"Linf={orders_Linf[i]:.2f}")

#     # ============================
#     # Courbe de convergence
#     # ============================
#     plt.figure(figsize=(7, 5))
#     plt.loglog(hs, errors_L2, 'o-', label=r"$\|e\|_{L^2}$")
#     plt.loglog(
#         hs,
#         errors_L2[0] * (np.array(hs) / hs[0])**2,
#         '--',
#         label="Référence ordre 2"
#     )

#     plt.gca().invert_xaxis()
#     plt.xlabel("h")
#     plt.ylabel("Erreur")
#     plt.title("Convergence MMS – norme $L^2$")
#     plt.legend()
#     plt.grid(True, which="both")
#     plt.tight_layout()
#     plt.show()

#     plot_solution_all_edges(graph, u_num)


##########################################################################################

##########################################################################################
##########################################################################################

# from graph_creation import MetricGraph
# from physics import SourceLocalizationEpsilon
# import numpy as np
# import matplotlib.pyplot as plt


# # ============================================================================
# # FONCTIONS DE CRÉATION DE GRAPHES
# # ============================================================================

# def create_simple_Y_graph():
#     """Crée un graphe en Y simple pour les tests"""
#     graph = MetricGraph()
    
#     positions = {
#         'v0': (0, 0),    # Centre
#         'v1': (1, 0),    # Droite
#         'v2': (0, 1),    # Haut
#         'v3': (-1, 0),   # Gauche
#     }
    
#     for v_id, pos in positions.items():
#         graph.set_vertex_position(v_id, pos[0], pos[1])
    
#     graph.add_edge(0, 'v0', 'v1', length=1.0, a_coef=1.0, n_points=40)
#     graph.add_edge(1, 'v0', 'v2', length=1.0, a_coef=1.0, n_points=40)
#     graph.add_edge(2, 'v0', 'v3', length=1.0, a_coef=1.0, n_points=40)
    
#     graph.set_boundary_vertices(['v1', 'v2', 'v3'])
    
#     return graph


# def create_2d_graph_example():
#     """Crée un graphe 2D plus complexe"""
#     graph = MetricGraph()
    
#     positions = {
#         'v0': (0, 2),
#         'v1': (2, 2),
#         'v2': (2, 0),
#         'v3': (0, 0),
#         'v4': (1, 1),
#     }
    
#     for v_id, pos in positions.items():
#         graph.set_vertex_position(v_id, pos[0], pos[1])
    
#     graph.add_edge(0, 'v3', 'v2', length=2.0, a_coef=1.0, n_points=50)
#     graph.add_edge(1, 'v3', 'v0', length=2.0, a_coef=1.0, n_points=50)
#     graph.add_edge(2, 'v3', 'v4', length=np.sqrt(2), a_coef=1.0, n_points=30)
    
#     graph.set_boundary_vertices(['v0', 'v1', 'v2'])
    
#     return graph


# # ============================================================================
# # TEST 1 : VALIDATION DES GRADIENTS dJ/dε (3 MÉTHODES)
# # ============================================================================

# def test_gradient_validation():
#     """
#     Validation complète des 3 méthodes de calcul de dJ/dε
    
#     Vérifie que :
#     1. Les 3 méthodes donnent le même gradient
#     2. Les valeurs de J sont cohérentes
#     3. L'erreur entre sensibilité et adjointe est < 1e-14
#     """
#     print("\n" + "="*80)
#     print("TEST : VALIDATION DES GRADIENTS dJ/dε")
#     print("="*80)
    
#     # 1. Création du graphe
#     graph = create_simple_Y_graph()
#     graph.build_dof_map()
    
#     # 2. Configuration
#     epsilon_true = 0.6
#     edge_id_source = 0
    
#     print(f"\nConfiguration :")
#     print(f"  Position source : ε = {epsilon_true}")
#     print(f"  Arête source    : {edge_id_source}")
#     print(f"  Intensité fixée : α = 1.0\n")
    
#     # 3. Génération données synthétiques
#     solver = SourceLocalizationEpsilon(graph)
#     epsilon_dict_true = {edge_id_source: epsilon_true}
    
#     u_true = solver.solve_direct(epsilon_dict_true)
#     noise_level = 0.01
#     u_data = u_true + noise_level * np.random.randn(len(u_true))
    
#     # 4. Position de test (différente de la vraie)
#     epsilon_test = 0.35
#     epsilon_dict_test = {edge_id_source: epsilon_test}
    
#     print(f"Test du gradient à ε = {epsilon_test}\n")
    
#     # 5. Appel de la validation (dans physics.py)
#     results = solver.validate_gradient_three_methods(
#         epsilon_dict_test, u_data, edge_id_source
#     )
    
#     # 6. Résumé final
#     print("\n" + "="*80)
#     print("RÉSUMÉ DE LA VALIDATION")
#     print("="*80)
#     print(f"\nCoût J(ε) = {results['J']:.6e}")
#     print(f"\nGradients calculés :")
#     print(f"  • Différences finies : {results['grad_fd']:.12e}")
#     print(f"  • Sensibilité directe: {results['grad_sensitivity']:.12e}")
#     print(f"  • Méthode adjointe   : {results['grad_adjoint']:.12e}")
    
#     print(f"\nErreurs relatives :")
#     print(f"  • Sensibilité vs FD  : {results['error_sens_vs_fd']:.3e}")
#     print(f"  • Adjointe vs FD     : {results['error_adj_vs_fd']:.3e}")
#     print(f"  • Sensibilité vs Adj : {results['error_sens_vs_adj']:.3e}")
    
#     # Critère de validation
#     if results['validation_passed']:
#         print(f"\n{'✓'*40}")
#         print("✓✓ VALIDATION RÉUSSIE !")
#         print(f"{'✓'*40}")
#     else:
#         print(f"\n{'⚠'*40}")
#         print("⚠ Validation partielle - vérifier les erreurs")
#         print(f"{'⚠'*40}")
    
#     print("="*80 + "\n")
    
#     return results


# # ============================================================================
# # TEST 2 : OPTIMISATION PAR MÉTHODE ADJOINTE
# # ============================================================================

# def test_optimization_adjoint():
#     """
#     Optimisation de ε par gradient conjugué (méthode adjointe)
    
#     Objectif : Retrouver la position de la source à partir de mesures
#     """
#     print("\n" + "="*80)
#     print("TEST : OPTIMISATION PAR MÉTHODE ADJOINTE")
#     print("="*80)
    
#     # 1. Graphe
#     graph = create_simple_Y_graph()
#     graph.build_dof_map()
    
#     # 2. Génération données
#     epsilon_true = 0.65
#     edge_id_source = 0
    
#     print(f"\nCONFIGURATION :")
#     print(f"  Position vraie : ε_true = {epsilon_true}")
#     print(f"  Intensité fixée: α = 1.0")
    
#     solver = SourceLocalizationEpsilon(graph)
#     epsilon_dict_true = {edge_id_source: epsilon_true}
    
#     u_true = solver.solve_direct(epsilon_dict_true)
#     u_data = u_true + 0.02 * np.random.randn(len(u_true))
    
#     # 3. Optimisation
#     epsilon_init = 0.2
#     print(f"  Position initiale : ε_init = {epsilon_init}\n")
    
#     result = solver.optimize_cg_adjoint(
#         epsilon_init, edge_id_source, u_data,
#         max_iter=30, tol=1e-8
#     )
    
#     # 4. Résultats
#     print(f"\n{'='*80}")
#     print(f"RÉSULTATS FINAUX :")
#     print(f"{'='*80}")
#     print(f"Position vraie      : ε_true = {epsilon_true}")
#     print(f"Position trouvée    : ε*      = {result.x[0]:.6f}")
#     print(f"Erreur absolue      : |ε* - ε_true| = {abs(result.x[0] - epsilon_true):.6e}")
#     print(f"Erreur relative     : {abs(result.x[0] - epsilon_true)/epsilon_true * 100:.3f}%")
#     print(f"Coût final J(ε*)    : {result.fun:.6e}")
#     print(f"Convergence         : {'OUI' if result.success else 'NON'}")
#     print(f"Nombre d'itérations : {result.nit}")
#     print(f"{'='*80}\n")
    
#     return result


# # ============================================================================
# # TEST 3 : COMPARAISON ADJOINTE VS SENSIBILITÉ
# # ============================================================================

# def test_comparison_adjoint_vs_sensitivity():
#     """
#     Compare les deux méthodes d'optimisation
    
#     Vérifie que les deux approches convergent vers la même solution
#     """
#     print("\n" + "="*80)
#     print("TEST : COMPARAISON ADJOINTE VS SENSIBILITÉ")
#     print("="*80)
    
#     # 1. Graphe
#     graph = create_2d_graph_example()
#     graph.build_dof_map()
    
#     # 2. Données
#     epsilon_true = 1.2
#     edge_id_source = 0
    
#     print(f"\nCONFIGURATION :")
#     print(f"  Position vraie : ε_true = {epsilon_true}")
#     print(f"  Arête source   : {edge_id_source}")
    
#     solver_adj = SourceLocalizationEpsilon(graph)
#     epsilon_dict_true = {edge_id_source: epsilon_true}
    
#     u_true = solver_adj.solve_direct(epsilon_dict_true)
#     u_data = u_true + 0.01 * np.random.randn(len(u_true))
    
#     # 3. Optimisation adjointe
#     print("\n" + "-"*80)
#     print("OPTIMISATION 1/2 : MÉTHODE ADJOINTE")
#     print("-"*80)
    
#     epsilon_init = 0.5
#     result_adj = solver_adj.optimize_cg_adjoint(
#         epsilon_init, edge_id_source, u_data, max_iter=25
#     )
#     history_adj = solver_adj.history.copy()
    
#     # 4. Optimisation sensibilité
#     print("\n" + "-"*80)
#     print("OPTIMISATION 2/2 : MÉTHODE SENSIBILITÉ")
#     print("-"*80)
    
#     solver_sens = SourceLocalizationEpsilon(graph)
#     result_sens = solver_sens.optimize_cg_sensitivity(
#         epsilon_init, edge_id_source, u_data, max_iter=25
#     )
#     history_sens = solver_sens.history.copy()
    
#     # 5. Comparaison
#     print(f"\n{'='*80}")
#     print(f"COMPARAISON DES RÉSULTATS")
#     print(f"{'='*80}")
#     print(f"\n{'Méthode':<20} {'ε optimal':<15} {'J final':<15} {'Iter':<8} {'Erreur |ε-ε_true|'}")
#     print("-"*80)
#     print(f"{'Adjointe':<20} {result_adj.x[0]:<15.6f} {result_adj.fun:<15.6e} {result_adj.nit:<8} {abs(result_adj.x[0]-epsilon_true):.6e}")
#     print(f"{'Sensibilité':<20} {result_sens.x[0]:<15.6f} {result_sens.fun:<15.6e} {result_sens.nit:<8} {abs(result_sens.x[0]-epsilon_true):.6e}")
#     print(f"{'Vraie valeur':<20} {epsilon_true:<15.6f} {'':<15} {'':<8}")
#     print("="*80)
    
#     # 6. Analyse de la différence
#     diff_epsilon = abs(result_adj.x[0] - result_sens.x[0])
#     diff_J = abs(result_adj.fun - result_sens.fun)
    
#     print(f"\nDIFFÉRENCES ENTRE LES MÉTHODES :")
#     print(f"  • Position : |ε_adj - ε_sens| = {diff_epsilon:.6e}")
#     print(f"  • Coût     : |J_adj - J_sens| = {diff_J:.6e}")
    
#     if diff_epsilon < 1e-6 and diff_J < 1e-10:
#         print(f"\n✓✓ Les deux méthodes convergent vers la MÊME solution!")
#     else:
#         print(f"\n⚠ Différences détectées - peut être dû à:")
#         print(f"    - Critère d'arrêt différent")
#         print(f"    - Nombre d'itérations insuffisant")
    
#     print("="*80 + "\n")
    
#     # 7. Graphiques
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
#     axes[0].plot(history_adj['epsilon'], 'o-', label='Adjointe', linewidth=2, markersize=6)
#     axes[0].plot(history_sens['epsilon'], 's--', label='Sensibilité', linewidth=2, markersize=6)
#     axes[0].axhline(epsilon_true, color='red', linestyle=':', linewidth=2, label='ε_true')
#     axes[0].set_xlabel('Itération', fontsize=11)
#     axes[0].set_ylabel('Position ε', fontsize=11)
#     axes[0].set_title('Convergence de ε', fontweight='bold')
#     axes[0].legend(fontsize=10)
#     axes[0].grid(True, alpha=0.3)
    
#     axes[1].semilogy(history_adj['J'], 'o-', label='Adjointe', linewidth=2, markersize=6)
#     axes[1].semilogy(history_sens['J'], 's--', label='Sensibilité', linewidth=2, markersize=6)
#     axes[1].set_xlabel('Itération', fontsize=11)
#     axes[1].set_ylabel('J(ε)', fontsize=11)
#     axes[1].set_title('Décroissance du coût', fontweight='bold')
#     axes[1].legend(fontsize=10)
#     axes[1].grid(True, alpha=0.3)
    
#     axes[2].semilogy(history_adj['grad_norm'], 'o-', label='Adjointe', linewidth=2, markersize=6)
#     axes[2].semilogy(history_sens['grad_norm'], 's--', label='Sensibilité', linewidth=2, markersize=6)
#     axes[2].set_xlabel('Itération', fontsize=11)
#     axes[2].set_ylabel('|∇J|', fontsize=11)
#     axes[2].set_title('Norme du gradient', fontweight='bold')
#     axes[2].legend(fontsize=10)
#     axes[2].grid(True, alpha=0.3)
    
#     plt.suptitle('Comparaison Adjointe vs Sensibilité', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
    
#     return result_adj, result_sens


# # ============================================================================
# # TEST 4 : SENSIBILITÉ AU BRUIT
# # ============================================================================

# def test_noise_sensitivity():
#     """
#     Analyse l'impact du bruit sur la reconstruction
#     """
#     print("\n" + "="*80)
#     print("TEST : SENSIBILITÉ AU BRUIT")
#     print("="*80)
    
#     graph = create_simple_Y_graph()
#     graph.build_dof_map()
    
#     epsilon_true = 0.7
#     edge_id_source = 0
#     epsilon_init = 0.2
    
#     solver = SourceLocalizationEpsilon(graph)
#     epsilon_dict_true = {edge_id_source: epsilon_true}
#     u_true = solver.solve_direct(epsilon_dict_true)
    
#     noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
#     results = []
    
#     print(f"\nPosition vraie : ε_true = {epsilon_true}")
#     print(f"Position initiale : ε_init = {epsilon_init}\n")
#     print("-"*80)
#     print(f"{'Bruit %':<12} {'ε optimal':<15} {'Erreur':<15} {'J final':<15}")
#     print("-"*80)
    
#     for noise in noise_levels:
#         u_data = u_true + noise * np.random.randn(len(u_true))
        
#         solver_test = SourceLocalizationEpsilon(graph)
#         result = solver_test.optimize_cg_adjoint(
#             epsilon_init, edge_id_source, u_data, max_iter=30, tol=1e-8
#         )
        
#         error = abs(result.x[0] - epsilon_true)
#         results.append((noise, result.x[0], error, result.fun))
        
#         print(f"{noise*100:<12.1f} {result.x[0]:<15.6f} {error:<15.6e} {result.fun:<15.6e}")
    
#     print("-"*80 + "\n")
    
#     # Graphique
#     noises = [r[0]*100 for r in results]
#     errors = [r[2] for r in results]
    
#     plt.figure(figsize=(8, 5))
#     plt.semilogy(noises, errors, 'o-', linewidth=2, markersize=8)
#     plt.xlabel('Niveau de bruit %', fontsize=12)
#     plt.ylabel('Erreur |ε* - ε_true|', fontsize=12)
#     plt.title('Impact du bruit sur la reconstruction', fontsize=14, fontweight='bold')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
    
#     return results


# # ============================================================================
# # VALIDATION MMS 1D (fonction manquante)
# # ============================================================================

# def validation_DF():
#     """
#     Validation par méthode des solutions manufacturées (MMS)
#     Test de convergence en h²
#     """
#     print("\n" + "="*80)
#     print("VALIDATION MMS 1D - CONVERGENCE h²")
#     print("="*80)
#     print("\nCette fonction n'est pas encore implémentée.")
#     print("Elle nécessite une classe de validation spécifique.")
#     print("Pour l'instant, utilisez les modes 2-6.\n")
#     print("="*80 + "\n")


# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == "__main__":
#     import sys
    
#     print("\n" + "="*80)
#     print("TESTS DE LOCALISATION DE SOURCE")
#     print("="*80)
#     print("\nTests disponibles :")
#     print("  1. Validation gradients (3 méthodes)")
#     print("  2. Optimisation adjointe")
#     print("  3. Comparaison Adjointe vs Sensibilité")
#     print("  4. Sensibilité au bruit")
#     print("  0. Quitter")
#     print("="*80)
    
#     try:
#         choix = int(input("\nVotre choix: "))
#     except:
#         choix = 1
    
#     if choix == 1:
#         test_gradient_validation()
#     elif choix == 2:
#         test_optimization_adjoint()
#     elif choix == 3:
#         test_comparison_adjoint_vs_sensitivity()
#     elif choix == 4:
#         test_noise_sensitivity()
#     elif choix == 0:
#         sys.exit(0)
#     else:
#         print("Choix invalide, lancement du test 1")
#         test_gradient_validation()


from graph_creation import * 
from physics import *
from test_case import * 
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GRAPHE 2D SIMPLE
# ============================================================

def create_simple_2d_graph():
    """
    Graphe 2D simple en forme de Y
    """
    graph = MetricGraph()

    positions = {
        'v0': (0.0, 0.0),
        'v1': (2.0, 0.0),
        'v2': (1.0, 1.5),
    }

    for v_id, (x, y) in positions.items():
        graph.set_vertex_position(v_id, x, y)

    graph.add_edge(
        0, 'v0', 'v2',
        length=np.sqrt(1.0**2 + 1.5**2),
        a_coef=1.0,
        n_points=40
    )
    graph.add_edge(
        1, 'v1', 'v2',
        length=np.sqrt(1.0**2 + 1.5**2),
        a_coef=1.0,
        n_points=40
    )

    graph.set_boundary_vertices(['v0', 'v1'])
    graph.build_dof_map()

    return graph


# ============================================================
# TEST INVERSE COMPLET
# ============================================================

def test_inverse_source_localization():
    print("\n" + "=" * 80)
    print("TEST PROBLÈME INVERSE – LOCALISATION DE SOURCE (ε)")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Création du graphe
    # ------------------------------------------------------------
    graph = create_simple_2d_graph()
    graph.plot_graph(title="Graphe 2D – test inverse")

    solver = SourceLocalization(graph)

    # ------------------------------------------------------------
    # 2. Vraie source
    # ------------------------------------------------------------
    edge_id = 0
    epsilon_true = 0.9
    source_intensity = 10.0

    epsilon_true_dict = {edge_id: epsilon_true}

    # ------------------------------------------------------------
    # 3. Génération des données observées
    # ------------------------------------------------------------
    print("\n>>> Génération des données observées")

    u_exact = solver.solve_direct(
        epsilon_true_dict,
        source_intensity
    )

    noise_level = 0.01
    np.random.seed(1)
    u_data = u_exact + noise_level * np.random.randn(len(u_exact))

    # ------------------------------------------------------------
    # 4. VALIDATION DES GRADIENTS
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε")
    print("=" * 80)

    epsilon_test = 0.6

    results = solver.validate_gradient_three_methods_epsilon(
        edge_id=edge_id,
        epsilon=epsilon_test,
        u_data=u_data,
        source_intensity=source_intensity,
        alpha_fd=None
    )

    print("\nRésumé gradients :")
    print(f"  dJ/dε (DF)          = {results['grad_fd']:.6e}")
    print(f"  dJ/dε (Sensibilité) = {results['grad_sensitivity']:.6e}")
    print(f"  dJ/dε (Adjoint)     = {results['grad_adjoint']:.6e}")

    # ------------------------------------------------------------
    # 5. OPTIMISATION PAR GRADIENT CONJUGUÉ (ADJOINT)
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – LOCALISATION DE LA SOURCE")
    print("=" * 80)

    epsilon_init = 0.2

    epsilon_identified = solver.conjugate_gradient_epsilon(
        edge_id=edge_id,
        u_data=u_data,
        epsilon_init=epsilon_init,
        source_intensity=source_intensity,
        max_iter=30,
        tol=1e-8,
        verbose=True
    )

    # ------------------------------------------------------------
    # 6. RÉSULTATS FINAUX
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)

    print(f"ε exact       : {epsilon_true:.6f}")
    print(f"ε initial     : {epsilon_init:.6f}")
    print(f"ε identifié   : {epsilon_identified:.6f}")
    print(
        f"Erreur relative : "
        f"{abs(epsilon_identified - epsilon_true)/epsilon_true:.3e}"
    )

    # ------------------------------------------------------------
    # 7. VISUALISATIONS
    # ------------------------------------------------------------
    solver.solve_direct(
        {edge_id: epsilon_identified},
        source_intensity
    )
    solver.solve_sensitivity_epsilon(
        {edge_id: epsilon_identified},
        source_intensity
    )
    solver.solve_adjoint(
        {edge_id: epsilon_identified},
        u_data,
        source_intensity
    )

    solver.plot_all_results(
        {edge_id: epsilon_identified},
        u_data
    )

    return solver, graph, results


def plot_solution_all_edges(graph, u_num):
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

def validation_DF():

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

        solver = validation(graph)

        u_num = solver.solve_direct_val()

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




