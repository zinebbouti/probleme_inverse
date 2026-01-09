from graph_creation import * 
from physics import *
from test_case import * 
import numpy as np
import matplotlib.pyplot as plt


    
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


def example_validation_complete():
    """VALIDATION COMPLÈTE avec visualisations"""
    print("\n" + "="*80)
    print("EXEMPLE: VALIDATION GRADIENT dJ/dα - 3 MÉTHODES AVEC VISUALISATIONS")
    print("="*80)
    
    # 1. Créer le graphe
    graph = create_2d_graph_example()
    graph.build_dof_map()
    
    # 2. Visualiser la structure du graphe
    print("\n>>> Affichage de la structure du graphe...")
    graph.plot_graph(title="Structure du graphe métrique")
    
    # 3. Définir les sources
    epsilon_dict = {
        0: 1.0,  # Source sur arête 0 à position 1.0
        2: 0.7,  # Source sur arête 2 à position 0.7
    }
    source_intensity = 10.0
    
    # 4. Créer le solveur
    solver = SourceLocalization(graph)
    
    # 5. Générer données "observées"
    u_exact = solver.solve_direct(epsilon_dict, source_intensity)
    noise_level = 0.01
    u_data = u_exact + noise_level * np.random.randn(len(u_exact))
    
    # 6. VALIDATION COMPLÈTE
    results = solver.validate_gradient_three_methods(
        epsilon_dict=epsilon_dict,
        u_data=u_data,
        source_intensity=source_intensity,
        delta=None
    )
    # 7. VISUALISATIONS
    print("\n>>> Affichage de la solution u...")
    solver.plot_solution_on_graph(epsilon_dict, title="Solution u du problème direct")
    
    print("\n>>> Affichage de la sensibilité w = ∂u/∂α...")
    solver.plot_sensitivity_on_graph(epsilon_dict, title="Sensibilité w = ∂u/∂α")
    
    print("\n>>> Affichage de l'état adjoint p...")
    solver.plot_adjoint_on_graph(epsilon_dict, title="État adjoint p")
    
    print("\n>>> Affichage de tous les résultats ensemble...")
    solver.plot_all_results(epsilon_dict, u_data)
    
    return solver, graph, results



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