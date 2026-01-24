from graph_creation import MetricGraph
from physics import SourceLocalization , validation

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
        n_points=1000
    )
    graph.add_edge(
        1, 'v1', 'v2',
        length=np.sqrt(1.0**2 + 1.5**2),
        a_coef=1.0,
        n_points=1000
    )

    graph.set_boundary_vertices(['v0', 'v1'])
    graph.build_dof_map()

    return graph
def create_simple_Y_graph():
    """Crée un graphe en Y simple pour les tests"""
    graph = MetricGraph()
    
    positions = {
        'v0': (0, 0),    # Centre
        'v1': (1, 0),    # Droite (bord)
        'v2': (0, 1),    # Haut   (bord)
        'v3': (-1, 0),   # Gauche (bord)
    }
    
    for v_id, (x, y) in positions.items():
        graph.set_vertex_position(v_id, x, y)
    
    graph.add_edge(0, 'v0', 'v1', length=1.0, a_coef=1.0, n_points=40)
    graph.add_edge(1, 'v0', 'v2', length=1.0, a_coef=1.0, n_points=40)
    graph.add_edge(2, 'v0', 'v3', length=1.0, a_coef=1.0, n_points=40)

    # Sommets au bord (conditions de Dirichlet)
    graph.set_boundary_vertices(['v1', 'v2', 'v3'])

    # ✅ INDISPENSABLE : construit dof_start/dof_end
    graph.build_dof_map()

    return graph


def create_decoupled_2d_graph():
    """
    Graphe avec deux arêtes totalement découplées
    → validation adjoint parfaite (10^-16)
    """
    graph = MetricGraph()

    # Sommets
    graph.set_vertex_position("v0", 0.0, 0.0)
    graph.set_vertex_position("v1", 1.0, 0.0)

    graph.set_vertex_position("v2", 0.0, 1.0)
    graph.set_vertex_position("v3", 1.0, 1.0)

    # Arête 0
    graph.add_edge(
        0, "v0", "v1",
        length=1.0,
        a_coef=1.0,
        n_points=40
    )

    # Arête 1
    graph.add_edge(
        1, "v2", "v3",
        length=1.0,
        a_coef=1.0,
        n_points=40
    )

    graph.set_boundary_vertices(["v0", "v1", "v2", "v3"])
    graph.build_dof_map()

    return graph


# ============================================================
# TEST INVERSE COMPLET - UNE SOURCE
# ============================================================

def test_inverse_source_localization():
    print("\n" + "=" * 80)
    print("TEST PROBLÈME INVERSE – LOCALISATION DE SOURCE (ε)")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Création du graphe
    # ------------------------------------------------------------
    graph = create_simple_Y_graph()
    graph.plot_graph(title="Graphe 2D – test inverse")

    solver = SourceLocalization(graph)

    # ------------------------------------------------------------
    # 2. Vraie source
    # ------------------------------------------------------------
    edge_id = 0
    epsilon_true = 0.9
    source_intensity = 1

    epsilon_true_dict = {edge_id: epsilon_true}

    # ------------------------------------------------------------
    # 3. Génération des données observées
    # ------------------------------------------------------------
    print("\n>>> Génération des données observées")

    u_exact = solver.solve_direct(
        epsilon_true_dict,
        source_intensity
    )

    noise_level = 0.0
    np.random.seed(1)
    u_data = u_exact.copy()

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
        alpha_fd=1e-7
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

    epsilon_init = 0.5

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


# ============================================================
# VALIDATION MMS
# ============================================================

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
        plt.title(f"Comparaison sur l'arête {edge_id}")
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


# ============================================================
# TEST INVERSE - DEUX SOURCES (VERSION GRAPHE EN Y)
# ============================================================

# def test_inverse_source_localization_two_sources():
#     """
#     Test d'optimisation inverse pour localiser DEUX sources sur graphe en Y.
#     Version avec graphe couplé mais avec paramètres optimisés pour convergence.
#     """
#     print("\n" + "=" * 80)
#     print("TEST PROBLÈME INVERSE – LOCALISATION DE DEUX SOURCES (ε)")
#     print("=" * 80)

#     # ------------------------------------------------------------
#     # 1. Création du graphe EN Y (graphe couplé)
#     # ------------------------------------------------------------
#     graph = create_simple_2d_graph()
#     graph.plot_graph(title="Graphe en Y – test inverse (2 sources)")

#     solver = SourceLocalization(graph)

#     # ------------------------------------------------------------
#     # 2. Vraies sources (2 sources)
#     # ------------------------------------------------------------
#     edge_ids = [0, 1]
#     epsilon_true = {
#         0: 0.4,   # proche de v2 sur arête 0
#         1: 0.7,  # proche de v1 sur arête 1
#     }

#     source_intensity = 1.0

#     print("\n>>> Localisation exacte des sources")
#     for e, eps in epsilon_true.items():
#         print(f"  - Arête {e} : ε = {eps}")

#     # ------------------------------------------------------------
#     # 3. Génération des données observées
#     # ------------------------------------------------------------
#     print("\n>>> Génération des données observées")

#     u_exact = solver.solve_direct(epsilon_true, source_intensity)

#     # Commencer sans bruit pour validation
#     noise_level = 0.0
#     np.random.seed(42)
#     u_data = u_exact.copy()
    
#     if noise_level > 0:
#         u_data = u_exact + noise_level * np.random.randn(len(u_exact))
#         print(f"Bruit ajouté : {noise_level * 100}%")
#     else:
#         print("Aucun bruit ajouté (validation parfaite)")

#     # ------------------------------------------------------------
#     # 4. VALIDATION DES GRADIENTS (AVANT OPTIMISATION)
#     # ------------------------------------------------------------
#     print("\n" + "=" * 80)
#     print("VALIDATION DES GRADIENTS dJ/dε – AVANT OPTIMISATION")
#     print("=" * 80)

#     epsilon_test = {
#         0: 0.4,
#         1: 0.7,
#     }

#     for edge_id, eps in epsilon_test.items():
#         print(f"\n--- Arête {edge_id} | ε = {eps} ---")

#         results = solver.validate_gradient_three_methods_epsilon(
#             edge_id=edge_id,
#             epsilon=eps,
#             u_data=u_data,
#             source_intensity=source_intensity,
#             alpha_fd=1e-7
#         )

#         print(f"  dJ/dε (DF)   = {results['grad_fd']:.12e}")
#         print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.12e}")
#         print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.12e}")
        
#         # Erreur relative Sens vs Adj
#         if abs(results['grad_sensitivity']) > 1e-15:
#             err_rel = abs(results['grad_adjoint'] - results['grad_sensitivity']) / abs(results['grad_sensitivity'])
#             print(f"  Erreur Sens/Adj : {err_rel:.3e}")

#     solver.solve_direct(epsilon_test, source_intensity)
#     J_init = solver.compute_cost_functional(u_data)
#     print(f"\nValeur initiale du coût J = {J_init:.6e}")

#     # # # ------------------------------------------------------------
#     # # # 5. OPTIMISATION PAR GRADIENT CONJUGUÉ (ADJOINT)
#     # # # ------------------------------------------------------------
#     # # print("\n" + "=" * 80)
#     # # print("OPTIMISATION PAR GRADIENT CONJUGUÉ (ADJOINT)")
#     # # print("=" * 80)

#     # epsilon_current = dict(epsilon_test)

#     # # Optimisation alternée
#     # n_outer = 6
#     # for k in range(n_outer):
#     #     print(f"\n>>> Itération externe {k+1}/{n_outer}")

#     #     for edge_id in edge_ids:
#     #         print(f"\n--- Optimisation de ε sur l'arête {edge_id} ---")

#     #         eps_new = solver.conjugate_gradient_epsilon(
#     #             edge_id=edge_id,
#     #             u_data=u_data,
#     #             epsilon_init=epsilon_current[edge_id],
#     #             source_intensity=source_intensity,
#     #             max_iter=40,
#     #             tol=1e-10,
#     #             verbose=(k == 0)  # Verbose seulement 1ère itération
#     #         )

#     #         epsilon_current[edge_id] = eps_new

#     # # # ------------------------------------------------------------
#     # # # 6. VALIDATION DES GRADIENTS (APRÈS OPTIMISATION)
#     # # # ------------------------------------------------------------
#     # # print("\n" + "=" * 80)
#     # # print("VALIDATION DES GRADIENTS dJ/dε – APRÈS OPTIMISATION")
#     # # print("=" * 80)

#     # for edge_id, eps in epsilon_current.items():
#     #     print(f"\n--- Arête {edge_id} | ε = {eps:.8f} ---")

#     #     results = solver.validate_gradient_three_methods_epsilon(
#     #         edge_id=edge_id,
#     #         epsilon=eps,
#     #         u_data=u_data,
#     #         source_intensity=source_intensity,
#     #         alpha_fd=1e-7
#     #     )

#     #     print(f"  dJ/dε (DF)   = {results['grad_fd']:.12e}")
#     #     print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.12e}")
#     #     print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.12e}")
        
#     #     if abs(results['grad_sensitivity']) > 1e-15:
#     #         err_rel = abs(results['grad_adjoint'] - results['grad_sensitivity']) / abs(results['grad_sensitivity'])
#     #         print(f"  Erreur Sens/Adj : {err_rel:.3e}")

#     # # solver.solve_direct(epsilon_current, source_intensity)
#     # # J_opt = solver.compute_cost_functional(u_data)

#     # # ------------------------------------------------------------
#     # # 7. RÉSULTATS FINAUX
#     # # ------------------------------------------------------------
#     # print("\n" + "=" * 80)
#     # print("RÉSULTATS FINAUX")
#     # print("=" * 80)

#     # for e in epsilon_true.keys():
#     #     err_abs = abs(epsilon_current[e] - epsilon_true[e])
#     #     err_rel = err_abs / abs(epsilon_true[e])
#     #     print(
#     #         f"Arête {e} : ε_exact = {epsilon_true[e]:.8f} | "
#     #         f"ε_identifié = {epsilon_current[e]:.8f} | "
#     #         f"erreur abs = {err_abs:.3e} | "
#     #         f"erreur rel = {err_rel:.3e}"
#     #     )

#     # print(f"\nJ initial = {J_init:.10e}")
#     # print(f"J final   = {J_opt:.10e}")
#     # print(f"Réduction : {(J_init - J_opt) / J_init * 100:.2f}%")

#     # # ------------------------------------------------------------
#     # # 8. VISUALISATIONS COMPLÈTES
#     # # ------------------------------------------------------------
#     # print("\n>>> Visualisations finales")

#     # solver.solve_direct(epsilon_current, source_intensity)
#     # solver.solve_sensitivity_epsilon(epsilon_current, source_intensity)
#     # solver.solve_adjoint(epsilon_current, u_data, source_intensity)

#     # solver.plot_all_results(epsilon_current, u_data)

#     # return solver, graph, {
#     #     "epsilon_true": epsilon_true,
#     #     "epsilon_init": epsilon_test,
#     #     "epsilon_opt": epsilon_current,
#     #     "J_init": J_init,
#     #     "J_opt": J_opt,
#     # }
def create_star_graph_three_sources():
    """
    Graphe en étoile à 3 branches
    → idéal pour tester 3 sources indépendantes
    """
    graph = MetricGraph()

    # -----------------------------
    # Sommets
    # -----------------------------
    positions = {
        "v0": (0.0, 0.0),   # centre
        "v1": (1.0, 0.0),   # branche 1
        "v2": (-0.5, 0.8),  # branche 2
        "v3": (-0.5, -0.8)  # branche 3
    }

    for v, (x, y) in positions.items():
        graph.set_vertex_position(v, x, y)

    # -----------------------------
    # Arêtes (3 branches)
    # -----------------------------
    graph.add_edge(
        0, "v0", "v1",
        length=1.0,
        a_coef=1.0,
        n_points=200
    )

    graph.add_edge(
        1, "v0", "v2",
        length=1.0,
        a_coef=1.0,
        n_points=200
    )

    graph.add_edge(
        2, "v0", "v3",
        length=1.0,
        a_coef=1.0,
        n_points=200
    )

    # -----------------------------
    # Conditions de Dirichlet
    # -----------------------------
    graph.set_boundary_vertices(["v1", "v2", "v3"])

    # INDISPENSABLE
    graph.build_dof_map()

    return graph


# ============================================================
# MAIN
# ============================================================
def test_inverse_one_source_vectorial():
    """
    CAS TEST COMPLET – LOCALISATION D'UNE SOURCE
    ✔ même structure que les cas 2 et 3 sources
    ✔ optimisation vectorielle (dimension 1)
    ✔ PG vs CG vs UZAWA
    """

    print("\n" + "=" * 80)
    print("CAS TEST COMPLET – LOCALISATION D'UNE SOURCE (ε vectoriel)")
    print("=" * 80)

    # ============================================================
    # 1. GRAPHE
    # ============================================================
    graph = create_simple_Y_graph()
    graph.plot_graph(title="Graphe en Y – problème inverse (1 source)")

    solver = SourceLocalization(graph)

    print(f"Nombre total de DDL: {graph.n_dof}")

    # ============================================================
    # 2. SOURCE EXACTE
    # ============================================================
    edge_ids = [0]                      # UNE seule arête
    source_intensity = 1.0

    # position relative sur l'arête (0 < ε̂ < 1)
    epsilon_hat_true = np.array([0.6])

    # conversion en position physique
    epsilon_true = np.array([
        epsilon_hat_true[0] * graph.edges[edge_ids[0]]["length"]
    ])

    print("\n>>> Source exacte")
    print(f"  - Arête {edge_ids[0]}")
    print(f"  - ε̂ = {epsilon_hat_true[0]:.3f}")
    print(f"  - ε  = {epsilon_true[0]:.6f}")

    epsilon_true_dict = {
        edge_ids[0]: epsilon_true[0]
    }

    # ============================================================
    # 3. DONNÉES OBSERVÉES
    # ============================================================
    print("\n>>> Génération des données observées (problème direct)")

    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)

    noise_level = 0.0
    np.random.seed(0)
    u_data = u_exact.copy()

    if noise_level > 0:
        u_data += noise_level * np.random.randn(len(u_data))
        print(f"Bruit ajouté : {noise_level*100:.1f}%")
    else:
        print("Aucun bruit (cas de validation)")

    # ============================================================
    # 4. VALIDATION DES GRADIENTS
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε")
    print("=" * 80)

    results = solver.validate_gradient_three_methods_epsilon(
        edge_id=edge_ids[0],
        epsilon=epsilon_true[0],
        u_data=u_data,
        source_intensity=source_intensity,
        alpha_fd=1e-7,
    )

    print("\nGradients :")
    print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
    print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
    print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")

    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.3e}")

    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    epsilon_init = np.array([
        0.5 * graph.edges[edge_ids[0]]["length"]
    ])

    # ------------------------------------------------------------
    # 5.1 Gradient projeté vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)

    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.2 * graph.edges[edge_ids[0]]["length"],
        max_iter=150,
        tol=1e-8,
    )

    print("\nRésultat PG :")
    print(f"  ε = {eps_pg[0]:.6f}")

    solver.solve_direct(
        {edge_ids[0]: eps_pg[0]},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.2 Gradient conjugué vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)

    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=80,
        tol=1e-8,
    )

    print("\nRésultat CG :")
    print(f"  ε = {eps_cg[0]:.6f}")

    solver.solve_direct(
        {edge_ids[0]: eps_cg[0]},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)
    
    # ------------------------------------------------------------
    # 5.3 UZAWA – contrainte de sparsité L1
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (contrainte sparsité L1)")
    print("=" * 80)

    eps_uzawa, lambda_uzawa = solver.uzawa_epsilon_vector(
    epsilon_init=epsilon_init,
    edge_ids=edge_ids,
    u_data=u_data,
    source_intensity=source_intensity,
    K_max=1.0,         # contrainte lâche
    step_init=0.5,     # ⚠️ PAS INITIAL PLUS GRAND (line search l'adaptera)
    rho=1.0,
    max_iter=500,
    tol=1e-10,
)
    print("\n" + "=" * 80)
    print("TEST UZAWA AVEC CONTRAINTE STRICTE (K_max=0.3)")
    print("=" * 80)

    eps_uzawa_strict, lambda_strict = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=0.3,         # ⚠️ CONTRAINTE STRICTE
        step_init=0.3,
        rho=5.0,           # pénalisation plus forte
        max_iter=500,
        tol=1e-10,
    )

    print("\n" + "=" * 80)
    print("COMPARAISON UZAWA")
    print("=" * 80)
    print(f"K_max=1.0 (lâche)   : ε = {eps_uzawa[0]:.6f} | λ = {lambda_uzawa:.3e}")
    print(f"K_max=0.3 (stricte) : ε = {eps_uzawa_strict[0]:.6f} | λ = {lambda_strict:.3e}")
    print(f"Solution exacte     : ε = {epsilon_true[0]:.6f}")


    print("\nRésultat Uzawa :")
    print(f"  ε = {eps_uzawa[0]:.6f}")
    print(f"  λ = {lambda_uzawa:.3e}")
    
    solver.solve_direct(
        {edge_ids[0]: eps_uzawa[0]},
        source_intensity,
    )
    J_uzawa = solver.compute_cost_functional(u_data)

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    print(
        f"Arête {edge_ids[0]} | "
        f"ε exact = {epsilon_true[0]:.6f}"
    )
    
    print(f"\nPositions identifiées :")
    print(f"  PG    : ε = {eps_pg[0]:.6f} | erreur = {abs(eps_pg[0] - epsilon_true[0]):.3e}")
    print(f"  CG    : ε = {eps_cg[0]:.6f} | erreur = {abs(eps_cg[0] - epsilon_true[0]):.3e}")
    print(f"  Uzawa : ε = {eps_uzawa[0]:.6f} | erreur = {abs(eps_uzawa[0] - epsilon_true[0]):.3e}")

    print("\nValeurs du coût :")
    print(f"  J référence = {J_ref:.3e}")
    print(f"  J PG        = {J_pg:.3e}")
    print(f"  J CG        = {J_cg:.3e}")
    print(f"  J Uzawa     = {J_uzawa:.3e}")

    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n>>> Visualisations finales (solution CG)")

    eps_dict_cg = {edge_ids[0]: eps_cg[0]}

    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)

    solver.plot_all_results(eps_dict_cg, u_data)

    return {
        "epsilon_true": epsilon_true,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa": eps_uzawa,
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
        "J_uzawa": J_uzawa,
    }

# def test_inverse_one_source_vectorial():
#     """
#     CAS TEST COMPLET – LOCALISATION D'UNE SOURCE
#     ✔ même structure que les cas 2 et 3 sources
#     ✔ optimisation vectorielle (dimension 1)
#     ✔ PG vs CG
#     """

#     print("\n" + "=" * 80)
#     print("CAS TEST COMPLET – LOCALISATION D’UNE SOURCE (ε vectoriel)")
#     print("=" * 80)

#     # ============================================================
#     # 1. GRAPHE
#     # ============================================================
#     graph = create_simple_Y_graph()
#     graph.plot_graph(title="Graphe en Y – problème inverse (1 source)")

#     solver = SourceLocalization(graph)

#     print(f"Nombre total de DDL: {graph.n_dof}")

#     # ============================================================
#     # 2. SOURCE EXACTE
#     # ============================================================
#     edge_ids = [0]                      # UNE seule arête
#     source_intensity = 1.0

#     # position relative sur l’arête (0 < ε̂ < 1)
#     epsilon_hat_true = np.array([0.6])

#     # conversion en position physique
#     epsilon_true = np.array([
#         epsilon_hat_true[0] * graph.edges[edge_ids[0]]["length"]
#     ])

#     print("\n>>> Source exacte")
#     print(f"  - Arête {edge_ids[0]}")
#     print(f"  - ε̂ = {epsilon_hat_true[0]:.3f}")
#     print(f"  - ε  = {epsilon_true[0]:.6f}")

#     epsilon_true_dict = {
#         edge_ids[0]: epsilon_true[0]
#     }

#     # ============================================================
#     # 3. DONNÉES OBSERVÉES
#     # ============================================================
#     print("\n>>> Génération des données observées (problème direct)")

#     u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)

#     noise_level = 0.0
#     np.random.seed(0)
#     u_data = u_exact.copy()

#     if noise_level > 0:
#         u_data += noise_level * np.random.randn(len(u_data))
#         print(f"Bruit ajouté : {noise_level*100:.1f}%")
#     else:
#         print("Aucun bruit (cas de validation)")

#     # ============================================================
#     # 4. VALIDATION DES GRADIENTS
#     # ============================================================
#     print("\n" + "=" * 80)
#     print("VALIDATION DES GRADIENTS dJ/dε")
#     print("=" * 80)

#     results = solver.validate_gradient_three_methods_epsilon(
#         edge_id=edge_ids[0],
#         epsilon=epsilon_true[0],
#         u_data=u_data,
#         source_intensity=source_intensity,
#         alpha_fd=1e-7,
#     )

#     print("\nGradients :")
#     print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
#     print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
#     print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")

#     solver.solve_direct(epsilon_true_dict, source_intensity)
#     J_ref = solver.compute_cost_functional(u_data)
#     print(f"\nValeur du coût J (référence) = {J_ref:.3e}")

#     # ============================================================
#     # 5. OPTIMISATION VECTORIELLE
#     # ============================================================
#     epsilon_init = np.array([
#         0.2 * graph.edges[edge_ids[0]]["length"]
#     ])

#     # ------------------------------------------------------------
#     # 5.1 Gradient projeté vectoriel
#     # ------------------------------------------------------------
#     print("\n" + "=" * 80)
#     print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
#     print("=" * 80)

#     eps_pg = solver.projected_gradient_epsilon_vector(
#         epsilon_init=epsilon_init,
#         edge_ids=edge_ids,
#         u_data=u_data,
#         source_intensity=source_intensity,
#         step=0.2 * graph.edges[edge_ids[0]]["length"],
#         max_iter=150,
#         tol=1e-8,
#     )

#     print("\nRésultat PG :")
#     print(f"  ε = {eps_pg[0]:.6f}")

#     solver.solve_direct(
#         {edge_ids[0]: eps_pg[0]},
#         source_intensity,
#     )
#     J_pg = solver.compute_cost_functional(u_data)

#     # ------------------------------------------------------------
#     # 5.2 Gradient conjugué vectoriel
#     # ------------------------------------------------------------
#     print("\n" + "=" * 80)
#     print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
#     print("=" * 80)

#     eps_cg = solver.conjugate_gradient_epsilon_vector(
#         epsilon_init=epsilon_init,
#         edge_ids=edge_ids,
#         u_data=u_data,
#         source_intensity=source_intensity,
#         max_iter=80,
#         tol=1e-8,
#     )

#     print("\nRésultat CG :")
#     print(f"  ε = {eps_cg[0]:.6f}")

#     solver.solve_direct(
#         {edge_ids[0]: eps_cg[0]},
#         source_intensity,
#     )
#     J_cg = solver.compute_cost_functional(u_data)
#     # ------------------------------------------------------------
#     # 5.3 UZAWA – contrainte sum(epsilon) <= 1
#     # ------------------------------------------------------------
#     print("\n" + "=" * 80)
#     print("OPTIMISATION – UZAWA (contrainte somme)")
#     print("=" * 80)

#     eps_uzawa, lambda_uzawa = solver.uzawa_epsilon_vector(
#         epsilon_init=epsilon_init,
#         edge_ids=edge_ids,
#         u_data=u_data,
#         source_intensity=source_intensity,
#         step=0.05,
#         rho=1,
#         max_iter=500,
#         tol=1e-10,
#     )

#     print("\nRésultat Uzawa :")
#     print(f"  ε = {eps_uzawa[0]:.6f}")
#     print(f"  λ = {lambda_uzawa:.3e}")
#     solver.solve_direct(
#     {edge_ids[0]: eps_uzawa[0]},
#     source_intensity,
#     )
#     J_uzawa = solver.compute_cost_functional(u_data)


#     # ============================================================
#     # 6. COMPARAISON FINALE
#     # ============================================================
#     print("\n" + "=" * 80)
#     print("COMPARAISON DES MÉTHODES")
#     print("=" * 80)

#     print(
#         f"Arête {edge_ids[0]} | "
#         f"ε exact = {epsilon_true[0]:.6f} | "
#         f"ε PG = {eps_pg[0]:.6f} | "
#         f"ε CG = {eps_cg[0]:.6f}"
#     )

#     print("\nValeurs du coût :")
#     print(f"  J référence = {J_ref:.3e}")
#     print(f"  J PG        = {J_pg:.3e}")
#     print(f"  J CG        = {J_cg:.3e}")
#     print(f"  J Uzawa     = {J_uzawa:.3e}")


#     # ============================================================
#     # 7. VISUALISATIONS FINALES (CG)
#     # ============================================================
#     print("\n>>> Visualisations finales (solution CG)")

#     eps_dict_cg = {edge_ids[0]: eps_cg[0]}

#     solver.solve_direct(eps_dict_cg, source_intensity)
#     solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
#     solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)

#     solver.plot_all_results(eps_dict_cg, u_data)

#     return {
#         "epsilon_true": epsilon_true,
#         "epsilon_init": epsilon_init,
#         "epsilon_pg": eps_pg,
#         "epsilon_cg": eps_cg,
#         "J_ref": J_ref,
#         "J_pg": J_pg,
#         "J_cg": J_cg,
#     }

def test_inverse_source_localization_two_sources_complete():
    """
    CAS TEST COMPLET – LOCALISATION DE DEUX SOURCES
    ✔ optimisation conjointe vectorielle
    ✔ gradient adjoint validé
    ✔ PG vectoriel vs CG vectoriel vs UZAWA
    """

    print("\n" + "=" * 80)
    print("CAS TEST COMPLET – LOCALISATION DE DEUX SOURCES (ε vectoriel)")
    print("=" * 80)

    # ============================================================
    # 1. GRAPHE
    # ============================================================
    graph = create_simple_2d_graph()
    graph.plot_graph(title="Graphe en Y – problème inverse (2 sources)")

    solver = SourceLocalization(graph)

    print(f"Nombre total de DDL: {graph.n_dof}")

    # ============================================================
    # 2. SOURCES EXACTES
    # ============================================================
    edge_ids = [0, 1]
    epsilon_true = np.array([0.4, 0.7])
    source_intensity = 1.0

    print("\n>>> Sources exactes")
    for i, eid in enumerate(edge_ids):
        print(f"  - Arête {eid} : ε = {epsilon_true[i]}")

    epsilon_true_dict = {
        edge_ids[i]: epsilon_true[i] for i in range(len(edge_ids))
    }

    # ============================================================
    # 3. DONNÉES OBSERVÉES
    # ============================================================
    print("\n>>> Génération des données observées (problème direct)")
    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)

    noise_level = 0.0
    np.random.seed(0)
    u_data = u_exact.copy()

    if noise_level > 0:
        u_data += noise_level * np.random.randn(len(u_data))
        print(f"Bruit ajouté : {noise_level*100:.1f}%")
    else:
        print("Aucun bruit (cas de validation)")

    # ============================================================
    # 4. VALIDATION DES GRADIENTS (PAR ARÊTE)
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε (par arête)")
    print("=" * 80)

    for i, eid in enumerate(edge_ids):
        print(f"\n--- Arête {eid} | ε = {epsilon_true[i]} ---")

        results = solver.validate_gradient_three_methods_epsilon(
            edge_id=eid,
            epsilon=epsilon_true[i],
            u_data=u_data,
            source_intensity=source_intensity,
            alpha_fd=1e-7,
        )

        print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
        print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
        print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")

    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.3e}")

    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    epsilon_init = np.array([0.5, 0.5])  # meilleur point de départ

    # ------------------------------------------------------------
    # 5.1 Gradient projeté
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)

    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.2,
        max_iter=200,
        tol=1e-8,
    )

    print("\nRésultat PG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_pg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_pg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.2 Gradient conjugué
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)

    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=100,
        tol=1e-8,
    )

    print("\nRésultat CG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_cg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.3 UZAWA avec différentes contraintes
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (plusieurs contraintes)")
    print("=" * 80)

    # Test avec K_max = 2.0 (lâche)
    print("\n>>> Uzawa avec K_max = 2.0 (lâche)")
    eps_uzawa_2, lam_uzawa_2 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=2.0,
        step_init=0.5,
        rho=1.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,  # silencieux
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_2[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_2 = solver.compute_cost_functional(u_data)

    print(f"K_max=2.0 : ε₀={eps_uzawa_2[0]:.6f}, ε₁={eps_uzawa_2[1]:.6f}, λ={lam_uzawa_2:.3e}, J={J_uzawa_2:.3e}")

    # Test avec K_max = 1.5 (modéré)
    print("\n>>> Uzawa avec K_max = 1.5 (modéré)")
    eps_uzawa_15, lam_uzawa_15 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=1.5,
        step_init=0.4,
        rho=2.0,
        max_iter=500,
        tol=1e-10,
        verbose=True,  # affichage détaillé
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_15[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_15 = solver.compute_cost_functional(u_data)

    # Test avec K_max = 1.0 (strict)
    print("\n>>> Uzawa avec K_max = 1.0 (strict)")
    eps_uzawa_1, lam_uzawa_1 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=1.0,
        step_init=0.3,
        rho=3.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_1[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_1 = solver.compute_cost_functional(u_data)

    print(f"K_max=1.0 : ε₀={eps_uzawa_1[0]:.6f}, ε₁={eps_uzawa_1[1]:.6f}, λ={lam_uzawa_1:.3e}, J={J_uzawa_1:.3e}")

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    print("\nSOLUTION EXACTE :")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {epsilon_true[i]:.6f}")
    print(f"  Sparsité L1 : {sum(epsilon_true / np.array([graph.edges[eid]['length'] for eid in edge_ids])):.4f}")

    print("\nRÉSULTATS :")
    print(f"{'Méthode':<20} {'ε₀':<10} {'ε₁':<10} {'Sparsité':<12} {'J':<12} {'Erreur ε₀':<12} {'Erreur ε₁':<12}")
    print("-" * 90)

    # CG
    sparsity_cg = sum(eps_cg / np.array([graph.edges[eid]['length'] for eid in edge_ids]))
    err0_cg = abs(eps_cg[0] - epsilon_true[0])
    err1_cg = abs(eps_cg[1] - epsilon_true[1])
    print(f"{'CG':<20} {eps_cg[0]:<10.6f} {eps_cg[1]:<10.6f} {sparsity_cg:<12.4f} {J_cg:<12.3e} {err0_cg:<12.3e} {err1_cg:<12.3e}")

    # PG
    sparsity_pg = sum(eps_pg / np.array([graph.edges[eid]['length'] for eid in edge_ids]))
    err0_pg = abs(eps_pg[0] - epsilon_true[0])
    err1_pg = abs(eps_pg[1] - epsilon_true[1])
    print(f"{'PG':<20} {eps_pg[0]:<10.6f} {eps_pg[1]:<10.6f} {sparsity_pg:<12.4f} {J_pg:<12.3e} {err0_pg:<12.3e} {err1_pg:<12.3e}")

    # Uzawa K=2.0
    sparsity_u2 = sum(eps_uzawa_2 / np.array([graph.edges[eid]['length'] for eid in edge_ids]))
    err0_u2 = abs(eps_uzawa_2[0] - epsilon_true[0])
    err1_u2 = abs(eps_uzawa_2[1] - epsilon_true[1])
    print(f"{'Uzawa (K=2.0)':<20} {eps_uzawa_2[0]:<10.6f} {eps_uzawa_2[1]:<10.6f} {sparsity_u2:<12.4f} {J_uzawa_2:<12.3e} {err0_u2:<12.3e} {err1_u2:<12.3e}")

    # Uzawa K=1.5
    sparsity_u15 = sum(eps_uzawa_15 / np.array([graph.edges[eid]['length'] for eid in edge_ids]))
    err0_u15 = abs(eps_uzawa_15[0] - epsilon_true[0])
    err1_u15 = abs(eps_uzawa_15[1] - epsilon_true[1])
    print(f"{'Uzawa (K=1.5)':<20} {eps_uzawa_15[0]:<10.6f} {eps_uzawa_15[1]:<10.6f} {sparsity_u15:<12.4f} {J_uzawa_15:<12.3e} {err0_u15:<12.3e} {err1_u15:<12.3e}")

    # Uzawa K=1.0
    sparsity_u1 = sum(eps_uzawa_1 / np.array([graph.edges[eid]['length'] for eid in edge_ids]))
    err0_u1 = abs(eps_uzawa_1[0] - epsilon_true[0])
    err1_u1 = abs(eps_uzawa_1[1] - epsilon_true[1])
    print(f"{'Uzawa (K=1.0)':<20} {eps_uzawa_1[0]:<10.6f} {eps_uzawa_1[1]:<10.6f} {sparsity_u1:<12.4f} {J_uzawa_1:<12.3e} {err0_u1:<12.3e} {err1_u1:<12.3e}")

    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n>>> Visualisations finales (solution CG)")

    eps_dict_cg = {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))}

    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)

    solver.plot_all_results(eps_dict_cg, u_data)

    return {
        "epsilon_true": epsilon_true,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa_2": eps_uzawa_2,
        "epsilon_uzawa_15": eps_uzawa_15,
        "epsilon_uzawa_1": eps_uzawa_1,
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
        "J_uzawa_2": J_uzawa_2,
        "J_uzawa_15": J_uzawa_15,
        "J_uzawa_1": J_uzawa_1,
    }


def test_inverse_three_sources_vectorial():
    """
    CAS TEST COMPLET – LOCALISATION DE TROIS SOURCES
    ✔ graphe étoile
    ✔ 1 source par arête (strictement sur l'arête)
    ✔ optimisation vectorielle conjointe
    ✔ Gradient projeté vs Gradient conjugué vs UZAWA
    """

    print("\n" + "=" * 80)
    print("CAS TEST COMPLET – LOCALISATION DE TROIS SOURCES (ε vectoriel)")
    print("=" * 80)

    # ============================================================
    # 1. GRAPHE ÉTOILE
    # ============================================================
    graph = create_star()
    graph.plot_graph(title="Graphe étoile – problème inverse (3 sources)")

    solver = SourceLocalization(graph)

    print(f"Nombre total de DDL: {graph.n_dof}")

    # ============================================================
    # 2. SOURCES EXACTES (UNE PAR ARÊTE)
    # ============================================================
    edge_ids = [0, 1, 2]
    source_intensity = 1.0

    # positions relatives (fractions) le long de CHAQUE arête
    epsilon_hat = np.array([0.6, 0.5, 0.7])  # ∈ (0,1)

    epsilon_true_dict = {
        eid: epsilon_hat[i] * graph.edges[eid]["length"]
        for i, eid in enumerate(edge_ids)
    }

    print("\n>>> Sources exactes")
    for i, eid in enumerate(edge_ids):
        print(
            f"  - Arête {eid} : "
            f"ε̂ = {epsilon_hat[i]:.2f}  |  "
            f"ε = {epsilon_true_dict[eid]:.4f}"
        )

    # ============================================================
    # 3. DONNÉES OBSERVÉES (PROBLÈME DIRECT)
    # ============================================================
    print("\n>>> Génération des données observées")

    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)
    u_data = u_exact.copy()  # sans bruit (cas de validation)

    # ============================================================
    # 4. VALIDATION DES GRADIENTS (PAR ARÊTE)
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε (par arête)")
    print("=" * 80)

    for i, eid in enumerate(edge_ids):
        eps_test = epsilon_true_dict[eid]

        print(f"\n--- Arête {eid} | ε = {eps_test:.6f} ---")

        results = solver.validate_gradient_three_methods_epsilon(
            edge_id=eid,
            epsilon=eps_test,
            u_data=u_data,
            source_intensity=source_intensity,
            alpha_fd=1e-7,
        )

        print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
        print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
        print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")

    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.3e}")

    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    epsilon_init = np.array([0.5, 0.5, 0.5])  # meilleur point de départ

    # ------------------------------------------------------------
    # 5.1 Gradient projeté vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)

    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.2,
        max_iter=300,
        tol=1e-8,
    )

    print("\nRésultat PG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_pg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_pg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.2 Gradient conjugué vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)

    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=150,
        tol=1e-8,
    )

    print("\nRésultat CG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_cg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.3 UZAWA avec différentes contraintes
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (plusieurs contraintes)")
    print("=" * 80)

    # Test avec K_max = 3.0 (lâche)
    print("\n>>> Uzawa avec K_max = 3.0 (lâche)")
    eps_uzawa_3, lam_uzawa_3 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=3.0,
        step_init=0.5,
        rho=1.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_3[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_3 = solver.compute_cost_functional(u_data)

    print(f"K_max=3.0 : ε₀={eps_uzawa_3[0]:.4f}, ε₁={eps_uzawa_3[1]:.4f}, ε₂={eps_uzawa_3[2]:.4f}, λ={lam_uzawa_3:.3e}, J={J_uzawa_3:.3e}")

    # Test avec K_max = 2.0 (modéré)
    print("\n>>> Uzawa avec K_max = 2.0 (modéré)")
    eps_uzawa_2, lam_uzawa_2 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=2.0,
        step_init=0.4,
        rho=2.0,
        max_iter=500,
        tol=1e-10,
        verbose=True,  # affichage détaillé
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_2[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_2 = solver.compute_cost_functional(u_data)

    # Test avec K_max = 1.5 (strict)
    print("\n>>> Uzawa avec K_max = 1.5 (strict)")
    eps_uzawa_15, lam_uzawa_15 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=1.5,
        step_init=0.3,
        rho=3.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_15[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_15 = solver.compute_cost_functional(u_data)

    print(f"K_max=1.5 : ε₀={eps_uzawa_15[0]:.4f}, ε₁={eps_uzawa_15[1]:.4f}, ε₂={eps_uzawa_15[2]:.4f}, λ={lam_uzawa_15:.3e}, J={J_uzawa_15:.3e}")

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    epsilon_true_vec = np.array([epsilon_true_dict[eid] for eid in edge_ids])
    L_vec = np.array([graph.edges[eid]["length"] for eid in edge_ids])

    print("\nSOLUTION EXACTE :")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {epsilon_true_vec[i]:.6f}")
    print(f"  Sparsité L1 : {sum(epsilon_true_vec / L_vec):.4f}")

    print("\nRÉSULTATS :")
    print(f"{'Méthode':<20} {'ε₀':<10} {'ε₁':<10} {'ε₂':<10} {'Sparsité':<12} {'J':<12}")
    print("-" * 80)

    # CG
    sparsity_cg = sum(eps_cg / L_vec)
    print(f"{'CG':<20} {eps_cg[0]:<10.4f} {eps_cg[1]:<10.4f} {eps_cg[2]:<10.4f} {sparsity_cg:<12.4f} {J_cg:<12.3e}")

    # PG
    sparsity_pg = sum(eps_pg / L_vec)
    print(f"{'PG':<20} {eps_pg[0]:<10.4f} {eps_pg[1]:<10.4f} {eps_pg[2]:<10.4f} {sparsity_pg:<12.4f} {J_pg:<12.3e}")

    # Uzawa K=3.0
    sparsity_u3 = sum(eps_uzawa_3 / L_vec)
    print(f"{'Uzawa (K=3.0)':<20} {eps_uzawa_3[0]:<10.4f} {eps_uzawa_3[1]:<10.4f} {eps_uzawa_3[2]:<10.4f} {sparsity_u3:<12.4f} {J_uzawa_3:<12.3e}")

    # Uzawa K=2.0
    sparsity_u2 = sum(eps_uzawa_2 / L_vec)
    print(f"{'Uzawa (K=2.0)':<20} {eps_uzawa_2[0]:<10.4f} {eps_uzawa_2[1]:<10.4f} {eps_uzawa_2[2]:<10.4f} {sparsity_u2:<12.4f} {J_uzawa_2:<12.3e}")

    # Uzawa K=1.5
    sparsity_u15 = sum(eps_uzawa_15 / L_vec)
    print(f"{'Uzawa (K=1.5)':<20} {eps_uzawa_15[0]:<10.4f} {eps_uzawa_15[1]:<10.4f} {eps_uzawa_15[2]:<10.4f} {sparsity_u15:<12.4f} {J_uzawa_15:<12.3e}")

    print("\nERREURS PAR RAPPORT À LA SOLUTION EXACTE :")
    print(f"{'Méthode':<20} {'Erreur ε₀':<12} {'Erreur ε₁':<12} {'Erreur ε₂':<12} {'Erreur moyenne':<15}")
    print("-" * 75)

    def print_errors(name, eps):
        errs = np.abs(eps - epsilon_true_vec)
        print(f"{name:<20} {errs[0]:<12.4f} {errs[1]:<12.4f} {errs[2]:<12.4f} {np.mean(errs):<15.4f}")

    print_errors("CG", eps_cg)
    print_errors("PG", eps_pg)
    print_errors("Uzawa (K=3.0)", eps_uzawa_3)
    print_errors("Uzawa (K=2.0)", eps_uzawa_2)
    print_errors("Uzawa (K=1.5)", eps_uzawa_15)

    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n>>> Visualisations finales (solution CG)")

    eps_dict_cg = {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))}

    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)

    solver.plot_all_results(eps_dict_cg, u_data)

    return {
        "epsilon_true": epsilon_true_dict,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa_3": eps_uzawa_3,
        "epsilon_uzawa_2": eps_uzawa_2,
        "epsilon_uzawa_15": eps_uzawa_15,
        "J_ref": J_ref,
        "J_pg": J_pg}
def test_inverse_three_sources_vectorial():
    """
    CAS TEST COMPLET – LOCALISATION DE TROIS SOURCES
    ✔ graphe étoile
    ✔ 1 source par arête (strictement sur l'arête)
    ✔ optimisation vectorielle conjointe
    ✔ Gradient projeté vs Gradient conjugué vs UZAWA
    """

    print("\n" + "=" * 80)
    print("CAS TEST COMPLET – LOCALISATION DE TROIS SOURCES (ε vectoriel)")
    print("=" * 80)

    # ============================================================
    # 1. GRAPHE ÉTOILE
    # ============================================================
    graph = create_star()
    graph.plot_graph(title="Graphe étoile – problème inverse (3 sources)")

    solver = SourceLocalization(graph)

    print(f"Nombre total de DDL: {graph.n_dof}")

    # ============================================================
    # 2. SOURCES EXACTES (UNE PAR ARÊTE)
    # ============================================================
    edge_ids = [0, 1, 2]
    source_intensity = 1.0

    # positions relatives (fractions) le long de CHAQUE arête
    epsilon_hat = np.array([0.6, 0.5, 0.7])  # ∈ (0,1)

    epsilon_true_dict = {
        eid: epsilon_hat[i] * graph.edges[eid]["length"]
        for i, eid in enumerate(edge_ids)
    }

    print("\n>>> Sources exactes")
    for i, eid in enumerate(edge_ids):
        print(
            f"  - Arête {eid} : "
            f"ε̂ = {epsilon_hat[i]:.2f}  |  "
            f"ε = {epsilon_true_dict[eid]:.4f}"
        )

    # ============================================================
    # 3. DONNÉES OBSERVÉES (PROBLÈME DIRECT)
    # ============================================================
    print("\n>>> Génération des données observées")

    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)
    u_data = u_exact.copy()  # sans bruit (cas de validation)

    # ============================================================
    # 4. VALIDATION DES GRADIENTS (PAR ARÊTE)
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε (par arête)")
    print("=" * 80)

    for i, eid in enumerate(edge_ids):
        eps_test = epsilon_true_dict[eid]

        print(f"\n--- Arête {eid} | ε = {eps_test:.6f} ---")

        results = solver.validate_gradient_three_methods_epsilon(
            edge_id=eid,
            epsilon=eps_test,
            u_data=u_data,
            source_intensity=source_intensity,
            alpha_fd=1e-7,
        )

        print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
        print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
        print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")

    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.3e}")

    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    epsilon_init = np.array([0.5, 0.5, 0.5])  # meilleur point de départ

    # ------------------------------------------------------------
    # 5.1 Gradient projeté vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)

    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.2,
        max_iter=300,
        tol=1e-8,
    )

    print("\nRésultat PG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_pg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_pg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.2 Gradient conjugué vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)

    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=150,
        tol=1e-8,
    )

    print("\nRésultat CG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_cg[i]:.6f}")

    solver.solve_direct(
        {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)

    # ------------------------------------------------------------
    # 5.3 UZAWA avec différentes contraintes
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (plusieurs contraintes)")
    print("=" * 80)

    # Test avec K_max = 3.0 (lâche)
    print("\n>>> Uzawa avec K_max = 3.0 (lâche)")
    eps_uzawa_3, lam_uzawa_3 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=3.0,
        step_init=0.5,
        rho=1.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_3[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_3 = solver.compute_cost_functional(u_data)

    print(f"K_max=3.0 : ε₀={eps_uzawa_3[0]:.4f}, ε₁={eps_uzawa_3[1]:.4f}, ε₂={eps_uzawa_3[2]:.4f}, λ={lam_uzawa_3:.3e}, J={J_uzawa_3:.3e}")

    # Test avec K_max = 2.0 (modéré)
    print("\n>>> Uzawa avec K_max = 2.0 (modéré)")
    eps_uzawa_2, lam_uzawa_2 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=2.0,
        step_init=0.4,
        rho=2.0,
        max_iter=500,
        tol=1e-10,
        verbose=True,  # affichage détaillé
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_2[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_2 = solver.compute_cost_functional(u_data)

    # Test avec K_max = 1.5 (strict)
    print("\n>>> Uzawa avec K_max = 1.5 (strict)")
    eps_uzawa_15, lam_uzawa_15 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=1.5,
        step_init=0.3,
        rho=3.0,
        max_iter=500,
        tol=1e-10,
        verbose=False,
    )

    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_15[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_15 = solver.compute_cost_functional(u_data)

    print(f"K_max=1.5 : ε₀={eps_uzawa_15[0]:.4f}, ε₁={eps_uzawa_15[1]:.4f}, ε₂={eps_uzawa_15[2]:.4f}, λ={lam_uzawa_15:.3e}, J={J_uzawa_15:.3e}")

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    epsilon_true_vec = np.array([epsilon_true_dict[eid] for eid in edge_ids])
    L_vec = np.array([graph.edges[eid]["length"] for eid in edge_ids])

    print("\nSOLUTION EXACTE :")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {epsilon_true_vec[i]:.6f}")
    print(f"  Sparsité L1 : {sum(epsilon_true_vec / L_vec):.4f}")

    print("\nRÉSULTATS :")
    print(f"{'Méthode':<20} {'ε₀':<10} {'ε₁':<10} {'ε₂':<10} {'Sparsité':<12} {'J':<12}")
    print("-" * 80)

    # CG
    sparsity_cg = sum(eps_cg / L_vec)
    print(f"{'CG':<20} {eps_cg[0]:<10.4f} {eps_cg[1]:<10.4f} {eps_cg[2]:<10.4f} {sparsity_cg:<12.4f} {J_cg:<12.3e}")

    # PG
    sparsity_pg = sum(eps_pg / L_vec)
    print(f"{'PG':<20} {eps_pg[0]:<10.4f} {eps_pg[1]:<10.4f} {eps_pg[2]:<10.4f} {sparsity_pg:<12.4f} {J_pg:<12.3e}")

    # Uzawa K=3.0
    sparsity_u3 = sum(eps_uzawa_3 / L_vec)
    print(f"{'Uzawa (K=3.0)':<20} {eps_uzawa_3[0]:<10.4f} {eps_uzawa_3[1]:<10.4f} {eps_uzawa_3[2]:<10.4f} {sparsity_u3:<12.4f} {J_uzawa_3:<12.3e}")

    # Uzawa K=2.0
    sparsity_u2 = sum(eps_uzawa_2 / L_vec)
    print(f"{'Uzawa (K=2.0)':<20} {eps_uzawa_2[0]:<10.4f} {eps_uzawa_2[1]:<10.4f} {eps_uzawa_2[2]:<10.4f} {sparsity_u2:<12.4f} {J_uzawa_2:<12.3e}")

    # Uzawa K=1.5
    sparsity_u15 = sum(eps_uzawa_15 / L_vec)
    print(f"{'Uzawa (K=1.5)':<20} {eps_uzawa_15[0]:<10.4f} {eps_uzawa_15[1]:<10.4f} {eps_uzawa_15[2]:<10.4f} {sparsity_u15:<12.4f} {J_uzawa_15:<12.3e}")

    print("\nERREURS PAR RAPPORT À LA SOLUTION EXACTE :")
    print(f"{'Méthode':<20} {'Erreur ε₀':<12} {'Erreur ε₁':<12} {'Erreur ε₂':<12} {'Erreur moyenne':<15}")
    print("-" * 75)

    def print_errors(name, eps):
        errs = np.abs(eps - epsilon_true_vec)
        print(f"{name:<20} {errs[0]:<12.4f} {errs[1]:<12.4f} {errs[2]:<12.4f} {np.mean(errs):<15.4f}")

    print_errors("CG", eps_cg)
    print_errors("PG", eps_pg)
    print_errors("Uzawa (K=3.0)", eps_uzawa_3)
    print_errors("Uzawa (K=2.0)", eps_uzawa_2)
    print_errors("Uzawa (K=1.5)", eps_uzawa_15)

    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n>>> Visualisations finales (solution CG)")

    eps_dict_cg = {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))}

    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)

    solver.plot_all_results(eps_dict_cg, u_data)

    return {
        "epsilon_true": epsilon_true_dict,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa_3": eps_uzawa_3,
        "epsilon_uzawa_2": eps_uzawa_2,
        "epsilon_uzawa_15": eps_uzawa_15,
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
        "J_uzawa_3": J_uzawa_3,
        "J_uzawa_2": J_uzawa_2,
        "J_uzawa_15": J_uzawa_15,
    }
def create_star():
    graph = MetricGraph()

    positions = {
        "v0": (0.0, 0.0),
        "v1": (1.0, 0.0),
        "v2": (-0.5, 0.8),
        "v3": (-0.5, -0.8),
    }

    for v, (x, y) in positions.items():
        graph.set_vertex_position(v, x, y)

    def edge_length(vA, vB):
        xA, yA = positions[vA]
        xB, yB = positions[vB]
        return np.sqrt((xB - xA)**2 + (yB - yA)**2)

    graph.add_edge(
        0, "v0", "v1",
        length=edge_length("v0", "v1"),
        a_coef=1.0,
        n_points=200
    )

    graph.add_edge(
        1, "v0", "v2",
        length=edge_length("v0", "v2"),
        a_coef=1.0,
        n_points=200
    )

    graph.add_edge(
        2, "v0", "v3",
        length=edge_length("v0", "v3"),
        a_coef=1.0,
        n_points=200
    )

    graph.set_boundary_vertices(["v1", "v2", "v3"])
    graph.build_dof_map()

    return graph
# ============================================================
# GRAPHE COMPLEXE - RÉSEAU
# ============================================================

def create_complex_network():
    """
    Graphe complexe en réseau (type vasculaire/routier)
    
    Structure :
           v3 --- v4
          /  \   /  \
        v0 -- v1 -- v2 -- v5
          \   |   /
           \ v6  /
            \   /
             v7
    
    9 arêtes, 8 sommets, topologie complexe
    """
    graph = MetricGraph()
    
    # -----------------------------
    # Positions des sommets
    # -----------------------------
    positions = {
        "v0": (0.0, 0.5),    # gauche centre
        "v1": (1.0, 1.0),    # centre haut
        "v2": (2.0, 0.5),    # droite centre
        "v3": (0.5, 2.0),    # haut gauche
        "v4": (1.5, 2.0),    # haut droite
        "v5": (3.0, 0.5),    # extrême droite (bord)
        "v6": (1.0, 0.0),    # centre bas
        "v7": (1.0, -1.0),   # bas (bord)
    }
    
    for v, (x, y) in positions.items():
        graph.set_vertex_position(v, x, y)
    
    # -----------------------------
    # Fonction pour calculer longueur
    # -----------------------------
    def edge_length(vA, vB):
        xA, yA = positions[vA]
        xB, yB = positions[vB]
        return np.sqrt((xB - xA)**2 + (yB - yA)**2)
    
    # -----------------------------
    # Arêtes du réseau (9 arêtes)
    # -----------------------------
    edges_def = [
        (0, "v0", "v1"),   # horizontal bas gauche
        (1, "v1", "v2"),   # horizontal centre
        (2, "v2", "v5"),   # vers extrême droite
        (3, "v0", "v3"),   # diagonale haut gauche
        (4, "v1", "v3"),   # vertical gauche
        (5, "v1", "v4"),   # vertical droite
        (6, "v2", "v4"),   # diagonale haut droite
        (7, "v0", "v6"),   # vers bas gauche
        (8, "v1", "v6"),   # vers bas centre
    ]
    
    n_points = 150  # résolution fine
    
    for eid, vA, vB in edges_def:
        graph.add_edge(
            eid, vA, vB,
            length=edge_length(vA, vB),
            a_coef=1.0,
            n_points=n_points
        )
    
    # -----------------------------
    # Conditions aux limites
    # -----------------------------
    graph.set_boundary_vertices(["v3", "v4", "v5", "v7"])
    
    graph.build_dof_map()
    
    return graph


def plot_network_with_sources(graph, epsilon_dict, title="Graphe avec sources"):
    """
    Visualise le graphe réseau avec les positions des sources
    
    Args:
        graph: MetricGraph
        epsilon_dict: {edge_id: epsilon} positions des sources
        title: titre du graphique
    """
    plt.figure(figsize=(12, 10))
    
    # Récupérer les positions des sommets
    vertex_positions = {}
    for edge in graph.edges:
        v_start = edge['v_start']
        v_end = edge['v_end']
        if v_start not in vertex_positions:
            vertex_positions[v_start] = graph.vertex_positions[v_start]
        if v_end not in vertex_positions:
            vertex_positions[v_end] = graph.vertex_positions[v_end]
    
    # 1. Dessiner les arêtes
    for edge in graph.edges:
        eid = edge['id']
        v_start = edge['v_start']
        v_end = edge['v_end']
        
        x_start, y_start = vertex_positions[v_start]
        x_end, y_end = vertex_positions[v_end]
        
        # Arête avec ou sans source
        if eid in epsilon_dict:
            plt.plot([x_start, x_end], [y_start, y_end], 
                    'b-', linewidth=3, alpha=0.7, label='Arête avec source' if eid == list(epsilon_dict.keys())[0] else '')
        else:
            plt.plot([x_start, x_end], [y_start, y_end], 
                    'gray', linewidth=2, alpha=0.5, label='Arête sans source' if eid == 1 else '')
        
        # Numéro de l'arête au milieu
        x_mid = (x_start + x_end) / 2
        y_mid = (y_start + y_end) / 2
        plt.text(x_mid, y_mid, f"{eid}", 
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Dessiner les sommets
    boundary_vertices = set(graph.boundary_vertices)
    
    for v_id, (x, y) in vertex_positions.items():
        if v_id in boundary_vertices:
            plt.plot(x, y, 'rs', markersize=15, label='Sommet bord' if v_id == list(boundary_vertices)[0] else '')
        else:
            plt.plot(x, y, 'ko', markersize=12, label='Sommet interne' if v_id == 'v1' else '')
        
        # Nom du sommet
        plt.text(x, y + 0.15, v_id, 
                fontsize=11, ha='center', va='bottom', fontweight='bold')
    
    # 3. Dessiner les sources (étoiles rouges)
    for eid, epsilon in epsilon_dict.items():
        edge = graph.edges[eid]
        v_start = edge['v_start']
        v_end = edge['v_end']
        L = edge['length']
        
        x_start, y_start = vertex_positions[v_start]
        x_end, y_end = vertex_positions[v_end]
        
        # Position de la source sur l'arête
        t = epsilon / L  # fraction de la longueur
        x_source = x_start + t * (x_end - x_start)
        y_source = y_start + t * (y_end - y_start)
        
        plt.plot(x_source, y_source, 'r*', markersize=20, 
                label='Source' if eid == list(epsilon_dict.keys())[0] else '')
        
        # Annotation de la source
        plt.text(x_source + 0.1, y_source + 0.1, 
                f"S{list(epsilon_dict.keys()).index(eid)}\nε={epsilon:.3f}", 
                fontsize=9, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.8))
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
# ============================================================
# GRAPHE COMPLEXE - RÉSEAU
# ============================================================

def create_complex_network():
    """
    Graphe complexe en réseau (type vasculaire/routier)
    
    Structure :
           v3 --- v4
          /  \   /  \
        v0 -- v1 -- v2 -- v5
          \   |   /
           \ v6  /
            \   /
             v7
    
    9 arêtes, 8 sommets, topologie complexe
    """
    graph = MetricGraph()
    
    # -----------------------------
    # Positions des sommets
    # -----------------------------
    positions = {
        "v0": (0.0, 0.5),    # gauche centre
        "v1": (1.0, 1.0),    # centre haut
        "v2": (2.0, 0.5),    # droite centre
        "v3": (0.5, 2.0),    # haut gauche
        "v4": (1.5, 2.0),    # haut droite
        "v5": (3.0, 0.5),    # extrême droite (bord)
        "v6": (1.0, 0.0),    # centre bas
        "v7": (1.0, -1.0),   # bas (bord)
    }
    
    for v, (x, y) in positions.items():
        graph.set_vertex_position(v, x, y)
    
    # -----------------------------
    # Fonction pour calculer longueur
    # -----------------------------
    def edge_length(vA, vB):
        xA, yA = positions[vA]
        xB, yB = positions[vB]
        return np.sqrt((xB - xA)**2 + (yB - yA)**2)
    
    # -----------------------------
    # Arêtes du réseau (9 arêtes)
    # -----------------------------
    edges_def = [
        (0, "v0", "v1"),   # horizontal bas gauche
        (1, "v1", "v2"),   # horizontal centre
        (2, "v2", "v5"),   # vers extrême droite
        (3, "v0", "v3"),   # diagonale haut gauche
        (4, "v1", "v3"),   # vertical gauche
        (5, "v1", "v4"),   # vertical droite
        (6, "v2", "v4"),   # diagonale haut droite
        (7, "v0", "v6"),   # vers bas gauche
        (8, "v1", "v6"),   # vers bas centre
    ]
    
    n_points = 150  # résolution fine
    
    for eid, vA, vB in edges_def:
        graph.add_edge(
            eid, vA, vB,
            length=edge_length(vA, vB),
            a_coef=1.0,
            n_points=n_points
        )
    
    # -----------------------------
    # Conditions aux limites
    # -----------------------------
    graph.set_boundary_vertices(["v3", "v4", "v5", "v7"])
    
    graph.build_dof_map()
    
    return graph


# ============================================================
# TEST INVERSE - CAS COMPLEXE (5 SOURCES)
# ============================================================

def test_inverse_complex_five_sources():
    """
    CAS TEST COMPLEXE – LOCALISATION DE 5 SOURCES
    ✔ graphe réseau complexe (9 arêtes)
    ✔ 5 sources à différentes positions
    ✔ topologie couplée (plusieurs chemins)
    ✔ comparaison CG vs PG vs UZAWA
    """
    
    print("\n" + "=" * 80)
    print("CAS TEST COMPLEXE – LOCALISATION DE 5 SOURCES")
    print("=" * 80)
    
    # ============================================================
    # 1. GRAPHE COMPLEXE
    # ============================================================
    graph = create_complex_network()
    graph.plot_graph(title="Graphe complexe – problème inverse (5 sources)")
    
    solver = SourceLocalization(graph)
    
    print(f"\nCaractéristiques du graphe :")
    print(f"  Nombre d'arêtes  : {len(graph.edges)}")
    print(f"  Nombre de DDL    : {graph.n_dof}")
    print(f"  Topologie        : Réseau couplé")
    
    # ============================================================
    # 2. SOURCES EXACTES (5 sources sur différentes arêtes)
    # ============================================================
    edge_ids = [0, 2, 4, 6, 8]  # 5 arêtes avec sources
    source_intensity = 1.0
    
    # Positions relatives sur chaque arête (plus variées)
    epsilon_hat = np.array([0.3, 0.7, 0.5, 0.4, 0.6])  # ∈ (0,1)
    
    epsilon_true_dict = {
        eid: epsilon_hat[i] * graph.edges[eid]["length"]
        for i, eid in enumerate(edge_ids)
    }
    
    print("\n>>> Sources exactes (5 sources)")
    print(f"{'Arête':<8} {'Position ε̂':<15} {'Position ε':<15} {'Longueur L':<15}")
    print("-" * 60)
    for i, eid in enumerate(edge_ids):
        L = graph.edges[eid]["length"]
        print(f"{eid:<8} {epsilon_hat[i]:<15.3f} {epsilon_true_dict[eid]:<15.6f} {L:<15.6f}")
    
    # ============================================================
    # 3. DONNÉES OBSERVÉES (PROBLÈME DIRECT)
    # ============================================================
    print("\n>>> Génération des données observées (problème direct)")
    
    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)
    
    noise_level = 0.0  # commencer sans bruit
    np.random.seed(42)
    u_data = u_exact.copy()
    
    if noise_level > 0:
        u_data += noise_level * np.random.randn(len(u_data))
        print(f"Bruit ajouté : {noise_level*100:.1f}%")
    else:
        print("Aucun bruit (cas de validation)")
    
    # ============================================================
    # 4. VALIDATION DES GRADIENTS (ÉCHANTILLON)
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε (échantillon sur 2 arêtes)")
    print("=" * 80)
    
    # Valider seulement 2 arêtes pour ne pas surcharger l'affichage
    for eid in [edge_ids[0], edge_ids[-1]]:
        i = edge_ids.index(eid)
        eps_test = epsilon_true_dict[eid]
        
        print(f"\n--- Arête {eid} | ε = {eps_test:.6f} ---")
        
        results = solver.validate_gradient_three_methods_epsilon(
            edge_id=eid,
            epsilon=eps_test,
            u_data=u_data,
            source_intensity=source_intensity,
            alpha_fd=1e-7,
        )
        
        print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
        print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
        print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")
    
    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.3e}")
    
    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    # Point de départ : milieu de chaque arête
    epsilon_init = np.array([
        0.5 * graph.edges[eid]["length"] for eid in edge_ids
    ])
    
    print(f"\n>>> Point de départ (milieu des arêtes)")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε_init = {epsilon_init[i]:.6f}")
    
    # ------------------------------------------------------------
    # 5.1 Gradient projeté vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)
    
    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.15,  # pas plus petit pour convergence stable
        max_iter=400,
        tol=1e-8,
    )
    
    print("\nRésultat PG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_pg[i]:.6f}")
    
    solver.solve_direct(
        {edge_ids[i]: eps_pg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)
    
    # ------------------------------------------------------------
    # 5.2 Gradient conjugué vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)
    
    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=200,
        tol=1e-8,
    )
    
    print("\nRésultat CG vectoriel:")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {eps_cg[i]:.6f}")
    
    solver.solve_direct(
        {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)
    
    # ------------------------------------------------------------
    # 5.3 UZAWA avec différentes contraintes
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (plusieurs contraintes)")
    print("=" * 80)
    
    # Contrainte lâche
    print("\n>>> Uzawa avec K_max = 4.0 (lâche)")
    eps_uzawa_4, lam_uzawa_4 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=4.0,
        step_init=0.4,
        rho=1.0,
        max_iter=600,
        tol=1e-10,
        verbose=False,
    )
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_4[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_4 = solver.compute_cost_functional(u_data)
    
    print(f"K_max=4.0 : ", end="")
    for i in range(len(edge_ids)):
        print(f"ε{i}={eps_uzawa_4[i]:.4f}, ", end="")
    print(f"λ={lam_uzawa_4:.3e}, J={J_uzawa_4:.3e}")
    
    # Contrainte modérée
    print("\n>>> Uzawa avec K_max = 3.0 (modéré)")
    eps_uzawa_3, lam_uzawa_3 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=3.0,
        step_init=0.35,
        rho=1.5,
        max_iter=600,
        tol=1e-10,
        verbose=True,  # affichage détaillé pour voir la convergence
    )
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_3[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_3 = solver.compute_cost_functional(u_data)
    
    # Contrainte stricte
    print("\n>>> Uzawa avec K_max = 2.0 (strict)")
    eps_uzawa_2, lam_uzawa_2 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=2.0,
        step_init=0.3,
        rho=2.0,
        max_iter=600,
        tol=1e-10,
        verbose=False,
    )
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_2[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_2 = solver.compute_cost_functional(u_data)
    
    print(f"K_max=2.0 : ", end="")
    for i in range(len(edge_ids)):
        print(f"ε{i}={eps_uzawa_2[i]:.4f}, ", end="")
    print(f"λ={lam_uzawa_2:.3e}, J={J_uzawa_2:.3e}")
    
    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)
    
    epsilon_true_vec = np.array([epsilon_true_dict[eid] for eid in edge_ids])
    L_vec = np.array([graph.edges[eid]["length"] for eid in edge_ids])
    
    print("\nSOLUTION EXACTE :")
    for i, eid in enumerate(edge_ids):
        print(f"  Arête {eid} : ε = {epsilon_true_vec[i]:.6f}")
    print(f"  Sparsité L1 : {sum(epsilon_true_vec / L_vec):.4f}")
    
    print("\nRÉSULTATS :")
    print(f"{'Méthode':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{'ε'+str(i):<10} ", end="")
    print(f"{'Sparsité':<12} {'J':<12}")
    print("-" * (20 + 10*len(edge_ids) + 24))
    
    # CG
    sparsity_cg = sum(eps_cg / L_vec)
    print(f"{'CG':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{eps_cg[i]:<10.4f} ", end="")
    print(f"{sparsity_cg:<12.4f} {J_cg:<12.3e}")
    
    # PG
    sparsity_pg = sum(eps_pg / L_vec)
    print(f"{'PG':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{eps_pg[i]:<10.4f} ", end="")
    print(f"{sparsity_pg:<12.4f} {J_pg:<12.3e}")
    
    # Uzawa K=4.0
    sparsity_u4 = sum(eps_uzawa_4 / L_vec)
    print(f"{'Uzawa (K=4.0)':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{eps_uzawa_4[i]:<10.4f} ", end="")
    print(f"{sparsity_u4:<12.4f} {J_uzawa_4:<12.3e}")
    
    # Uzawa K=3.0
    sparsity_u3 = sum(eps_uzawa_3 / L_vec)
    print(f"{'Uzawa (K=3.0)':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{eps_uzawa_3[i]:<10.4f} ", end="")
    print(f"{sparsity_u3:<12.4f} {J_uzawa_3:<12.3e}")
    
    # Uzawa K=2.0
    sparsity_u2 = sum(eps_uzawa_2 / L_vec)
    print(f"{'Uzawa (K=2.0)':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{eps_uzawa_2[i]:<10.4f} ", end="")
    print(f"{sparsity_u2:<12.4f} {J_uzawa_2:<12.3e}")
    
    # Analyse des erreurs
    print("\nERREURS PAR RAPPORT À LA SOLUTION EXACTE :")
    print(f"{'Méthode':<20} ", end="")
    for i in range(len(edge_ids)):
        print(f"{'Err ε'+str(i):<12} ", end="")
    print(f"{'Err moy':<12} {'Err max':<12}")
    print("-" * (20 + 12*len(edge_ids) + 24))
    
    def print_errors(name, eps):
        errs = np.abs(eps - epsilon_true_vec)
        print(f"{name:<20} ", end="")
        for err in errs:
            print(f"{err:<12.4f} ", end="")
        print(f"{np.mean(errs):<12.4f} {np.max(errs):<12.4f}")
    
    print_errors("CG", eps_cg)
    print_errors("PG", eps_pg)
    print_errors("Uzawa (K=4.0)", eps_uzawa_4)
    print_errors("Uzawa (K=3.0)", eps_uzawa_3)
    print_errors("Uzawa (K=2.0)", eps_uzawa_2)
    
    # Analyse statistique
    print("\n" + "=" * 80)
    print("ANALYSE STATISTIQUE")
    print("=" * 80)
    
    for method_name, eps in [("CG", eps_cg), ("PG", eps_pg), 
                              ("Uzawa K=4.0", eps_uzawa_4),
                              ("Uzawa K=3.0", eps_uzawa_3),
                              ("Uzawa K=2.0", eps_uzawa_2)]:
        errs = np.abs(eps - epsilon_true_vec)
        err_rel = errs / epsilon_true_vec
        print(f"\n{method_name} :")
        print(f"  Erreur absolue  : moy={np.mean(errs):.4f}, max={np.max(errs):.4f}, std={np.std(errs):.4f}")
        print(f"  Erreur relative : moy={np.mean(err_rel)*100:.2f}%, max={np.max(err_rel)*100:.2f}%")
    
    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n>>> Visualisations finales (solution CG)")
    
    eps_dict_cg = {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))}
    
    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)
    
    solver.plot_all_results(eps_dict_cg, u_data)
    
    return {
        "epsilon_true": epsilon_true_dict,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa_4": eps_uzawa_4,
        "epsilon_uzawa_3": eps_uzawa_3,
        "epsilon_uzawa_2": eps_uzawa_2,
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
        "J_uzawa_4": J_uzawa_4,
        "J_uzawa_3": J_uzawa_3,
        "J_uzawa_2": J_uzawa_2,
    }
# ============================================================
# GRAPHE TRÈS COMPLEXE - RÉSEAU DENSE (20 ARÊTES)
# ============================================================

def create_very_complex_network():
    """
    Graphe très complexe en réseau dense (type vascularisation)
    
    Structure en grille étendue avec connexions diagonales
    20 arêtes, 12 sommets, topologie fortement couplée
    
    v8 --- v9 --- v10 --- v11
    |  \   |  \   |   /   |
    |   \  |   \  |  /    |
    v4 --- v5 --- v6 --- v7
    |  \   |  \   |   /   |
    |   \  |   \  |  /    |
    v0 --- v1 --- v2 --- v3
    """
    graph = MetricGraph()
    
    # -----------------------------
    # Positions des sommets (grille 3x4)
    # -----------------------------
    positions = {
        # Ligne du bas
        "v0": (0.0, 0.0),
        "v1": (1.0, 0.0),
        "v2": (2.0, 0.0),
        "v3": (3.0, 0.0),
        
        # Ligne du milieu
        "v4": (0.0, 1.0),
        "v5": (1.0, 1.0),
        "v6": (2.0, 1.0),
        "v7": (3.0, 1.0),
        
        # Ligne du haut
        "v8": (0.0, 2.0),
        "v9": (1.0, 2.0),
        "v10": (2.0, 2.0),
        "v11": (3.0, 2.0),
    }
    
    for v, (x, y) in positions.items():
        graph.set_vertex_position(v, x, y)
    
    # -----------------------------
    # Fonction pour calculer longueur
    # -----------------------------
    def edge_length(vA, vB):
        xA, yA = positions[vA]
        xB, yB = positions[vB]
        return np.sqrt((xB - xA)**2 + (yB - yA)**2)
    
    # -----------------------------
    # Arêtes du réseau (20 arêtes)
    # -----------------------------
    edges_def = [
        # Ligne du bas (horizontales)
        (0, "v0", "v1"),
        (1, "v1", "v2"),
        (2, "v2", "v3"),
        
        # Ligne du milieu (horizontales)
        (3, "v4", "v5"),
        (4, "v5", "v6"),
        (5, "v6", "v7"),
        
        # Ligne du haut (horizontales)
        (6, "v8", "v9"),
        (7, "v9", "v10"),
        (8, "v10", "v11"),
        
        # Verticales
        (9, "v0", "v4"),
        (10, "v1", "v5"),
        (11, "v2", "v6"),
        (12, "v3", "v7"),
        (13, "v4", "v8"),
        (14, "v5", "v9"),
        (15, "v6", "v10"),
        (16, "v7", "v11"),
        
        # Diagonales (connexions croisées)
        (17, "v0", "v5"),  # bas-gauche vers centre
        (18, "v5", "v10"), # centre vers haut-centre
        (19, "v6", "v11"), # centre-droite vers haut-droite
    ]
    
    n_points = 120  # résolution fine pour 20 arêtes
    
    for eid, vA, vB in edges_def:
        graph.add_edge(
            eid, vA, vB,
            length=edge_length(vA, vB),
            a_coef=1.0,
            n_points=n_points
        )
    
    # -----------------------------
    # Conditions aux limites (sommets périphériques)
    # -----------------------------
    graph.set_boundary_vertices(["v0", "v3", "v8", "v11"])
    
    graph.build_dof_map()
    
    return graph


# ============================================================
# TEST INVERSE - CAS EXTRÊME (10 SOURCES SUR 20 ARÊTES)
# ============================================================

def test_inverse_extreme_ten_sources():
    """
    CAS TEST EXTRÊME – LOCALISATION DE 10 SOURCES SUR 20 ARÊTES
    ✔ graphe réseau très dense (20 arêtes, 12 sommets)
    ✔ 10 sources à positions variées
    ✔ topologie fortement couplée
    ✔ comparaison CG vs PG vs UZAWA (3 contraintes)
    ✔ analyse de performance et robustesse
    """
    
    print("\n" + "=" * 80)
    print("CAS TEST EXTRÊME – LOCALISATION DE 10 SOURCES SUR 20 ARÊTES")
    print("=" * 80)
    
    # ============================================================
    # 1. GRAPHE TRÈS COMPLEXE
    # ============================================================
    graph = create_very_complex_network()
    graph.plot_graph(title="Réseau dense – problème inverse (10 sources sur 20 arêtes)")
    
    solver = SourceLocalization(graph)
    
    print(f"\nCaractéristiques du graphe :")
    print(f"  Nombre d'arêtes  : {len(graph.edges)}")
    print(f"  Nombre de sommets: {len(set([e['v_start'] for e in graph.edges] + [e['v_end'] for e in graph.edges]))}")
    print(f"  Nombre de DDL    : {graph.n_dof}")
    print(f"  Topologie        : Réseau dense fortement couplé")
    
    # ============================================================
    # 2. SOURCES EXACTES (10 sources sur différentes arêtes)
    # ============================================================
    # Sélection stratégique : mélange d'arêtes horizontales, verticales et diagonales
    edge_ids = [0, 2, 4, 6, 8, 10, 13, 17, 18, 19]  # 10 arêtes variées
    source_intensity = 1.0
    
    # Positions relatives variées (pattern non uniforme)
    epsilon_hat = np.array([0.25, 0.40, 0.55, 0.70, 0.35, 
                            0.60, 0.45, 0.30, 0.65, 0.50])  # ∈ (0,1)
    
    epsilon_true_dict = {
        eid: epsilon_hat[i] * graph.edges[eid]["length"]
        for i, eid in enumerate(edge_ids)
    }
    
    print("\n>>> Sources exactes (10 sources)")
    print(f"{'Arête':<8} {'Type':<12} {'Position ε̂':<15} {'Position ε':<15} {'Longueur L':<15}")
    print("-" * 75)
    
    edge_types = {
        0: "Horiz-bas", 2: "Horiz-bas", 4: "Horiz-mid", 
        6: "Horiz-haut", 8: "Horiz-haut", 10: "Vertical",
        13: "Vertical", 17: "Diagonale", 18: "Diagonale", 19: "Diagonale"
    }
    
    for i, eid in enumerate(edge_ids):
        L = graph.edges[eid]["length"]
        edge_type = edge_types.get(eid, "Autre")
        print(f"{eid:<8} {edge_type:<12} {epsilon_hat[i]:<15.3f} {epsilon_true_dict[eid]:<15.6f} {L:<15.6f}")
    
    # ============================================================
    # 3. DONNÉES OBSERVÉES (PROBLÈME DIRECT)
    # ============================================================
    print("\n>>> Génération des données observées (problème direct)")
    
    import time
    start_time = time.time()
    u_exact = solver.solve_direct(epsilon_true_dict, source_intensity)
    direct_time = time.time() - start_time
    
    print(f"Temps de résolution directe : {direct_time:.3f} s")
    
    noise_level = 0.0  # commencer sans bruit
    np.random.seed(123)
    u_data = u_exact.copy()
    
    if noise_level > 0:
        u_data += noise_level * np.random.randn(len(u_data))
        print(f"Bruit ajouté : {noise_level*100:.1f}%")
    else:
        print("Aucun bruit (validation exacte)")
    
    # ============================================================
    # 4. VALIDATION DES GRADIENTS (ÉCHANTILLON)
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION DES GRADIENTS dJ/dε (échantillon sur 3 arêtes)")
    print("=" * 80)
    
    # Valider 3 arêtes : une horizontale, une verticale, une diagonale
    validation_edges = [edge_ids[0], edge_ids[5], edge_ids[-1]]
    
    for eid in validation_edges:
        i = edge_ids.index(eid)
        eps_test = epsilon_true_dict[eid]
        edge_type = edge_types.get(eid, "Autre")
        
        print(f"\n--- Arête {eid} ({edge_type}) | ε = {eps_test:.6f} ---")
        
        results = solver.validate_gradient_three_methods_epsilon(
            edge_id=eid,
            epsilon=eps_test,
            u_data=u_data,
            source_intensity=source_intensity,
            alpha_fd=1e-7,
        )
        
        print(f"  dJ/dε (DF)   = {results['grad_fd']:.6e}")
        print(f"  dJ/dε (Sens) = {results['grad_sensitivity']:.6e}")
        print(f"  dJ/dε (Adj)  = {results['grad_adjoint']:.6e}")
        
        if abs(results['grad_sensitivity']) > 1e-15:
            err_rel = abs(results['grad_adjoint'] - results['grad_sensitivity']) / abs(results['grad_sensitivity'])
            print(f"  Erreur Sens/Adj : {err_rel:.3e}")
    
    solver.solve_direct(epsilon_true_dict, source_intensity)
    J_ref = solver.compute_cost_functional(u_data)
    print(f"\nValeur du coût J (référence) = {J_ref:.6e}")
    
    # ============================================================
    # 5. OPTIMISATION VECTORIELLE
    # ============================================================
    # Point de départ : positions aléatoires entre 0.3 et 0.7
    np.random.seed(456)
    epsilon_init = np.array([
        (0.3 + 0.4 * np.random.rand()) * graph.edges[eid]["length"] 
        for eid in edge_ids
    ])
    
    print(f"\n>>> Point de départ (positions aléatoires)")
    for i, eid in enumerate(edge_ids):
        eps_hat_init = epsilon_init[i] / graph.edges[eid]["length"]
        print(f"  Arête {eid} : ε̂_init = {eps_hat_init:.3f}, ε_init = {epsilon_init[i]:.6f}")
    
    # ------------------------------------------------------------
    # 5.1 Gradient conjugué vectoriel (référence)
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT CONJUGUÉ VECTORIEL")
    print("=" * 80)
    
    start_time = time.time()
    eps_cg = solver.conjugate_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        max_iter=250,
        tol=1e-9,
    )
    cg_time = time.time() - start_time
    
    print(f"\nTemps d'optimisation CG : {cg_time:.2f} s")
    print("\nRésultat CG vectoriel:")
    for i, eid in enumerate(edge_ids):
        eps_hat_cg = eps_cg[i] / graph.edges[eid]["length"]
        print(f"  Arête {eid} : ε̂ = {eps_hat_cg:.4f}, ε = {eps_cg[i]:.6f}")
    
    solver.solve_direct(
        {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_cg = solver.compute_cost_functional(u_data)
    
    # ------------------------------------------------------------
    # 5.2 Gradient projeté vectoriel
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – GRADIENT PROJETÉ VECTORIEL")
    print("=" * 80)
    
    start_time = time.time()
    eps_pg = solver.projected_gradient_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        step=0.12,  # pas adapté au nombre de sources
        max_iter=500,
        tol=1e-9,
    )
    pg_time = time.time() - start_time
    
    print(f"\nTemps d'optimisation PG : {pg_time:.2f} s")
    print("\nRésultat PG vectoriel:")
    for i, eid in enumerate(edge_ids):
        eps_hat_pg = eps_pg[i] / graph.edges[eid]["length"]
        print(f"  Arête {eid} : ε̂ = {eps_hat_pg:.4f}, ε = {eps_pg[i]:.6f}")
    
    solver.solve_direct(
        {edge_ids[i]: eps_pg[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_pg = solver.compute_cost_functional(u_data)
    
    # ------------------------------------------------------------
    # 5.3 UZAWA avec trois niveaux de contrainte
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OPTIMISATION – UZAWA (trois niveaux de contrainte)")
    print("=" * 80)
    
    # Contrainte lâche
    print("\n>>> Uzawa avec K_max = 8.0 (très lâche)")
    start_time = time.time()
    eps_uzawa_8, lam_uzawa_8 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=8.0,
        step_init=0.35,
        rho=1.0,
        max_iter=700,
        tol=1e-10,
        verbose=False,
    )
    uzawa_8_time = time.time() - start_time
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_8[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_8 = solver.compute_cost_functional(u_data)
    
    print(f"Temps : {uzawa_8_time:.2f} s | λ = {lam_uzawa_8:.3e} | J = {J_uzawa_8:.3e}")
    
    # Contrainte modérée
    print("\n>>> Uzawa avec K_max = 5.0 (modéré)")
    start_time = time.time()
    eps_uzawa_5, lam_uzawa_5 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=5.0,
        step_init=0.3,
        rho=1.5,
        max_iter=700,
        tol=1e-10,
        verbose=True,  # affichage détaillé
    )
    uzawa_5_time = time.time() - start_time
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_5[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_5 = solver.compute_cost_functional(u_data)
    
    print(f"Temps : {uzawa_5_time:.2f} s | λ = {lam_uzawa_5:.3e} | J = {J_uzawa_5:.3e}")
    
    # Contrainte stricte
    print("\n>>> Uzawa avec K_max = 3.0 (strict)")
    start_time = time.time()
    eps_uzawa_3, lam_uzawa_3 = solver.uzawa_epsilon_vector(
        epsilon_init=epsilon_init,
        edge_ids=edge_ids,
        u_data=u_data,
        source_intensity=source_intensity,
        K_max=3.0,
        step_init=0.25,
        rho=2.0,
        max_iter=700,
        tol=1e-10,
        verbose=False,
    )
    uzawa_3_time = time.time() - start_time
    
    solver.solve_direct(
        {edge_ids[i]: eps_uzawa_3[i] for i in range(len(edge_ids))},
        source_intensity,
    )
    J_uzawa_3 = solver.compute_cost_functional(u_data)
    
    print(f"Temps : {uzawa_3_time:.2f} s | λ = {lam_uzawa_3:.3e} | J = {J_uzawa_3:.3e}")
    
    # ============================================================
    # 6. COMPARAISON DÉTAILLÉE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)
    
    epsilon_true_vec = np.array([epsilon_true_dict[eid] for eid in edge_ids])
    L_vec = np.array([graph.edges[eid]["length"] for eid in edge_ids])
    
    print("\nSOLUTION EXACTE :")
    for i, eid in enumerate(edge_ids):
        eps_hat_true = epsilon_true_vec[i] / L_vec[i]
        print(f"  Arête {eid:2d} ({edge_types.get(eid, 'Autre'):<11}) : ε̂ = {eps_hat_true:.4f}, ε = {epsilon_true_vec[i]:.6f}")
    print(f"\n  Sparsité L1 exacte : {sum(epsilon_true_vec / L_vec):.4f}")
    
    # Tableau récapitulatif
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 80)
    
    methods = [
        ("CG", eps_cg, J_cg, cg_time),
        ("PG", eps_pg, J_pg, pg_time),
        ("Uzawa K=8.0", eps_uzawa_8, J_uzawa_8, uzawa_8_time),
        ("Uzawa K=5.0", eps_uzawa_5, J_uzawa_5, uzawa_5_time),
        ("Uzawa K=3.0", eps_uzawa_3, J_uzawa_3, uzawa_3_time),
    ]
    
    print(f"\n{'Méthode':<20} {'Sparsité L1':<15} {'Coût J':<15} {'Temps (s)':<12} {'Err. moy':<12} {'Err. max':<12}")
    print("-" * 95)
    
    for method_name, eps, J, exec_time in methods:
        sparsity = sum(eps / L_vec)
        errs = np.abs(eps - epsilon_true_vec)
        err_mean = np.mean(errs)
        err_max = np.max(errs)
        print(f"{method_name:<20} {sparsity:<15.4f} {J:<15.6e} {exec_time:<12.2f} {err_mean:<12.4f} {err_max:<12.4f}")
    
    # Erreurs détaillées par arête
    print("\n" + "=" * 80)
    print("ERREURS DÉTAILLÉES PAR ARÊTE")
    print("=" * 80)
    
    print(f"\n{'Arête':<8} {'Type':<12} {'ε exact':<12} ", end="")
    for method_name, _, _, _ in methods:
        print(f"{method_name[:6]:<12} ", end="")
    print()
    print("-" * (32 + 12 * len(methods)))
    
    for i, eid in enumerate(edge_ids):
        edge_type = edge_types.get(eid, "Autre")
        print(f"{eid:<8} {edge_type:<12} {epsilon_true_vec[i]:<12.6f} ", end="")
        for _, eps, _, _ in methods:
            err = abs(eps[i] - epsilon_true_vec[i])
            print(f"{err:<12.6f} ", end="")
        print()
    
    # Analyse statistique complète
    print("\n" + "=" * 80)
    print("ANALYSE STATISTIQUE COMPLÈTE")
    print("=" * 80)
    
    for method_name, eps, J, exec_time in methods:
        errs = np.abs(eps - epsilon_true_vec)
        err_rel = errs / epsilon_true_vec
        
        print(f"\n{method_name} :")
        print(f"  Coût J          : {J:.6e}")
        print(f"  Réduction J     : {(J_ref - J)/J_ref * 100:.2f}%")
        print(f"  Sparsité L1     : {sum(eps / L_vec):.4f}")
        print(f"  Temps exec.     : {exec_time:.2f} s")
        print(f"  Erreur absolue  : moy={np.mean(errs):.4f}, médiane={np.median(errs):.4f}, max={np.max(errs):.4f}, std={np.std(errs):.4f}")
        print(f"  Erreur relative : moy={np.mean(err_rel)*100:.2f}%, max={np.max(err_rel)*100:.2f}%")
        print(f"  Erreur quadr.   : {np.sqrt(np.mean(errs**2)):.4f}")
    
    # ============================================================
    # 7. VISUALISATIONS FINALES (CG)
    # ============================================================
    print("\n" + "=" * 80)
    print("VISUALISATIONS FINALES (solution CG)")
    print("=" * 80)
    
    eps_dict_cg = {edge_ids[i]: eps_cg[i] for i in range(len(edge_ids))}
    
    solver.solve_direct(eps_dict_cg, source_intensity)
    solver.solve_adjoint(eps_dict_cg, u_data, source_intensity)
    solver.solve_sensitivity_epsilon(eps_dict_cg, source_intensity)
    
    solver.plot_all_results(eps_dict_cg, u_data)
    
    # Visualisation comparative des positions
    plot_source_comparison(graph, edge_ids, epsilon_true_dict, eps_dict_cg, edge_types)
    
    return {
        "graph": graph,
        "epsilon_true": epsilon_true_dict,
        "epsilon_init": epsilon_init,
        "epsilon_pg": eps_pg,
        "epsilon_cg": eps_cg,
        "epsilon_uzawa_8": eps_uzawa_8,
        "epsilon_uzawa_5": eps_uzawa_5,
        "epsilon_uzawa_3": eps_uzawa_3,
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
        "J_uzawa_8": J_uzawa_8,
        "J_uzawa_5": J_uzawa_5,
        "J_uzawa_3": J_uzawa_3,
        "times": {
            "direct": direct_time,
            "pg": pg_time,
            "cg": cg_time,
            "uzawa_8": uzawa_8_time,
            "uzawa_5": uzawa_5_time,
            "uzawa_3": uzawa_3_time,
        }
    }


def plot_source_comparison(graph, edge_ids, epsilon_true_dict, epsilon_identified_dict, edge_types):
    """
    Visualise la comparaison entre sources exactes et identifiées
    
    Args:
        graph: MetricGraph
        edge_ids: liste des IDs d'arêtes avec sources
        epsilon_true_dict: positions exactes {edge_id: epsilon}
        epsilon_identified_dict: positions identifiées {edge_id: epsilon}
        edge_types: dictionnaire {edge_id: type}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Graphique 1 : Comparaison des positions ε̂
    epsilon_hat_true = []
    epsilon_hat_identified = []
    labels = []
    
    for eid in edge_ids:
        L = graph.edges[eid]["length"]
        epsilon_hat_true.append(epsilon_true_dict[eid] / L)
        epsilon_hat_identified.append(epsilon_identified_dict[eid] / L)
        labels.append(f"{eid}\n{edge_types.get(eid, 'Autre')[:4]}")
    
    x = np.arange(len(edge_ids))
    width = 0.35
    
    ax1.bar(x - width/2, epsilon_hat_true, width, label='Exact', alpha=0.8, color='green')
    ax1.bar(x + width/2, epsilon_hat_identified, width, label='Identifié (CG)', alpha=0.8, color='orange')
    
    ax1.set_xlabel('Arête', fontsize=12)
    ax1.set_ylabel('Position relative ε̂', fontsize=12)
    ax1.set_title('Comparaison positions relatives (10 sources)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Graphique 2 : Erreurs absolues
    errors = np.abs(np.array(epsilon_hat_identified) - np.array(epsilon_hat_true))
    colors = ['red' if err > 0.05 else 'orange' if err > 0.02 else 'green' for err in errors]
    
    ax2.bar(x, errors, color=colors, alpha=0.7)
    ax2.axhline(y=0.02, color='orange', linestyle='--', linewidth=1.5, label='Seuil 2%', alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='Seuil 5%', alpha=0.7)
    
    ax2.set_xlabel('Arête', fontsize=12)
    ax2.set_ylabel('Erreur absolue |ε̂_id - ε̂_exact|', fontsize=12)
    ax2.set_title('Erreurs d\'identification par arête', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    print(f"\nSTATISTIQUES D'ERREUR :")
    print(f"  Erreur moyenne : {np.mean(errors):.4f}")
    print(f"  Erreur médiane : {np.median(errors):.4f}")
    print(f"  Erreur maximale: {np.max(errors):.4f}")
    print(f"  Écart-type     : {np.std(errors):.4f}")
    print(f"  Nb erreurs > 5%: {sum(1 for e in errors if e > 0.05)}/{len(errors)}")
    print(f"  Nb erreurs > 2%: {sum(1 for e in errors if e > 0.02)}/{len(errors)}")