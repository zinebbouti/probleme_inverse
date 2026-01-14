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
    ✔ PG vs CG
    """

    print("\n" + "=" * 80)
    print("CAS TEST COMPLET – LOCALISATION D’UNE SOURCE (ε vectoriel)")
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

    # position relative sur l’arête (0 < ε̂ < 1)
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
        0.2 * graph.edges[edge_ids[0]]["length"]
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

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    print(
        f"Arête {edge_ids[0]} | "
        f"ε exact = {epsilon_true[0]:.6f} | "
        f"ε PG = {eps_pg[0]:.6f} | "
        f"ε CG = {eps_cg[0]:.6f}"
    )

    print("\nValeurs du coût :")
    print(f"  J référence = {J_ref:.3e}")
    print(f"  J PG        = {J_pg:.3e}")
    print(f"  J CG        = {J_cg:.3e}")

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
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
    }


def test_inverse_source_localization_two_sources_complete():
    """
    CAS TEST COMPLET – LOCALISATION DE DEUX SOURCES
    ✔ optimisation conjointe vectorielle
    ✔ gradient adjoint validé
    ✔ PG vectoriel vs CG vectoriel
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
    epsilon_init = np.array([0.2, 0.2])

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

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    for i, eid in enumerate(edge_ids):
        print(
            f"Arête {eid} | "
            f"ε exact = {epsilon_true[i]:.6f} | "
            f"ε PG = {eps_pg[i]:.6f} | "
            f"ε CG = {eps_cg[i]:.6f}"
        )

    print("\nValeurs du coût :")
    print(f"  J référence = {J_ref:.3e}")
    print(f"  J PG        = {J_pg:.3e}")
    print(f"  J CG        = {J_cg:.3e}")

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
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
    }

def test_inverse_three_sources_vectorial():
    """
    CAS TEST COMPLET – LOCALISATION DE TROIS SOURCES
    ✔ graphe étoile
    ✔ 1 source par arête (strictement sur l’arête)
    ✔ optimisation vectorielle conjointe
    ✔ Gradient projeté vs Gradient conjugué
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

    # sécurité
    for eid, eps in epsilon_true_dict.items():
        L = graph.edges[eid]["length"]
        assert 0.0 < eps < L, f"Source hors arête {eid}"

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
    epsilon_init = np.array([0.2, 0.2, 0.2])

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

    # ============================================================
    # 6. COMPARAISON FINALE
    # ============================================================
    print("\n" + "=" * 80)
    print("COMPARAISON DES MÉTHODES")
    print("=" * 80)

    for i, eid in enumerate(edge_ids):
        print(
            f"Arête {eid} | "
            f"ε exact = {epsilon_true_dict[eid]:.6f} | "
            f"ε PG = {eps_pg[i]:.6f} | "
            f"ε CG = {eps_cg[i]:.6f}"
        )

    print("\nValeurs du coût :")
    print(f"  J référence = {J_ref:.3e}")
    print(f"  J PG        = {J_pg:.3e}")
    print(f"  J CG        = {J_cg:.3e}")

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
        "J_ref": J_ref,
        "J_pg": J_pg,
        "J_cg": J_cg,
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
