
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(filepath):
    """Charge les résultats depuis le fichier"""
    data = np.genfromtxt(filepath, skip_header=1, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    rho = data[:, 2]
    u = data[:, 3]
    v = data[:, 4]
    p = data[:, 5]
    
    # Déterminer la taille de la grille
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    
    # Reconstruction 2D
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    rho2d = rho.reshape(ny, nx)
    u2d = u.reshape(ny, nx)
    v2d = v.reshape(ny, nx)
    p2d = p.reshape(ny, nx)
    
    return X, Y, rho2d, u2d, v2d, p2d

def get_case_info(case_num):
    """Retourne les informations spécifiques au cas"""
    case_info = {
        3: {
            'name': 'Case 3: S, S, S, S (4 shocks)',
            'levels_rho': np.linspace(0.138, 1.5, 30),
            'vmin_rho': 0.138,
            'vmax_rho': 1.5
        },
        12: {
            'name': 'Case 12: J, S, S, J (contacts + shocks)',
            'levels_rho': np.arange(0.54, 1.71, 0.04),
            'vmin_rho': 0.54,
            'vmax_rho': 1.71
        },
        15: {
            'name': 'Case 15: J, R, S, J (mixed waves)',
            'levels_rho': np.linspace(0.5, 1.2, 30),
            'vmin_rho': 0.5,
            'vmax_rho': 1.2
        }
    }
    return case_info.get(case_num, case_info[12])

def select_case(folder="./resultats/riemann/"):
    """Détecte les cas disponibles et demande à l'utilisateur de choisir"""
    result_files = sorted(Path(folder).glob("case*_t*.txt"))
    
    if not result_files:
        print(f" Aucun fichier trouvé dans {folder}")
        return None
    
    # Extraire les numéros de cas disponibles
    available_cases = set()
    for f in result_files:
        filename = f.stem  # ex: "case12_t0.250000"
        case_num = int(filename.split('_')[0].replace('case', ''))
        available_cases.add(case_num)
    
    available_cases = sorted(available_cases)
    
    print("\n" + "="*70)
    print("CAS DISPONIBLES DANS LE DOSSIER")
    print("="*70)
    for case in available_cases:
        case_info = get_case_info(case)
        print(f"  {case} : {case_info['name']}")
    print("="*70)
    
    while True:
        try:
            case_choice = int(input(f"Choisir le cas à visualiser ({', '.join(map(str, available_cases))}): "))
            if case_choice in available_cases:
                return case_choice
            else:
                print(f" Cas invalide. Choisissez parmi: {available_cases}")
        except ValueError:
            print("Entrée invalide. Entrez un numéro de cas.")

def plot_results(case_num=None, folder="./resultats/riemann/"):
    """Visualise la solution numérique pour un cas donné"""
    
    # Si pas de cas spécifié, demander à l'utilisateur
    if case_num is None:
        case_num = select_case(folder)
        if case_num is None:
            return
    
    # Chercher les fichiers pour ce cas spécifique
    pattern = f"case{case_num}_t*.txt"
    result_files = sorted(Path(folder).glob(pattern))
    
    if not result_files:
        print(f" Aucun fichier trouvé pour le cas {case_num} dans {folder}")
        return
    
    # Prendre le dernier fichier (temps final)
    file_final = result_files[-1]
    
    print("\n" + "="*70)
    print(f"VISUALISATION RIEMANN 2D - CASE {case_num}")
    print("="*70)
    print(f" Fichier: {file_final.name}")
    print(f" Nombre de snapshots trouvés: {len(result_files)}")
    
    # Charger solution numérique
    print("Chargement des données...")
    X, Y, rho, u, v, p = load_results(file_final)
    print(f"  ✓ Grille: {X.shape[1]} × {X.shape[0]}")
    
    # Obtenir les paramètres du cas
    case_params = get_case_info(case_num)
    levels_rho = case_params['levels_rho']
    
    # ========== FIGURE 1 : Densité seule ==========
    fig1 = plt.figure(figsize=(10, 9))
    
    ax1 = plt.subplot(111)
    cf1 = ax1.contourf(X, Y, rho, levels=50, cmap='viridis')
    ax1.contour(X, Y, rho, levels=levels_rho, colors='white', linewidths=0.5, alpha=0.6)
    ax1.set_title(f'Solution numérique WENO5 - Densité\n{case_params["name"]}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_aspect('equal')
    ax1.plot([0.5, 0.5], [0, 1], 'w--', linewidth=1, alpha=0.5)
    ax1.plot([0, 1], [0.5, 0.5], 'w--', linewidth=1, alpha=0.5)
    plt.colorbar(cf1, ax=ax1, label='ρ')
    
    plt.tight_layout()
    output_file1 = Path(folder) / f'riemann2d_case{case_num}_density.png'
    plt.savefig(output_file1, dpi=200, bbox_inches='tight')
    print(f"\n✓ Figure sauvegardée: {output_file1}")
    plt.show()
    
    # ========== FIGURE 2 : Pression + densité + vitesse ==========
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    
    levels_p = np.linspace(p.min(), p.max(), 50)
    cf2 = ax2.contourf(X, Y, p, levels=levels_p, cmap='jet', extend='both')
    ax2.contour(X, Y, rho, levels=levels_rho, colors='black', 
                linewidths=0.8, linestyles='solid')
    
    # Champ de vitesse
    skip = max(1, X.shape[0] // 25)
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u[::skip, ::skip], v[::skip, ::skip],
               scale=15, width=0.003, color='white', alpha=0.7)
    
    cbar = plt.colorbar(cf2, ax=ax2, label='Pression', orientation='vertical')
    cbar.ax.tick_params(labelsize=10)
    
    ax2.set_xlabel('x', fontsize=13, fontweight='bold')
    ax2.set_ylabel('y', fontsize=13, fontweight='bold')
    ax2.set_title(f'2D Riemann Problem - WENO5\n{case_params["name"]}\n' + 
                  'Pressure (color) + Density (contours) + Velocity',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.plot([0.5, 0.5], [0, 1], 'k--', linewidth=1.5, alpha=0.3)
    ax2.plot([0, 1], [0.5, 0.5], 'k--', linewidth=1.5, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = Path(folder) / f'riemann2d_case{case_num}_result.png'
    plt.savefig(output_file2, dpi=200, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_file2}")
    plt.show()
    
    # ========== FIGURE 3 : Coupes 1D ==========
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coupe horizontale à y=0.5 (densité)
    j_mid = Y.shape[0] // 2
    ax_h = axes[0, 0]
    ax_h.plot(X[j_mid, :], rho[j_mid, :], 'b-', linewidth=2)
    ax_h.set_xlabel('x', fontsize=11, fontweight='bold')
    ax_h.set_ylabel('ρ', fontsize=11, fontweight='bold')
    ax_h.set_title('Densité - Coupe horizontale (y=0.5)', fontsize=12, fontweight='bold')
    ax_h.grid(True, alpha=0.3)
    ax_h.axvline(0.5, color='k', linestyle='--', alpha=0.3)
    
    # Coupe verticale à x=0.5 (densité)
    i_mid = X.shape[1] // 2
    ax_v = axes[0, 1]
    ax_v.plot(Y[:, i_mid], rho[:, i_mid], 'b-', linewidth=2)
    ax_v.set_xlabel('y', fontsize=11, fontweight='bold')
    ax_v.set_ylabel('ρ', fontsize=11, fontweight='bold')
    ax_v.set_title('Densité - Coupe verticale (x=0.5)', fontsize=12, fontweight='bold')
    ax_v.grid(True, alpha=0.3)
    ax_v.axvline(0.5, color='k', linestyle='--', alpha=0.3)
    
    # Coupe horizontale (pression)
    ax_ph = axes[1, 0]
    ax_ph.plot(X[j_mid, :], p[j_mid, :], 'r-', linewidth=2)
    ax_ph.set_xlabel('x', fontsize=11, fontweight='bold')
    ax_ph.set_ylabel('p', fontsize=11, fontweight='bold')
    ax_ph.set_title('Pression - Coupe horizontale (y=0.5)', fontsize=12, fontweight='bold')
    ax_ph.grid(True, alpha=0.3)
    ax_ph.axvline(0.5, color='k', linestyle='--', alpha=0.3)
    
    # Coupe verticale (pression)
    ax_pv = axes[1, 1]
    ax_pv.plot(Y[:, i_mid], p[:, i_mid], 'r-', linewidth=2)
    ax_pv.set_xlabel('y', fontsize=11, fontweight='bold')
    ax_pv.set_ylabel('p', fontsize=11, fontweight='bold')
    ax_pv.set_title('Pression - Coupe verticale (x=0.5)', fontsize=12, fontweight='bold')
    ax_pv.grid(True, alpha=0.3)
    ax_pv.axvline(0.5, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_file3 = Path(folder) / f'riemann2d_case{case_num}_slices.png'
    plt.savefig(output_file3, dpi=200, bbox_inches='tight')
    print(f"✓ Figure sauvegardée: {output_file3}")
    plt.show()
    
    # Statistiques détaillées
    print("\n" + "="*70)
    print("STATISTIQUES DE LA SOLUTION")
    print("="*70)
    print(f"Densité    : min = {rho.min():.6f}, max = {rho.max():.6f}")
    print(f"Pression   : min = {p.min():.6f}, max = {p.max():.6f}")
    print(f"Vitesse u  : min = {u.min():.6f}, max = {u.max():.6f}")
    print(f"Vitesse v  : min = {v.min():.6f}, max = {v.max():.6f}")
    print("="*70)

if __name__ == "__main__":
    print("="*70)
    print("VISUALISATION RIEMANN 2D MULTI-CAS")
    print("="*70)
    
    # Le script demande automatiquement quel cas visualiser
    plot_results()
    
    print("\n✓ Visualisation terminée")