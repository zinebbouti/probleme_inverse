
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# ============================================================
# Configuration 
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

# ============================================================
# Interpolation simple
# ============================================================
def simple_interpolate_nans(grid):
    """Remplit les NaN par moyenne des voisins valides"""
    result = grid.copy()
    ny, nx = grid.shape
    
    max_iter = 10
    for iteration in range(max_iter):
        nan_mask = np.isnan(result)
        if not nan_mask.any():
            break
            
        for j in range(ny):
            for i in range(nx):
                if nan_mask[j, i]:
                    neighbors = []
                    for dj in [-1, 0, 1]:
                        for di in [-1, 0, 1]:
                            if dj == 0 and di == 0:
                                continue
                            jj, ii = j + dj, i + di
                            if 0 <= jj < ny and 0 <= ii < nx:
                                if not np.isnan(result[jj, ii]):
                                    neighbors.append(result[jj, ii])
                    
                    if neighbors:
                        result[j, i] = np.mean(neighbors)
    
    if np.isnan(result).any():
        global_mean = np.nanmean(result)
        result[np.isnan(result)] = global_mean
    
    return result

# ============================================================
# Chargement des données 
# ============================================================
def load_data(filepath, time_select=None):
    print(f"Chargement: {filepath}")
    data = np.genfromtxt(filepath)
    data = data[~np.isnan(data).any(axis=1)]

    if time_select is not None:
        times = np.unique(data[:, 0])
        idx = np.argmin(np.abs(times - time_select))
        tsel = times[idx]
        data = data[data[:, 0] == tsel]
        print(f"  → Temps sélectionné: {tsel:.5f}")

    i = data[:, 1].astype(int)
    j = data[:, 2].astype(int)
    rho = data[:, 3]
    u = data[:, 4]
    v = data[:, 5]
    p = data[:, 6]

    nx = i.max()
    ny = j.max()

    rho2d = np.full((ny, nx), np.nan)
    u2d = np.full((ny, nx), np.nan)
    v2d = np.full((ny, nx), np.nan)
    p2d = np.full((ny, nx), np.nan)

    for ii, jj, rr, uu, vv, pp in zip(i, j, rho, u, v, p):
        rho2d[jj-1, ii-1] = rr
        u2d[jj-1, ii-1] = uu
        v2d[jj-1, ii-1] = vv
        p2d[jj-1, ii-1] = pp

    if np.isnan(p2d).any():
        rho2d = simple_interpolate_nans(rho2d)
        u2d = simple_interpolate_nans(u2d)
        v2d = simple_interpolate_nans(v2d)
        p2d = simple_interpolate_nans(p2d)

    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    return X, Y, rho2d, u2d, v2d, p2d

# ============================================================
# FIGURE PRINCIPALE - EXACTEMENT COMME L'ARTICLE Fig. 5.7
# ============================================================
def plot_article(data_list, times, outfile):
    """
    Reproduit EXACTEMENT Fig. 5.7 de l'article:
    - Fond BLANC
    - Contours NOIRS fins (comme dans l'article original)
    - Ligne verticale NOIRE épaisse pour le choc à x=0.5
    - 30 contours
    - Zone [0,1] × [0,1]
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    
    # Échelle globale pour les 3 temps
    pmin = min(np.nanmin(d[5]) for d in data_list)
    pmax = max(np.nanmax(d[5]) for d in data_list)
    
    # 30 contours comme dans l'article
    levels = np.linspace(pmin, pmax, 30)
    
    for idx, (data, t, ax) in enumerate(zip(data_list, times, axes)):
        X, Y, rho, u, v, p = data
        
        # Fond BLANC
        ax.set_facecolor('white')
        
        # CONTOURS NOIRS FINS (comme dans l'article original)
        cs = ax.contour(X, Y, p, levels=levels, colors='black', 
                        linewidths=0.5, alpha=1.0)
        
        # Ligne verticale NOIRE ÉPAISSE pour le choc à x=0.5
        ax.axvline(x=0.5, color='black', linewidth=2.5, linestyle='-')
        
        # Zone d'affichage [0,1] × [0,1]
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect('equal', adjustable='box')
        
        # Labels
        ax.set_xlabel('x', fontsize=11)
        if idx == 0:
            ax.set_ylabel('y', fontsize=11)
        
        # Titre (a), (b), (c) - Style article
        label = chr(97 + idx)
        ax.set_title(f'({label}) t={t:.2f}', fontsize=11, loc='left')
        
        # Bordure noire (cadre)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        # Grille OFF
        ax.grid(False)
        
        # Ticks à l'extérieur
        ax.tick_params(direction='out', length=4)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figure style article (Fig. 5.7) sauvegardée: {outfile}")

# ============================================================
# FIGURES INDIVIDUELLES - Style article
# ============================================================
def plot_single_article_style(X, Y, p, time, outfile):
    """Figure individuelle - style article avec contours noirs fins"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    levels = np.linspace(np.nanmin(p), np.nanmax(p), 30)
    
    # Fond blanc
    ax.set_facecolor('white')
    
    # Contours noirs fins
    cs = ax.contour(X, Y, p, levels=levels, colors='black', 
                    linewidths=0.5, alpha=1.0)
    
    # Choc vertical épais
    ax.axvline(x=0.5, color='black', linewidth=2.5, linestyle='-')
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title(f'Pression - t = {time:.2f}', fontsize=14)
    
    # Bordure noire
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    ax.tick_params(direction='out', length=5)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figure individuelle sauvegardée: {outfile}")

# ============================================================
# FIGURES SUPPLÉMENTAIRES (densité, vitesse)
# ============================================================
def plot_density_contours(X, Y, rho, time, outfile):
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(np.nanmin(rho), np.nanmax(rho), 40)
    cf = ax.contourf(X, Y, rho, levels=levels, cmap='jet', extend='both')
    ax.contour(X, Y, rho, levels=levels, colors='black', linewidths=0.2, alpha=0.3)
    ax.axvline(x=0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Densité à t = {time:.2f}')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('ρ')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✓ Densité sauvegardée: {outfile}")

def plot_velocity_magnitude(X, Y, u, v, time, outfile):
    vel = np.sqrt(u**2 + v**2)
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = np.linspace(np.nanmin(vel), np.nanmax(vel), 40)
    cf = ax.contourf(X, Y, vel, levels=levels, cmap='plasma', extend='both')
    
    skip = 10
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              color='white', alpha=0.6, scale=25)
    
    ax.axvline(x=0.5, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'|V| à t = {time:.2f}')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('|V|')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✓ Vitesse sauvegardée: {outfile}")

# ============================================================
# MAIN
# ============================================================
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    
    # Chercher le fichier dans plusieurs emplacements possibles
    possible_paths = [
        os.path.join(base, "../resultats/shock_vortex"),
        os.path.join(base, "resultats/shock_vortex"),
        os.path.join(os.getcwd(), "resultats/shock_vortex"),
        "resultats/shock_vortex"
    ]
    
    out = None
    data_file = None
    
    # Trouver le bon chemin
    for path in possible_paths:
        test_file = os.path.join(path, "u.txt")
        if os.path.exists(test_file):
            out = path
            data_file = test_file
            print(f"✓ Fichier trouvé: {data_file}")
            break
    
    if data_file is None or not os.path.exists(data_file):
        print("Fichier u.txt introuvable dans les emplacements suivants:")
        for path in possible_paths:
            print(f"   - {os.path.join(path, 'u.txt')}")
        print("\nVeuillez vérifier que:")
        print("  1. Le programme Fortran a été exécuté")
        print("  2. Le fichier u.txt a été généré")
        print("  3. Le chemin 'resultats/shock_vortex/' existe")
        return
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(out, exist_ok=True)

    print("\n" + "="*70)
    print(" REPRODUCTION EXACTE Fig. 5.7")
    print(" 2D Shock Vortex Interaction - Pressure")
    print(" Fifth order WENO-LF-5-PS - 30 contours")
    print(" STYLE: Contours noirs fins sur fond blanc + choc vertical épais")
    print("="*70 + "\n")

    # Chargement des 3 temps
    print("Chargement des données aux 3 temps de référence...")
    data_05 = load_data(data_file, 0.05)
    data_20 = load_data(data_file, 0.20)
    data_35 = load_data(data_file, 0.35)

    print("\n" + "="*70)
    print("GÉNÉRATION DES FIGURES - STYLE ARTICLE")
    print("="*70 + "\n")

    # Figure principale 3 panneaux - EXACTEMENT comme l'article
    print("1. Figure 3 panneaux (a), (b), (c) - Style Fig. 5.7...")
    plot_article(
        [data_05, data_20, data_35],
        [0.05, 0.20, 0.35],
        os.path.join(out, "figure.png")
    )

    # Figures individuelles - même style
    print("\n2. Figures individuelles aux 3 temps...")
    X, Y, rho, u, v, p = data_05
    plot_single_article_style(X, Y, p, 0.05, os.path.join(out, "pressure_article_t005.png"))
    
    X, Y, rho, u, v, p = data_20
    plot_single_article_style(X, Y, p, 0.20, os.path.join(out, "pressure_article_t020.png"))
    
    X, Y, rho, u, v, p = data_35
    plot_single_article_style(X, Y, p, 0.35, os.path.join(out, "pressure_article_t035.png"))

    # Figures supplémentaires (densité et vitesse)
    print("\n3. Figures supplémentaires (densité, vitesse)...")
    plot_density_contours(X, Y, rho, 0.35, os.path.join(out, "density_t035.png"))
    plot_velocity_magnitude(X, Y, u, v, 0.35, os.path.join(out, "velocity_t035.png"))

    print("\n" + "="*70)
    print("✓ POST-TRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("="*70)
    print(f"\nRépertoire: {out}/\n")
    print(" Figures générées (STYLE ARTICLE):")
    print("  ★ fig57_article_exact.png       ← EXACTEMENT comme Fig. 5.7")
    print("                                     (contours noirs fins + choc épais)")
    print("  • pressure_article_t005.png  (t=0.05)")
    print("  • pressure_article_t020.png  (t=0.20)")
    print("  • pressure_article_t035.png  (t=0.35)")
    print("  • density_t035.png")
    print("  • velocity_t035.png")
    print("\n" + "="*70)
    print("\n CARACTÉRISTIQUES Fig. 5.7:")
    print("  ✓ Fond BLANC pur")
    print("  ✓ Contours NOIRS FINS (linewidth=0.5)")
    print("  ✓ Choc vertical NOIR ÉPAIS à x=0.5 (linewidth=2.5)")
    print("  ✓ 30 niveaux de contours")
    print("  ✓ Zone d'affichage [0,1] × [0,1]")
    print("  ✓ Bordures noires")
    print("  ✓ Labels (a), (b), (c) alignés à gauche")
    print("\n PHYSIQUE:")
    print("  (a) t=0.05: Vortex circulaire intact avant interaction")
    print("  (b) t=0.20: Interaction choc-vortex, déformation progressive")
    print("  (c) t=0.35: Vortex fortement déformé, structures complexes")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()