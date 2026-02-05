#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction de la figure pour le problème de Lax
avec schéma WENO-5 (style article scientifique)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Chargement des données
# ---------------------------------------------------------
def load_data(filepath):
    """Charge t, x, rho, u, p depuis un fichier u.txt."""
    data = np.loadtxt(filepath, skiprows=1)
    t_all = data[:, 0]
    x_all = data[:, 1]
    rho_all = data[:, 2]
    u_all  = data[:, 3]
    p_all  = data[:, 4]
    t = np.unique(t_all)
    nt = len(t)
    nx = len(x_all) // nt
    x = x_all[:nx]
    rho = rho_all.reshape(nt, nx)
    u   = u_all.reshape(nt, nx)
    p   = p_all.reshape(nt, nx)
    return t, x, rho, u, p


# ---------------------------------------------------------
# NOUVEAU : Solveur exact du problème de Riemann (Toro)
# ---------------------------------------------------------
def exact_riemann_solver(rho_L, u_L, p_L, rho_R, u_R, p_R, x, t, gamma=1.4):
    """
    Solveur exact du problème de Riemann pour les équations d'Euler 1D
    Implémentation basée sur l'algorithme de Toro (Riemann Solvers).
    Renvoie (rho, u, p) évalués aux positions x au temps t.
    """
    # Cas t == 0 : renvoyer conditions initiales
    if t == 0.0:
        rho = np.where(x < 0.0, rho_L, rho_R)
        u   = np.where(x < 0.0, u_L, u_R)
        p   = np.where(x < 0.0, p_L, p_R)
        return rho, u, p

    # vitesses du son
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # estimation initiale de p_star (PVRS)
    p_pvrs = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p_star = max(1e-8, p_pvrs)

    # fonctions auxiliaires pour f(p) et df(p)
    def f_and_df(p, rho, p_k, a_k):
        if p > p_k:  # choc
            A = 2.0 / ((gamma + 1.0) * rho)
            B = (gamma - 1.0) / (gamma + 1.0) * p_k
            sqrt_term = np.sqrt(A / (p + B))
            f = (p - p_k) * sqrt_term
            df = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + B))
        else:  # détente
            f = (2.0 * a_k / (gamma - 1.0)) * ((p / p_k)**((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            df = (1.0 / (rho * a_k)) * (p / p_k)**(-(gamma + 1.0) / (2.0 * gamma)) * (1.0 / (2.0 * gamma)) * (2.0 * a_k * (gamma - 1.0))
            # simplification : utiliser expression analytique stable
            df = (1.0 / (rho * a_k)) * (p / p_k)**(-(gamma + 1.0) / (2.0 * gamma))
        return f, df

    # Newton-Raphson pour p_star
    for _ in range(50):
        fL, dfL = f_and_df(p_star, rho_L, p_L, a_L)
        fR, dfR = f_and_df(p_star, rho_R, p_R, a_R)
        func = fL + fR + (u_R - u_L)
        deriv = dfL + dfR
        # protection
        if deriv == 0:
            break
        p_new = p_star - func / deriv
        if p_new < 0:
            p_new = 1e-8
        if abs(p_new - p_star) / (p_new + 1e-12) < 1e-8:
            p_star = p_new
            break
        p_star = p_new

    # vitesse dans la région étoile
    fL, _ = f_and_df(p_star, rho_L, p_L, a_L)
    fR, _ = f_and_df(p_star, rho_R, p_R, a_R)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (fR - fL)

    # densités dans la région étoile
    if p_star > p_L:
        rho_star_L = rho_L * ( (p_star / p_L + (gamma - 1.0) / (gamma + 1.0)) /
                               ((gamma - 1.0) / (gamma + 1.0) * p_star / p_L + 1.0) )
    else:
        rho_star_L = rho_L * (p_star / p_L)**(1.0 / gamma)

    if p_star > p_R:
        rho_star_R = rho_R * ( (p_star / p_R + (gamma - 1.0) / (gamma + 1.0)) /
                               ((gamma - 1.0) / (gamma + 1.0) * p_star / p_R + 1.0) )
    else:
        rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma)

    # Préparer tableaux de sortie
    rho = np.zeros_like(x)
    u   = np.zeros_like(x)
    p   = np.zeros_like(x)

    # échantillonnage de la solution au point (x/t) — origine de la discontinuité en 0
    S = x / t

    # calcul des ondes pour la gauche
    if p_star > p_L:  # choc gauche
        S_L = u_L - a_L * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_L + (gamma - 1.0) / (2.0 * gamma))
    else:  # détente gauche
        S_HL = u_L - a_L
        a_star_L = a_L * (p_star / p_L)**((gamma - 1.0) / (2.0 * gamma))
        S_TL = u_star - a_star_L

    # calcul des ondes pour la droite
    if p_star > p_R:  # choc droite
        S_R = u_R + a_R * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_R + (gamma - 1.0) / (2.0 * gamma))
    else:  # détente droite
        S_HR = u_R + a_R
        a_star_R = a_R * (p_star / p_R)**((gamma - 1.0) / (2.0 * gamma))
        S_TR = u_star + a_star_R

    # boucle sur points spatiaux
    for i, s in enumerate(S):
        if s < u_star:
            # zone à gauche de la discontinuité de contact
            if p_star > p_L:
                # choc à gauche
                if s < S_L:
                    rho[i] = rho_L; u[i] = u_L; p[i] = p_L
                else:
                    rho[i] = rho_star_L; u[i] = u_star; p[i] = p_star
            else:
                # détente à gauche (fan)
                if s < S_HL:
                    rho[i] = rho_L; u[i] = u_L; p[i] = p_L
                elif s > S_TL:
                    rho[i] = rho_star_L; u[i] = u_star; p[i] = p_star
                else:
                    # dans la rarefaction self-similaire
                    # relations analytiques pour la rarefaction
                    u_i = (2.0/(gamma+1.0))*(a_L + 0.5*(gamma-1.0)*u_L + s)
                    a_i = (2.0/(gamma+1.0))*(a_L + 0.5*(gamma-1.0)*(u_L - s))
                    rho[i] = rho_L * (a_i / a_L)**(2.0/(gamma-1.0))
                    u[i] = u_i
                    p[i] = p_L * (a_i / a_L)**(2.0*gamma/(gamma-1.0))
        else:
            # zone à droite de la discontinuité de contact
            if p_star > p_R:
                # choc à droite
                if s > S_R:
                    rho[i] = rho_R; u[i] = u_R; p[i] = p_R
                else:
                    rho[i] = rho_star_R; u[i] = u_star; p[i] = p_star
            else:
                # détente à droite (fan)
                if s > S_HR:
                    rho[i] = rho_R; u[i] = u_R; p[i] = p_R
                elif s < S_TR:
                    rho[i] = rho_star_R; u[i] = u_star; p[i] = p_star
                else:
                    # dans la rarefaction droite (formule adaptée)
                    u_i = (2.0/(gamma+1.0))*(-a_R + 0.5*(gamma-1.0)*u_R + s)
                    a_i = (2.0/(gamma+1.0))*(a_R - 0.5*(gamma-1.0)*(u_R - s))
                    rho[i] = rho_R * (a_i / a_R)**(2.0/(gamma-1.0))
                    u[i] = u_i
                    p[i] = p_R * (a_i / a_R)**(2.0*gamma/(gamma-1.0))

    return rho, u, p
def exact_solution_lax(x, t=1.3, gamma=1.4):
    """
    Solution exacte du problème de Lax
    (ρ_L, u_L, P_L) = (0.445, 0.698, 3.528)
    (ρ_R, u_R, P_R) = (0.5, 0.0, 0.571)
    """
    rho_L, u_L, p_L = 0.445, 0.698, 3.528
    rho_R, u_R, p_R = 0.5, 0.0, 0.571
    
    return exact_riemann_solver(rho_L, u_L, p_L, rho_R, u_R, p_R, x, t, gamma)

# ---------------------------------------------------------
# 3) Figure style article 
# ---------------------------------------------------------
def plot_lax_comparison(t, x, rho, u, p, output_dir):
    """
    Figure reproduisant le style de l'article pour le problème de Lax
    avec une seule sous-figure montrant WENO-RF-5
    """
    tf = -1  # temps final
    time_final = t[tf]
    
    # Solution exacte
    rho_exact, u_exact, p_exact = exact_solution_lax(x, time_final)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Titre global avec paramètres
    fig.suptitle(f'DENSITY    t={time_final:.1f}  cfl=0.6  n={len(x)}', 
                 fontsize=11, y=0.98)
    
    # Tracer la solution numérique (cercles vides)
    ax.plot(x, rho[tf], 'o', markersize=4, markerfacecolor='none', 
            markeredgecolor='black', markeredgewidth=0.8, label='WENO-RF-5')
    
    # Tracer la solution exacte (ligne continue)
    ax.plot(x, rho_exact, '-', color='black', linewidth=1.5, label='EXACT')
    
    # Configuration des axes
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.40, 1.30)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('ρ', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    
  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lax_weno5_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  → lax_weno5_comparison.png créée")

# ---------------------------------------------------------
# 4) Figure : état final détaillé (3 variables)
# ---------------------------------------------------------
def plot_final_state(t, x, rho, u, p, output_dir):
    """Affiche les 3 variables à l'état final"""
    tf = -1  # dernier instant
    
    # Solutions exactes
    rho_exact, u_exact, p_exact = exact_solution_lax(x, t[tf])
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"État final - Problème de Lax (t={t[tf]:.2f}s)", fontsize=14)
    
    # DENSITÉ
    axs[0].plot(x, rho[tf], 'o-', markersize=3, label='WENO-5')
    axs[0].plot(x, rho_exact, 'k--', linewidth=1.5, label='Exact')
    axs[0].set_ylabel("Densité ρ", fontsize=11)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # VITESSE
    axs[1].plot(x, u[tf], 'o-', markersize=3, label='WENO-5')
    axs[1].plot(x, u_exact, 'k--', linewidth=1.5, label='Exact')
    axs[1].set_ylabel("Vitesse u", fontsize=11)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # PRESSION
    axs[2].plot(x, p[tf], 'o-', markersize=3, label='WENO-5')
    axs[2].plot(x, p_exact, 'k--', linewidth=1.5, label='Exact')
    axs[2].set_ylabel("Pression p", fontsize=11)
    axs[2].set_xlabel("x", fontsize=11)
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lax_final_state.png"), dpi=200)
    plt.close()
    print("  → lax_final_state.png créée")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "../resultats/lax")
    data_file = os.path.join(out, "u.txt")
    
    print("=" * 60)
    print("Visualisation du problème de Lax avec WENO-5")
    print("=" * 60)
    
    if not os.path.exists(data_file):
        print(f"ERREUR: Fichier {data_file} introuvable!")
        print("Assurez-vous d'avoir exécuté le programme Fortran.")
        return
    
    print("\n[1/4] Chargement des données...")
    t, x, rho, u, p = load_data(data_file)
    print(f"      {len(t)} pas de temps, {len(x)} points d'espace")
    print(f"      Temps: t=[{t[0]:.3f}, {t[-1]:.3f}]")
    
    print("\n[2/4] Figure style article (WENO-5 vs Exact)...")
    plot_lax_comparison(t, x, rho, u, p, out)
    
    print("\n[3/4] État final (ρ, u, p)...")
    plot_final_state(t, x, rho, u, p, out)
    
    print("\n[4/4] Évolution temporelle...")
    
    print("\n" + "=" * 60)
    print("✓ Terminé ! Figures créées dans:", out)
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - lax_weno3_comparison.png  (style article)")
    print("  - lax_final_state.png       (3 variables)")
    print()

if __name__ == "__main__":
    main()
