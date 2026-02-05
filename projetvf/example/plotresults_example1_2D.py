import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# PARAMÈTRES PHYSIQUES
# ===============================
gamma = 1.4
eps = 5.0   # vortex doux
Lx = 10.0
Ly = 10.0

# ===============================
# 1) Lecture des centres (x.txt) + u.txt
# ===============================
def load_u_txt(folder, time_select=None):
    # ---- lecture centres exacts (Fortran) ----
    xdata = np.loadtxt(os.path.join(folder, "x.txt"), skiprows=1)
    i = xdata[:,0].astype(int)
    j = xdata[:,1].astype(int)
    x = xdata[:,2]
    y = xdata[:,3]

    nx = i.max()
    ny = j.max()

    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)

    # ---- lecture solution numérique ----
    data = np.genfromtxt(os.path.join(folder, "u.txt"), skip_header=1)
    data = data[~np.isnan(data).any(axis=1)]

    if time_select is not None:
        unique_times = np.unique(data[:,0])
        idx = np.argmin(np.abs(unique_times - time_select))
        selected_time = unique_times[idx]
        data = data[data[:,0] == selected_time]
        print(f"Temps sélectionné: {selected_time:.6f} (demandé: {time_select})")

    if data.size == 0:
        raise ValueError("Aucun snapshot trouvé")

    rho = data[:,3].reshape(ny, nx)
    u   = data[:,4].reshape(ny, nx)
    v   = data[:,5].reshape(ny, nx)
    p   = data[:,6].reshape(ny, nx)

    return X, Y, rho, u, v, p

# ===============================
# 2) Solution exacte du vortex (CENTRES)
# ===============================
def exact_vortex(X, Y, t):
    Xs = np.mod(X - t, Lx)
    Ys = np.mod(Y - t, Ly)

    xb = Xs - 5.0
    yb = Ys - 5.0
    r2 = xb**2 + yb**2

    factor = np.exp(0.5 * (1.0 - r2))
    du = -eps/(2*np.pi) * yb * factor
    dv =  eps/(2*np.pi) * xb * factor
    dT = -(gamma-1)*eps**2/(8*gamma*np.pi**2) * np.exp(1.0 - r2)

    T = 1.0 + dT
    rho = T**(1.0/(gamma-1))
    p = rho**gamma
    u = 1.0 + du
    v = 1.0 + dv

    return rho, u, v, p

# ===============================
# 3) FIGURES 
# ===============================
def plot_2d_vortex(X, Y, rho_num, rho_exact, outfile):
    vmin = min(rho_exact.min(), rho_num.min())
    vmax = max(rho_exact.max(), rho_num.max())
    levels = np.linspace(vmin, vmax, 30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cf1 = ax1.contourf(X, Y, rho_num, levels=levels, cmap="jet")
    ax1.contour(X, Y, rho_num, colors="black", linewidths=0.3, levels=levels)
    ax1.set_aspect('equal')
    ax1.set_title("Solution numérique (ρ)")
    plt.colorbar(cf1, ax=ax1)

    cf2 = ax2.contourf(X, Y, rho_exact, levels=levels, cmap="jet")
    ax2.contour(X, Y, rho_exact, colors="black", linewidths=0.3, levels=levels)
    ax2.set_aspect('equal')
    ax2.set_title("Solution exacte (ρ)")
    plt.colorbar(cf2, ax=ax2)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_cut(X, Y, rho_num, rho_exact, outfile):
    idx = np.argmin(np.abs(X[0,:] - 5.0))
    ycut = Y[:,idx]
    numcut = rho_num[:,idx]
    exactcut = rho_exact[:,idx]

    plt.figure(figsize=(8,6))
    plt.plot(ycut, exactcut, '-', label="Exact", lw=2)
    plt.plot(ycut, numcut, 'o', label="Numérique", ms=4)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_error_map(X, Y, rho_num, rho_exact, outfile):
    error = np.abs(rho_num - rho_exact)
    plt.figure(figsize=(8,6))
    cf = plt.contourf(X, Y, error, levels=30, cmap="hot")
    plt.gca().set_aspect('equal')
    plt.colorbar(cf)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

# ===============================
# 4) ERREURS
# ===============================
def compute_errors(num, exact):
    diff = num - exact
    return (
        np.mean(np.abs(diff)),
        np.sqrt(np.mean(diff**2)),
        np.max(np.abs(diff))
    )

# ===============================
# 5) MAIN
# ===============================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "../resultats/example2")
    t_final = 50.0

    print("="*60)
    print("POST-TRAITEMENT VORTEX ISENTROPIQUE 2D (CENTRES)")
    print("="*60)

    X, Y, rho_n, u_n, v_n, p_n = load_u_txt(out, time_select=t_final)
    rho_e, u_e, v_e, p_e = exact_vortex(X, Y, t_final)

    L1, L2, Linf = compute_errors(rho_n, rho_e)
        # --- erreurs ---
    L1_rho, L2_rho, Linf_rho = compute_errors(rho_n, rho_e)
    L1_u,   L2_u,   Linf_u   = compute_errors(u_n,   u_e)
    L1_v,   L2_v,   Linf_v   = compute_errors(v_n,   v_e)
    L1_p,   L2_p,   Linf_p   = compute_errors(p_n,   p_e)

    print("\nErreurs sur ρ:")
    print(f"  L1   = {L1_rho:.6e}")
    print(f"  L2   = {L2_rho:.6e}")
    print(f"  L∞   = {Linf_rho:.6e}")

    print("\nErreurs sur u:")
    print(f"  L1   = {L1_u:.6e}")
    print(f"  L2   = {L2_u:.6e}")
    print(f"  L∞   = {Linf_u:.6e}")

    print("\nErreurs sur v:")
    print(f"  L1   = {L1_v:.6e}")
    print(f"  L2   = {L2_v:.6e}")
    print(f"  L∞   = {Linf_v:.6e}")

    print("\nErreurs sur p:")
    print(f"  L1   = {L1_p:.6e}")
    print(f"  L2   = {L2_p:.6e}")
    print(f"  L∞   = {Linf_p:.6e}")


    plot_2d_vortex(X, Y, rho_n, rho_e, os.path.join(out,"vortex_2D_comparison.png"))
    plot_cut(X, Y, rho_n, rho_e, os.path.join(out,"vortex_cut_x5.png"))
    plot_error_map(X, Y, rho_n, rho_e, os.path.join(out,"vortex_error_map.png"))

    print("✓ Post-traitement terminé (géométrie cohérente)")

if __name__ == "__main__":
    main()

