import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Données numériques (L2)
# ==========================
Nx = np.array([160, 80, 40])
dx = 10.0 / Nx

L2_rho = np.array([1.01e-6, 9.17e-6, 1.15e-4])
L2_u   = np.array([2.18e-6, 1.36e-5, 2.44e-4])
L2_p   = np.array([1.23e-6, 5.66e-6, 6.29e-5])

# ==========================
# Ordres 
# ==========================
def observed_order(err, dx):
    p1 = np.log(err[1]/err[0]) / np.log(dx[1]/dx[0])  # 160 → 80
    p2 = np.log(err[2]/err[1]) / np.log(dx[2]/dx[1])  # 80 → 40
    return p1, p2, 0.5*(p1+p2)

p_rho = observed_order(L2_rho, dx)
p_u   = observed_order(L2_u, dx)
p_p   = observed_order(L2_p, dx)

print("Ordres observés (L2) :")
print(f"ρ  : {p_rho}")
print(f"u  : {p_u}")
print(f"p  : {p_p}")

# ==========================
# Référence ordre 5
# ==========================
dx_ref = np.linspace(dx.min(), dx.max(), 100)
C5 = L2_rho[0] / dx[0]**5
err_dx5 = C5 * dx_ref**5

# ==========================
# Tracé
# ==========================
plt.figure(figsize=(7,6))

plt.loglog(dx, L2_rho, 'o-', lw=2,
           label=rf'$L^2(\rho)$  (p≈{p_rho[2]:.2f})')
plt.loglog(dx, L2_u,   's-', lw=2,
           label=rf'$L^2(u)$    (p≈{p_u[2]:.2f})')
plt.loglog(dx, L2_p,   '^-', lw=2,
           label=rf'$L^2(p)$    (p≈{p_p[2]:.2f})')

plt.loglog(dx_ref, err_dx5, 'k--', lw=2.2,
           label=r'$\mathcal{O}(\Delta x^5)$ ')

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'Erreur $L^2$')
plt.title('Vortex isentropique 2D — Convergence numérique')

plt.legend()
plt.tight_layout()
plt.savefig("convergence_vortex_L2_order5.png", dpi=300)
plt.show()
