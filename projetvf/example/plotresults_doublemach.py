import numpy as np
import matplotlib.pyplot as plt

# ============================
# PARAMÈTRES
# ============================
folder = "./resultats/double_mach/"
time_target = 0.2
nx, ny = 480, 119

# Shu (figure)
rho_min = 1.731
rho_max = 20.92
nlevels = 30

# Nom du fichier
outfile = folder + "double_mach_density_t0p2.png"

# ============================
# LECTURE DE LA GRILLE
# ============================
xdata = np.loadtxt(folder + "x.txt", skiprows=1)
x = xdata[:,2].reshape(ny, nx)
y = xdata[:,3].reshape(ny, nx)

# ============================
# LECTURE DES DONNÉES
# ============================
raw = np.loadtxt(folder + "u.txt", skiprows=1)
times = raw[:,0]
itime = np.argmin(np.abs(times - time_target))
tval = times[itime]

mask = np.isclose(raw[:,0], tval)
snap = raw[mask]
rho = snap[:,3].reshape(ny, nx)

print(f"Plot at time t = {tval}")
print(f"Density range: [{rho.min():.3f}, {rho.max():.3f}]")

# ============================
# TRACÉ (style article)
# ============================
fig, ax = plt.subplots(figsize=(12, 3.5))

levels = np.linspace(rho_min, rho_max, nlevels)

# Premier contour plus épais (comme dans l'article)
ax.contour(x, y, rho, levels=[levels[0]], colors='black', linewidths=1.8)
# Contours suivants plus fins
ax.contour(x, y, rho, levels=levels[1:], colors='black', linewidths=0.6)

# Limites et aspect
ax.set_xlim(0.0, 3.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect('equal')

# Titre à gauche
ax.text(0.01, 1.02, "DENSITY", 
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=14, weight='bold')

# Nom du schéma à droite
ax.text(0.99, 1.02, "WENO-LF-5",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=14, weight='bold')

# Labels des axes
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)

# Annotation façon Shu (en bas)
ax.text(0.5, -0.16,
        f"{nlevels} contours from {rho_min:.3f} to {rho_max:.2f}    "
        f"Grid: {nx}×{ny}    cfl=0.6    t={tval:.2f}",
        transform=ax.transAxes, 
        ha='center', va='top', fontsize=11)

plt.tight_layout()

# ============================
# SAUVEGARDE IMAGE
# ============================
plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor='white')
print(f"✓ Image saved to: {outfile}")
plt.show()
