# Localisation de source sur graphe métrique par méthode adjointe

## 📌 Description

Ce projet traite un **problème inverse de localisation de sources** sur des **graphes métriques en dimension 1 et 2**.

La méthode repose sur :
- une **discrétisation par différences finies** sur les arêtes,
- la **résolution du problème direct elliptique**,
- le **calcul des sensibilités**,
- la **méthode adjointe** pour le calcul efficace du gradient,
- une **validation par différences finies**.

Le cadre est celui d’un **problème inverse gouverné par une équation elliptique sur graphe métrique**.

---

## 🧠 Modèle mathématique

On considère le **problème direct** :

**A · u = g(ε)**

où :
- **u** est l’état (solution du problème direct),
- **g(ε)** est une source localisée (gaussienne) dépendant du paramètre **ε**,
- **A** est l’opérateur de diffusion discret sur le graphe métrique.

La **fonctionnelle de coût** est définie par :

**J(ε) = 1/2 ∫ (u − u_data)² dx  
   + (ϖ / 2) Σ_bord (flux − flux_data)²**

où **u_data** représente les données de référence et **ϖ ≥ 0** un paramètre de pondération.

---

## 🎯 Objectifs du code

Le code permet de :
- construire des **graphes métriques** (topologie et géométrie),
- résoudre le **problème direct**,
- calculer les **sensibilités** ∂u/∂ε,
- implémenter la **méthode adjointe**,
- calculer le **gradient de la fonctionnelle de coût** :

**dJ/dε = − pᵀ · ∂g/∂ε**

- comparer les résultats aux **différences finies** (validation).

---

## 🧩 Structure du code

### 1️⃣ `MetricGraph`

Classe représentant un **graphe métrique** :
- sommets internes et sommets de bord,
- arêtes avec longueur, coefficient de diffusion et discrétisation,
- construction des degrés de liberté (DDL),
- visualisation géométrique en 2D.

### 2️⃣ `SourceLocalization`

Classe principale dédiée au **problème inverse** :
- assemblage du système linéaire,
- résolution du problème direct,
- calcul des sensibilités,
- résolution de l’équation adjointe,
- calcul du gradient,
- évaluation de la fonctionnelle de coût,
- visualisation des solutions directes et adjointes.

---

## 🔬 Méthode adjointe (principe)

Plutôt que de calculer une sensibilité par paramètre, la méthode repose sur :

1. **Problème direct**  
   A · u = g(ε)

2. **Problème adjoint**  
   Aᵀ · p = − ∂J/∂u

3. **Gradient**  
   dJ/dε = − pᵀ · ∂g/∂ε

👉 Le coût de calcul est **indépendant du nombre de paramètres**.

---

## ▶️ Exemples fournis

Le script principal inclut :
- ✅ **Validation 1D** (sensibilité vs différences finies),
- 📊 **Étude de sensibilité sur graphe métrique 2D**,
- 🔁 **Validation complète de la méthode adjointe**,
- 🎨 **Visualisations** :
  - graphe métrique,
  - solution directe,
  - état adjoint,
  - champs de sensibilité.

---

## 🖥️ Dépendances

- Python ≥ 3.8
- `numpy`
- `scipy`
- `matplotlib`

Installation :
```bash
pip install numpy scipy matplotlib
