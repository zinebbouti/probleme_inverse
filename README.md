# Localisation de source sur graphe métrique par méthode adjointe

## 📌 Description
Ce projet implémente la **localisation de sources** sur des **graphes métriques 1D/2D** en utilisant :
- une discrétisation par **différences finies** sur les arêtes,
- la **résolution du problème direct**,
- le **calcul de sensibilités**,
- la **méthode adjointe** pour le calcul efficace du gradient,
- une **validation systématique par différences finies**.

Le cadre est celui d’un **problème inverse** gouverné par une équation elliptique sur graphe métrique.

---

## 🧠 Modèle mathématique

On considère le problème direct :
\[
A u = g(\varepsilon)
\]

où :
- \( u \) est l’état (solution),
- \( g(\varepsilon) \) est une source localisée (gaussienne) dépendant du paramètre \( \varepsilon \),
- \( A \) est l’opérateur de diffusion discret sur le graphe.

La fonctionnelle de coût est :
\[
J(\varepsilon) =
\frac{1}{2} \int (u - u_{\text{data}})^2 \, dx
+ \frac{\varpi}{2} \sum_{\text{bord}} ( \text{flux} - \text{flux}_{\text{data}} )^2
\]

---

## 🎯 Objectifs du code

- Construire des **graphes métriques** (topologie + géométrie)
- Résoudre le **problème direct**
- Calculer les **sensibilités** \( \partial u / \partial \varepsilon \)
- Implémenter la **méthode adjointe**
- Calculer le **gradient du coût** :
\[
\frac{dJ}{d\varepsilon} = - p^T \frac{\partial g}{\partial \varepsilon}
\]
- Comparer avec les **différences finies** (validation)

---

## 🧩 Structure du code

### 1️⃣ `MetricGraph`
Classe représentant un **graphe métrique** :
- sommets internes / de bord,
- arêtes avec longueur, diffusion, discrétisation,
- construction des degrés de liberté (DDL),
- visualisation 2D du graphe.

### 2️⃣ `SourceLocalization`
Classe principale pour le problème inverse :
- assemblage du système linéaire,
- résolution du problème direct,
- calcul des sensibilités,
- équation adjointe,
- calcul du gradient,
- fonctionnelle de coût,
- visualisation des solutions et états adjoints.

---

## 🔬 Méthode adjointe (idée clé)

Au lieu de calculer une sensibilité par paramètre (coût élevé), on résout :
1. **Problème direct** :  
   \[
   A u = g(\varepsilon)
   \]
2. **Problème adjoint** :  
   \[
   A^T p = -\frac{\partial J}{\partial u}
   \]
3. **Gradient** :
   \[
   \frac{dJ}{d\varepsilon} = -p^T \frac{\partial g}{\partial \varepsilon}
   \]

➡️ **Coût indépendant du nombre de paramètres**.

---

## ▶️ Exemples fournis

Le script principal contient plusieurs cas de test :

- ✅ **Validation 1D** (sensibilité vs différences finies)
- 📊 **Étude de sensibilité sur graphe 2D**
- 🔁 **Validation complète de la méthode adjointe**
- 🎨 **Visualisation** :
  - graphe métrique,
  - solution directe,
  - état adjoint,
  - sensibilités.

---

## 🖥️ Dépendances

- Python ≥ 3.8
- `numpy`
- `scipy`
- `matplotlib`

Installation :
```bash
pip install numpy scipy matplotlib
