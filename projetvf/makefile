FC = gfortran
FFLAGS = -O2 -Wall -Wextra -std=f2018 -J$(BUILD_DIR)
LDFLAGS = 

# Répertoires
SRC_DIR = src
EXAMPLE_DIR = example
BUILD_DIR = build
BIN_DIR = bin
RESULTS_DIR_1 = resultats/example1
RESULTS_DIR_2 = resultats/example2
RESULTS_DIR_LAX = resultats/lax
RESULTS_DIR_RIEMANN = resultats/riemann
RESULTS_DIR_SHOCKVORTEX = resultats/shock_vortex
RESULTS_DIR_DOUBLEMACH = resultats/double_mach

# Noms des exécutables
TARGET_1 = $(BIN_DIR)/example1_euler_1D
TARGET_2 = $(BIN_DIR)/example2_euler_2D
TARGET_LAX = $(BIN_DIR)/lax
TARGET_RIEMANN = $(BIN_DIR)/riemann_2d_multi_case
TARGET_SHOCKVORTEX = $(BIN_DIR)/shock_vortex2D
TARGET_DOUBLEMACH = $(BIN_DIR)/double_mach

# Fichiers sources des modules 
MODULE_SOURCES = $(SRC_DIR)/precisions.F90 \
                 $(SRC_DIR)/grille.f90 \
                 $(SRC_DIR)/schema_weno.f90 \
                 $(SRC_DIR)/schema_temps.f90

# Fichiers sources des exemples
EXAMPLE1_SOURCE = $(EXAMPLE_DIR)/example1_euler_1D.f90
EXAMPLE2_SOURCE = $(EXAMPLE_DIR)/example2_euler_2D.f90
LAX_SOURCE      = $(EXAMPLE_DIR)/lax.f90
RIEMANN_MULTI_SOURCE = $(EXAMPLE_DIR)/riemann_2d_multi_case.f90
SHOCKVORTEX_SOURCE = $(EXAMPLE_DIR)/shock_vortex2D.f90
DOUBLEMACH_SOURCE = $(EXAMPLE_DIR)/double_mach.f90

# Objets des modules
MODULE_OBJECTS = $(patsubst $(SRC_DIR)/%.F90,$(BUILD_DIR)/%.o,$(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(MODULE_SOURCES)))

# Objets des exemples
EXAMPLE1_OBJECT = $(BUILD_DIR)/example1_euler_1D.o
EXAMPLE2_OBJECT = $(BUILD_DIR)/example2_euler_2D.o
LAX_OBJECT      = $(BUILD_DIR)/lax.o
RIEMANN_OBJECT = $(BUILD_DIR)/riemann_2d_multi_case.o
SHOCKVORTEX_OBJECT = $(BUILD_DIR)/shock_vortex2D.o
DOUBLEMACH_OBJECT = $(BUILD_DIR)/double_mach.o

# ===============================================
all: directories $(TARGET_1) $(TARGET_2) $(TARGET_LAX) $(TARGET_RIEMANN) $(TARGET_SHOCKVORTEX) $(TARGET_DOUBLEMACH)

# Compiler seulement l'exemple 1D
example1: directories $(TARGET_1)

# Compiler seulement l'exemple 2D
example2: directories $(TARGET_2)

# Compiler le problème de Lax
lax: directories $(TARGET_LAX)

# Compiler Riemann multi-cas
riemann: directories $(TARGET_RIEMANN)

# Compiler Shock-Vortex
shockvortex: directories $(TARGET_SHOCKVORTEX)

# Compiler Double Mach Reflection
doublemach: directories $(TARGET_DOUBLEMACH)

# Créer les répertoires nécessaires
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) $(RESULTS_DIR_1) $(RESULTS_DIR_2) \
	          $(RESULTS_DIR_LAX) $(RESULTS_DIR_RIEMANN) $(RESULTS_DIR_SHOCKVORTEX) \
	          $(RESULTS_DIR_DOUBLEMACH)

# ===============================================
# Lier les exécutables
# ===============================================
$(TARGET_1): $(MODULE_OBJECTS) $(EXAMPLE1_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_1)"

$(TARGET_2): $(MODULE_OBJECTS) $(EXAMPLE2_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_2)"

$(TARGET_LAX): $(MODULE_OBJECTS) $(LAX_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_LAX)"

$(TARGET_RIEMANN): $(MODULE_OBJECTS) $(RIEMANN_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_RIEMANN)"

$(TARGET_SHOCKVORTEX): $(MODULE_OBJECTS) $(SHOCKVORTEX_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_SHOCKVORTEX)"

$(TARGET_DOUBLEMACH): $(MODULE_OBJECTS) $(DOUBLEMACH_OBJECT)
	$(FC) $(LDFLAGS) -o $@ $^
	@echo "✓ Compilation terminée : $(TARGET_DOUBLEMACH)"

# ===============================================
# Compiler les modules
# ===============================================
$(BUILD_DIR)/precisions.o: $(SRC_DIR)/precisions.F90
	$(FC) $(FFLAGS) -cpp -c $< -o $@

$(BUILD_DIR)/grille.o: $(SRC_DIR)/grille.f90 $(BUILD_DIR)/precisions.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/schema_weno.o: $(SRC_DIR)/schema_weno.f90 $(BUILD_DIR)/precisions.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/schema_temps.o: $(SRC_DIR)/schema_temps.f90 $(BUILD_DIR)/precisions.o
	$(FC) $(FFLAGS) -c $< -o $@

# ===============================================
# Compiler les exemples
# ===============================================
$(EXAMPLE1_OBJECT): $(EXAMPLE1_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

$(EXAMPLE2_OBJECT): $(EXAMPLE2_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

$(LAX_OBJECT): $(LAX_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

$(RIEMANN_OBJECT): $(RIEMANN_MULTI_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

$(SHOCKVORTEX_OBJECT): $(SHOCKVORTEX_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

$(DOUBLEMACH_OBJECT): $(DOUBLEMACH_SOURCE) $(MODULE_OBJECTS)
	$(FC) $(FFLAGS) -c $< -o $@

# ===============================================
# Nettoyage
# ===============================================
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Nettoyage terminé"

cleanall: clean
	rm -rf $(RESULTS_DIR_1) $(RESULTS_DIR_2) $(RESULTS_DIR_LAX) \
	       $(RESULTS_DIR_RIEMANN) $(RESULTS_DIR_SHOCKVORTEX) $(RESULTS_DIR_DOUBLEMACH)
	@echo "✓ Nettoyage complet terminé"

rebuild: clean all

# ===============================================
# Exécution
# ===============================================
run1: example1
	@mkdir -p $(RESULTS_DIR_1)
	@echo "=== Exécution Example 1 (Euler 1D - Sod) ==="
	./$(TARGET_1)

run2: example2
	@mkdir -p $(RESULTS_DIR_2)
	@echo "=== Exécution Example 2 (Euler 2D - Vortex isentropique) ==="
	./$(TARGET_2)

run3: lax
	@mkdir -p $(RESULTS_DIR_LAX)
	@echo "=== Exécution Lax Problem ==="
	./$(TARGET_LAX)

runriemann: riemann
	@mkdir -p $(RESULTS_DIR_RIEMANN)
	@echo "=== Exécution Riemann 2D Multi-Cas ==="
	./$(TARGET_RIEMANN)

runshockvortex: shockvortex
	@mkdir -p $(RESULTS_DIR_SHOCKVORTEX)
	@echo "=== Exécution Shock-Vortex Interaction 2D ==="
	./$(TARGET_SHOCKVORTEX)

rundoublemach: doublemach
	@mkdir -p $(RESULTS_DIR_DOUBLEMACH)
	@echo "=== Exécution Double Mach Reflection 2D ==="
	./$(TARGET_DOUBLEMACH)

# Exécuter tous les exemples
runall: run1 run2 run3 runriemann runshockvortex rundoublemach

# ===============================================
# Post-traitement / visualisation
# ===============================================
runplot1: run1
	@echo "=== Visualisation Example 1 (Sod) ==="
	python3 $(EXAMPLE_DIR)/plotresults_example1.py

runplot2: run2
	@echo "=== Visualisation Example 2 (Vortex) ==="
	python3 $(EXAMPLE_DIR)/plotresults_example1_2D.py

runplot3: run3
	@echo "=== Visualisation Lax Problem ==="
	python3 $(EXAMPLE_DIR)/python_lax.py

runplotriemann: runriemann
	@echo "=== Visualisation Riemann 2D ==="
	python3 $(EXAMPLE_DIR)/plotresults_riemann2d.py

runplotshockvortex: runshockvortex
	@echo "=== Visualisation Shock-Vortex Interaction ==="
	python3 $(EXAMPLE_DIR)/plotsresultas_shockvortex.py

runplotdoublemach: rundoublemach
	@echo "=== Visualisation Double Mach Reflection ==="
	python3 $(EXAMPLE_DIR)/plotresults_doublemach.py

# Exécuter et visualiser tous les exemples
runplotall: runplot1 runplot2 runplot3 runplotriemann runplotshockvortex runplotdoublemach

# ===============================================
# PHONY
# ===============================================
.PHONY: all example1 example2 lax riemann shockvortex doublemach clean cleanall rebuild \
        run1 run2 run3 runriemann runshockvortex rundoublemach runall \
        runplot1 runplot2 runplot3 runplotriemann runplotshockvortex runplotdoublemach runplotall \
        directories print help