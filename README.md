# 2D gLSM reproduction of Yashin & Balazs 2007

Minimal NumPy/PyTorch reimplementation of the **two-dimensional gel lattice spring
model (gLSM)** from
> V. V. Yashin and A. C. Balazs,
> *Theoretical and computational modeling of self-oscillating polymer gels*,
> J. Chem. Phys. **126**, 124707 (2007), DOI 10.1063/1.2672951.

The paper combines:

1. **Modified 2-variable Oregonator** kinetics in the presence of a polymer
   diluent (eqs 4–7),
2. **Two-fluid model** linking solvent flux to polymer velocity field
   (eqs 8–16),
3. **Neo-Hookean rubber elasticity + Flory–Huggins mixing** free energy
   (eqs 19–25),
4. **Lattice spring discretisation** on a square lattice with bilinear
   quadrilateral elements (eqs 37–53), yielding explicit formulas for the
   nodal force (eq 48) and mobility (eq 53).

The reacting BZ gel is a chemomechanical transducer: the Ru(bpy)₃ catalyst,
covalently bound to the polymer network, alternately oxidises and reduces,
and this redox cycle modulates the polymer–solvent χ parameter (via the
coupling constant χ*), producing self-sustained swelling/deswelling waves.

## Scope of this reimplementation

Reproduces **Case I** boundary condition (small-IC + left edge pinned, as
in Figs 5(I) and 6 of the paper) on a 20×40 lattice with χ* = 0.105.
Goal: show the travelling oxidation wave in v and the accompanying
swelling wave in φ, matching the qualitative structure of Fig 6(B).

**Not implemented (yet):**
- Case II expansion conditions (Figs 9–11).
- Removal of boundary constraints (Fig 12, sample migration).
- The "drifting spiral tip" morphologies seen in the 20×40 Case II sample
  (Fig 11) — these require a larger, unconstrained domain.
- 3D scroll waves — those are the later 2008 PRE paper / the user's CUDA code.

## Layout

```
glsm/
  params.py       # physical constants from Tables I and II
  reaction.py     # modified Oregonator F(u,v,φ), G(u,v,φ) — eqs 6, 7
  mechanics.py    # osmotic pressure, nodal force, mobility — eqs 24, 25, 48, 53
  diffusion.py    # interdiffusion flux J_s^(p), J_s^(u) — eqs 60, 61
  simulator.py    # integration loop + boundary conditions
runs/
  run_case_I.py           # main experiment: χ* = 0.105 chemoresponsive, 20×40
  run_case_I_chi_star_0.py # sanity: χ* = 0 nonresponsive (Fig 5(I))
plots/
  plot_cross_section.py   # x-t density plot along central horizontal line
```

## Units and conventions

- Time in units of `T_0 = (k3·H·A)^-1 ≈ 1 s`.
- Length in units of `L_0 = sqrt(Du · T_0) ≈ 40 μm`, so one undeformed
  lattice element is `Δ = 1` (undeformed) with thickness `λ⊥·H_0`.
- Dimensionless fields `u = [HBrO2]/X_0`, `v = [Mox]/Z_0` per Tyson–Fife
  scaling.
- Nodes indexed as `(k, l)` with k spanning height, l spanning width.
  Element `m=(k,l)` has its lower-left corner at node `(k,l)`.

## Running

```bash
cd /media/b418/Wangyj/PINN/yashin2007_2d_glsm
# sanity (χ*=0, gel should not swell/deswell rhythmically):
python runs/run_case_I_chi_star_0.py
# main run (travelling waves of oxidation + swelling):
python runs/run_case_I.py
```
