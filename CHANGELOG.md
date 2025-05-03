# Changelog

<!--next-version-placeholder-->

## v1.0.0 (03/05/2025)

### Feat

- Add `vfp.VFP` which does not rely on `refnx` or `refl1d` dependencies.
- Add ability to plot posterior samples via `BaseVFP.plot()`.
- Customise colours and labels in `BaseVFP.plot()`.

### Fix

- Update `refl1dVFP` to work with `refl1d` and `bumps` tagged versions > 1.
- Generalise demaglocs and demagwidths to handle n peaks. 
- Remove superfluous vfp attributes in `BaseVFP`.

### Docs

- Add three example notebooks showcasing usage of `vfp` with refnx, refl1d and how to use the `sld_constraint` parameter of the vfp classes.
- Update docstrings to match new package structure.
- Add type hints for all objects and functions.

### Tests
- Add tests for vfp calculations.

## v0.2.0 (25/03/2024)

- First release of `vfp`!