# Changelog

<!--next-version-placeholder-->

## v1.0.0 (26/07/2025)

- First stable release of `vfp`.

### Major (breaking change)

- Parameters passed to an instance of a child class of `BaseVFP` are now stored in `vfp_attrs` dataclass, rather than being an attribute of the child class.

### Minor (feat)

- Add `vfp.VFP` which does not rely on `refnx` or `refl1d` dependencies.
- Add ability to plot posterior samples via `BaseVFP.plot`.
- Customise which plots returned by `BaseVFP.plot`
- Customise colours and labels in `BaseVFP.plot`.

### Patch (fix)

- Update `refl1dVFP` to work with `refl1d` and `bumps` tagged versions > 1.
- Generalise demaglocs and demagwidths to handle :math:`n` peaks.
- Remove superfluous vfp attributes in `BaseVFP`.

### Docs

- Add three example notebooks showcasing usage of `vfp` with refnx, refl1d and how to use the `sld_constraint` parameter of the vfp classes.
- Update docstrings to match new package structure.
- Add type hints for all objects and functions.

### Tests
- Add tests for vfp calculations and some tests for vfp functions.

## v0.2.0 (25/03/2024)

- alpha release of `vfp`!
