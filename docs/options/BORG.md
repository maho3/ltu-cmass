Installing BORG 
===============

We use BORG solely to run the BORG-LPT and BORG-PM gravity solvers, in `cmass.nbody.borglpt`, `cmass.nbody.borgpm`, and `cmass.nbody.borgpm_lc`. If you don't want to use these features, you can skip this section.

Install the public version of borg with:
```bash
pip install --no-cache-dir aquila-borg
```
The build process for this package may take a while (~20 minutes). Note, this public version of BORG lacks several features, such as BORG-PM simulators and CLASS transfer functions.

For access to the private BORG implementation, consider joining the [Aquila consortium](https://www.aquila-consortium.org/) :).
