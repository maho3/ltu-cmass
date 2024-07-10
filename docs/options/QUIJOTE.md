
Generating Quijote ICs
======================

You can also use the scripts in [`quijote_wn/NgenicWhiteNoise`](./quijote_wn/NgenicWhiteNoise) to generate initial white noise fields for the Quijote simulations. Then, these can be used to seed the `ltu-cmass` gravity solvers, and further to calibrate the halo biasing models.

To generate the Quijote ICs, you must first make the executable.
```bash
cd ltu-cmass/quijote_wn/NgenicWhiteNoise
make
```
Then, edit and run `gen_quijote_ic.sh` to generate the ICs. In the below command, this script will generate e.g. $128^3$ white noise fields for the first latin-hypercube cosmology and place them in my `quijote/wn/N128` directory.
```bash
sh gen_quijote_ic.sh 128 0
```

Refitting bias models using Quijote
===================================

Then, you can run the gravity solvers in [`cmass.nbody`](./cmass/nbody) using the configuration flag `matchIC: 2` to match the Quijote ICs. 

Lastly, you can use these phase-matched density fields to refit the halo biasing parameters by first [downloading the Quijote halos](https://quijote-simulations.readthedocs.io/en/latest/halos.html) and using [cmass.bias.fit_halo_bias](./cmass/bias/fit_halo_bias.py) to fit the bias models. The configuration for this fitting is in [`cmass/conf/fit/quijote_HR.yaml`](./cmass/conf/fit/quijote_HR.yaml). The `path_to_qhalos` parameter specifies the relative path within the working directory to the Quijote halos. For example, my Quijote halos are stored as:
```yaml
+-- /path/to/working/directory
|   +-- quijote
|   |   +-- source
|   |   |   +-- Halos
|   |   |   |   +-- latin_hypercube_HR
|   |   |   |   |   +-- 0
|   |   |   |   |   +-- 1
|   |   |   |   |   +-- ...
```
and my `path_to_qhalos` would be `quijote/source/Halos/latin_hypercube_HR`.
