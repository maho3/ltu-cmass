
Running filtering
=================

Filtering and weighting (of galaxy positions) is not a default step in the pipeline. To apply filtering, you would use the `cmass.filter` module, after the `cmass.survey` step but before summary measurement. For example,
```bash
...
python -m cmass.survey.ngc_selection
python -m cmass.filter.single_filter +filter=random
python -m cmass.summary.Pk +filter=random
```
This would generate new `hod000_aug000.h5` files within the `filter` subdirectory, condaining ra/dec/z and weights. They will then be automatically loaded into the `summary` module, if the filter configuration is included.
