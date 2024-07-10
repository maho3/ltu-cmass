
Running filtering
=================

Filtering and weighting (of galaxy positions) is not a default step in the pipeline. To apply filtering, you would use the `cmass.filter` module, after the `ngc_selection` step but before summary measurement. For example,
```bash
...
python -m cmass.survey.ngc_selection
python -m cmass.filter.single_filter +filter=random
python -m cmass.summaries.Pk +filter=random
```
This would generate, e.g. ra/dec/z `rdz0_filter.npy` and weight `rdz0_filter_weight.npy` files within the `obs/filtered` subdirectory. They will then be automatically loaded into the summaries module, if the filter configuration is included.
