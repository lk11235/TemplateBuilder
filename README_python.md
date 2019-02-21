Template builder python implementation
======================================

The main script for the python implementation is [scripts/buildTemplates.py](scripts/buildTemplates.py).

This is meant as a partial replacement for [Jean-Baptiste's template builder](README).
It doesn't have all the functionality - most notably, no smoothing.  However, for templates that don't need to
be smoothed it works much faster, and when reweighting from multiple samples it automatically averages based
on the statistics of each sample.

Improvements with respect to Jean-Baptiste's code are marked with :new:.  Limitations are marked with :warning:.

The syntax is inspired by Jean-Baptiste's json syntax, with some things removed.  Here's the full syntax:

- `inputDirectory`: where all the root files are stored, can be `/` if you want to give the full paths later.
- `outputFile`: where the final templates should be saved
- `templates`: list of templates, where each one can contain:
  - `binning`:
    - `bins`: xbins, xmin, xmax, ybins, ymin, ymax, zbins, zmin, zmax
    - :warning: only fixed binning is allowed and there's no need to give the `type`
  - `files`:
    - root files to fill from
  - `name`: the name (and title) of the output histogram
  - `postprocessing`: list of postprocessing configs, where each one can contain:
    - `type`: `mirror`, `rescale`, or `floor`
    - `antisymmetric`: for `mirror`, either `true` or `false`
    - `factor`: for `rescale`
    - `factorerror`: :new: for `rescale`, added in quadrature to the statistical bin errors.  This is not exactly right because it loses the covariance between bins.
    - `floorvalue`: for `floor`, what to set the zero bins to
    - `floorerror`: :new: for `floor`, the error to set for the floored bins.  If it's 0 (the default), then it finds the bin with the biggest error/value and uses the error from that bin
  - `selection`: a cut to be used on the tree, which can be any expression that TTree::Draw can interpret
  - `tree`: the tree name in the input root files
  - `variables`: a list of three variables from the tree, each of which can be any expression that TTree::Draw can interpret
  - `weight`: a weight to be used for filling, which can be any expression that TTree::Draw can interpret
  - :warning: all other features are removed, including:
    - `conserveSumOfWeights`, which is always true
    - `filloverflows`, which is always true
    - smoothing
    - `templatesum` - actually there's really no need for this anymore, as you can fill the interference templates directly from the tree.  The primary reason not to do this was to not introduce biases by smoothing.

:new: This code only loops through each tree once, and fills all the templates from the tree at the same time.
In addition, it deactivates all the branches that aren't needed.  This makes filling the histograms a lot faster.

:new: It also combines the different root files taking their effective statistics into account,
using the following procedure:

  1. :new::warning: Each individual tree should be normalized as if it's filling
     the template independently.  For example if you expect the final template to have an integral of 5 and are filling
     it from two trees, normalize each tree to 5.  (Normally what you want to do is normalize the whole sample,
     including events that fail any cuts, to the cross section, and then the tree normalization comes out correct
     automatically.  But this is just an example.)
  2. From each tree, fill a histogram.  If there are any other templates being made at the same time that use the
     same tree, histograms for those templates will also be filled now.
  3. Then, construct the final histogram and loop over the bins.  For each bin, loop over the individual histograms
     and take the weighted average of bin contents in those histograms, weighting by 1/error^2.  This means
     that if you try to combine a small sample and a large sample, the difference in stats will be automatically
     taken into account.  Similarly if you reweight from one hypothesis to another.  The error is set by
     1/variance = sum(1/individual variance)

:new: mirroring also weights by 1/error^2 in the two bins, and sets the bin errors correctly.

:new: flooring also sets the error for the floored bins, as described above in `floorerror`
