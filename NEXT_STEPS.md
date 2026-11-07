# Release next steps

All three training jobs completed on 2026-11-06. The selected models are MRI ConvNeXt-Tiny, grouped CT ConvNeXt-Tiny, and the updated NIH ResNet-18. See README.md, RESULTS.md and MODEL_CARDS.md for the final release record rather than the old local queue commands.

1. Review the reported results and limitations before publishing. MRI did not improve on the historical baseline's held-out macro-F1; CT is exploratory with unknown patient overlap; NIH comparisons involve changed training/selection protocols.
2. Publish only the prepared source-and-aggregate-results archive. The original `data/`, `runs/`, `.venv/`, old nested source copy, prediction CSVs and checkpoints are excluded. MIT covers the source; it grants no rights to third-party datasets or pretrained weights.
3. The public source repository is now https://github.com/bamboo081/medscan. Check GitHub Actions before adding a version tag or creating a GitHub release.
4. The local browser workflow was smoke-tested using a synthetic noise image. Use only authorized examples in any additional screenshots; synthetic predictions have no medical meaning.
5. Further model decisions require new validation work. These test partitions have now been examined; do not tune against their scores or call a repeated evaluation a new independent test.

For future X-ray retraining, `--cache-dir data/cache/nih_224` enables lossless on-demand preprocessing caching. It contains only train/validation inputs accessed by training. The cache uses roughly 16.5 GiB, remains local, and can be removed when no job is using it. Originals are still needed. Existing historical checkpoints and manifests remain preserved locally.
