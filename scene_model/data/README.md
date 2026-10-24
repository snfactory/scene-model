# scene_model/data

`scala_base_vectors.pkl` (not tracked in git; ignored by `.gitignore`, install it by hand): the SCALA pupil-mask base vectors used by
`scene_model.scala_mask.ScalaMaskPsfElement`
(`SnifsFourierSceneModel(inst_core='scala_mask')`, `extract-star2
--inst-core scala_mask`). It is a byte-for-byte copy of Daniel Kuesters'
`base_vectors_python2.pkl` (received 2026-10-23; path in his copy of the code:
`/home/scala/Desktop/Kyles_PSF/scene_model/scene_model/base_vectors_python2.pkl`),
a Python-2 pickle of a (324, 2) float64 numpy array; `scala_mask.load_base_vectors`
reads it directly (`encoding='latin1'`), or an `.npy` copy of the array.
`SCENE_MODEL_SCALA_BASE_VECTORS` overrides the location.

Contents (checked 2026-10-24): 324 = 18 x 18 (x, y) pairs, 163 distinct,
symmetric under negation, |b| <= 1.80; the origin (0, 0) appears 18 times and
is the first entry. The structure is that of the pairwise difference vectors of
18 apertures (the zero-lag terms first); the sum of the aperture OTF over all
324 entries is 18 at zero frequency. Production used only the first entry
(`max_vectors=1`), for which the OTF is normalized.
