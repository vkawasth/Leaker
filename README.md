
SEAL (facebook research (https://github.com/facebookresearch/SEAL_OGB) is used as baseline for link prediction using node attributes on Rat brain (ogb-vessel) dataset.
My enhancements are adding a GV/BV bracket to resolve singularities on entire brain graph to allow entropy to flow freely and settle at curvatures leading to maximum stability.
This extracted features of maximum importance, which were then used to reverse map voxels.

This method does not flow messages as messages based embedding identification can only detect local similarities or local links, so there are no NN to train with a loss function to detect large functional structures.

Here I flow Shannon Entropy (-plogp) where channels lengths and cross section areas are considered into ease of flow considerations in BV brackets. Jacobians = 0, is a singularity identifier. I flow entropy via these points without allowing them to degrade to 0 (issue of vanishing gradients).
