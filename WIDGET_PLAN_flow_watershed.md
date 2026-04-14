# Flow-Guided Watershed Cell Segmentation Widget

## Overview
A cell segmentation widget that expands nuclear labels (from s00) into full cell masks using cellpose flow fields (from s1b) to guide and scale the expansion velocity. Nuclei expand uniformly until they touch, with expansion velocity modulated by cellpose flow direction.

## Goal
Transform point-like nuclear labels into cell-boundary segmentation by leveraging the flow field information from cellpose (which contains directional information about membrane location relative to the cell interior).

## Input Data
- **Nuclear labels** (s00): Integer label map where each connected component is a tracked nucleus
- **Cellpose flow field** (s1b): 2D vector field (shape: [H, W, 2]) representing flow direction at each pixel
- **Cellpose probability** (s1b): Confidence map (shape: [H, W]) for cellpose predictions
- **Cell image** (reference, optional): Original membrane image for visualization context

## Algorithm Approach

### Core Concept: Flow-Scaled Watershed Expansion
1. **Initialization**: Each nucleus is a seed point/region (from the label layer)
2. **Expansion**: For each pixel not yet assigned to a cell:
   - Compute distance to nearest nucleus center
   - Look up local cellpose flow vector at that pixel
   - Compute **flow bias**: dot product between distance-direction and flow vector
   - Scale expansion velocity using: `velocity = 1 + λ * flow_bias`
     - When flow points outward from nucleus → boost velocity (positive bias)
     - When flow points inward → reduce velocity (negative bias)
     - When flow is perpendicular → neutral (no bias)
3. **Assignment**: Assign pixel to nearest nucleus, accounting for flow-scaled distances
4. **Blending**: Mix between pure uniform expansion (λ=0) and flow-guided expansion (λ=1)

### Implementation Strategy
- Use scipy's `distance_transform_edt` for initial distance map
- Iterate expansion with flow-scaled velocity field
- Consider using level-set or fast-marching approaches for efficiency
- Or: morphological watershed with flow-modulated "height" map

**Option A (Simpler)**: Flow-modulated distance map
- Create distance field from nuclei
- Modulate with flow dot product to create an "effective distance"
- Apply watershed on modulated distance

**Option B (More sophisticated)**: Iterative flow-guided expansion
- Frame as a velocity field problem
- Expand boundary pixels proportional to flow-guided velocity
- Continue until no unassigned pixels remain or convergence

## Tunable Parameters

### Primary Parameters
- **flow_scale (0.0 - 2.0, default 1.0)**: Blend factor
  - 0.0 = pure uniform expansion (ignore flow)
  - 1.0 = full flow influence
  - >1.0 = amplify flow effect
  
- **cellpose_prob_threshold (0.0 - 1.0, default 0.0)**: Mask out low-confidence regions
  - Pixels below threshold treated as "don't expand here"
  - Allows respecting cellpose's own uncertainty

### Secondary Parameters
- **expand_method**: Choice of "distance_field" or "iterative" (advanced)
- **flow_smoothing_sigma (0.0 - 3.0, default 0.0)**: Smooth flow field to reduce noise artifacts
- **distance_metric**: "euclidean" or "geodesic" (geodesic = respect boundaries)

### Preview Parameters
- **show_nuclei**: Toggle nuclear labels on/off
- **show_flow_vectors**: Overlay flow field as arrows
- **show_confidence**: Show cellpose probability heatmap
- **contour_thickness**: Outline width for preview

## Output
- **Cell labels** (integer label map): Each pixel assigned to nearest nucleus
- **Expansion quality metrics** (optional):
  - Coverage: % of image assigned to a cell
  - Mean expansion distance per nucleus
  - Flow alignment: How well expansion follows flow vectors

## Implementation Outline

### Phase 1: Core Algorithm
1. Load nuclear labels, cellpose flow, cellpose probability
2. Compute nuclear centroids and initialize seeds
3. Compute base distance transform from nuclei
4. Compute flow dot products (flow · gradient_direction)
5. Create modulated distance field: `dist_eff = dist - λ * flow_bias`
6. Apply watershed/morphological operators to assign pixels

### Phase 2: Widget UI
1. Create widget class inheriting from base widget
2. Add parameter controls (sliders for flow_scale, prob_threshold, etc.)
3. Connect to preview pipeline
4. Implement live preview update on parameter change

### Phase 3: Visualization
1. Display nuclear labels (colored)
2. Overlay cell boundaries (contours)
3. Optional: show flow field overlay
4. Optional: show confidence heatmap
5. Display coverage/quality metrics

### Phase 4: Output & Integration
1. Save cell labels to output file
2. Add to stage metadata
3. Consider next-stage integration (if any)

## File Structure
```
src/ultrack_wrapper/
├── widgets/
│   └── flow_watershed.py       # Main widget class
├── stages/
│   └── s02_flow_watershed.py   # Stage orchestrator (if needed)
├── processing/
│   └── flow_watershed_processing.py  # Core algorithm
└── WIDGET_PLAN_flow_watershed.md     # This file
```

## Questions for Refinement
- [ ] Should we use iterative expansion or distance field approach?
- [ ] Is cellpose probability already thresholded, or should widget apply threshold?
- [ ] Should output be hard labels or soft probabilities?
- [ ] Do we need to handle nuclear touches/overlaps specially?
- [ ] Should we preserve nuclear boundaries or allow complete replacement?
- [ ] Preview mode: run on full image or ROI/subset for speed?

## Dependencies
- scipy (distance_transform_edt, watershed, ndimage)
- numpy
- scikit-image (maybe for morphology)
- napari (for visualization, if interactive preview needed)

## Success Criteria
- ✓ Cell segmentation respects nuclear boundaries
- ✓ Expansion follows cellpose flow field
- ✓ Parameters tune behavior intuitively
- ✓ Preview updates smoothly (<1s for 512x512)
- ✓ Output integrates with downstream stages
