/* st_pipeline.h — the one call flight software makes.
 *
 * Wires the five stages together:
 *   image -> centroids -> vectors -> histogram -> network -> shortlist
 *
 * Everything it needs is in st_pipeline_t, which you declare once, statically.
 */
#ifndef ST_PIPELINE_H
#define ST_PIPELINE_H

#include "st_types.h"
#include "st_centroid.h"
#include "st_feature.h"
#include "st_mlp.h"
#include "st_select.h"

typedef struct {
    /* Configuration, set once at init. */
    st_camera_t     cam;
    st_bins_t       bins;
    st_model_t      model;
    st_select_cfg_t sel;
    float           thresh_sigma;

    /* Scratch. Not part of the public contract; do not read these except for
     * debugging and telemetry. */
    st_centroid_ws_t ws;
    st_star_t        stars[ST_MAX_BLOBS];
    st_vec3_t        vecs[ST_MAX_STARS];
    float            hist[ST_NUM_BINS];
    float            prob[ST_NUM_CELLS];

    /* Telemetry from the last run — worth downlinking. */
    int   last_n_detected;
    int   last_n_used;
    float last_background;
    float last_sigma;
} st_pipeline_t;

/* Populate with the compiled-in model and default thresholds. You still must
 * set p->cam to your calibrated camera parameters afterwards. */
void st_pipeline_init(st_pipeline_t *p);

/* Run one frame. Returns ST_OK even when the result is a rejection — check
 * out->rejected. Returns ST_ERR_TOO_FEW only when the frame had too few stars
 * to form any feature at all. */
st_status_t st_pipeline_run(st_pipeline_t *p,
                            const uint8_t *img,
                            st_shortlist_t *out);

#endif /* ST_PIPELINE_H */
