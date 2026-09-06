#include "st_pipeline.h"
#include "st_camera.h"
#include "model_params.h"
#include <string.h>

void st_pipeline_init(st_pipeline_t *p)
{
    if (!p) return;
    memset(p, 0, sizeof(*p));

    /* Model and bin edges come from the generated header, which is written by
     * python/export_model.py. That file is the single source of truth shared
     * by training and flight — never hand-edit it. */
    st_model_params_load(&p->model, &p->bins);

    st_select_default_cfg(&p->sel);
    p->thresh_sigma = ST_THRESH_SIGMA;

    /* Placeholder camera. REPLACE with your calibrated values before flight;
     * a wrong focal length silently rescales every angle and the histogram
     * will not match anything the network was trained on. */
    p->cam.fx = 1800.0f;
    p->cam.fy = 1800.0f;
    p->cam.cx = (float)ST_IMG_W * 0.5f;
    p->cam.cy = (float)ST_IMG_H * 0.5f;
    p->cam.k1 = 0.0f;
    p->cam.k2 = 0.0f;
}

st_status_t st_pipeline_run(st_pipeline_t *p,
                            const uint8_t *img,
                            st_shortlist_t *out)
{
    st_status_t rc;
    int n_detected = 0, n_used;

    if (!p || !img || !out) return ST_ERR_NULL;

    /* 0. Background telemetry. Cheap, and the two numbers most worth
     *    downlinking when a frame fails: a rising background is the signature
     *    of stray light from the Sun, Moon, or Earth limb. */
    (void)st_estimate_background(img, &p->last_background, &p->last_sigma);

    /* 1. Image -> star centroids. ST_ERR_OVERFLOW is not fatal: it means the
     *    frame had more blobs than the table holds, and we keep what we got. */
    rc = st_detect_stars(img, &p->ws, p->thresh_sigma,
                         p->stars, ST_MAX_BLOBS, &n_detected);
    if (rc != ST_OK && rc != ST_ERR_OVERFLOW) return rc;
    p->last_n_detected = n_detected;

    if (n_detected < 2) return ST_ERR_TOO_FEW;

    /* 2. Keep the brightest stars only. This bounds the pair count and, more
     *    importantly, must match exactly what training did. */
    n_used = st_select_brightest(p->stars, n_detected, ST_MAX_STARS);
    p->last_n_used = n_used;

    /* 3. Pixels -> unit direction vectors. */
    rc = st_pixels_to_vectors(&p->cam, p->stars, n_used, p->vecs);
    if (rc != ST_OK) return rc;

    /* 4. Vectors -> rotation-invariant histogram. */
    rc = st_histogram(p->vecs, n_used, &p->bins, p->hist);
    if (rc != ST_OK) return rc;

    /* 5. Histogram -> per-cell probabilities. */
    rc = st_mlp_forward(&p->model, p->hist, p->prob);
    if (rc != ST_OK) return rc;

    /* 6. Probabilities -> shortlist for the pattern matcher. */
    return st_select_cells(p->prob, &p->sel, NULL, NULL, out);
}
