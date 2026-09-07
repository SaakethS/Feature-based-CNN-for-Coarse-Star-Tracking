#include "st_select.h"
#include <string.h>
#include <math.h>

void st_select_default_cfg(st_select_cfg_t *cfg)
{
    if (!cfg) return;
    /* These are starting points, not tuned values. Set them from a
     * precision/recall sweep on held-out synthetic attitudes: pick the
     * accept_threshold that keeps the true cell in the shortlist at your
     * required rate, then the confident_threshold that lets you drop to a
     * single cell without losing it. */
    cfg->accept_threshold    = 0.15f;
    cfg->confident_threshold = 0.90f;
    cfg->tie_margin          = 0.05f;
}

st_status_t st_select_cells(const float *prob,
                            const st_select_cfg_t *cfg,
                            st_richness_fn richness, void *user,
                            st_shortlist_t *out)
{
    int i, k, n = 0;

    if (!prob || !cfg || !out) return ST_ERR_NULL;
    memset(out, 0, sizeof(*out));
    out->rejected = 1u;
    if (!isfinite(cfg->accept_threshold) || cfg->accept_threshold < 0.0f ||
        cfg->accept_threshold > 1.0f || !isfinite(cfg->confident_threshold) ||
        cfg->confident_threshold < 0.0f || !isfinite(cfg->tie_margin) ||
        cfg->tie_margin < 0.0f) return ST_ERR_BAD_PARAM;
    for (i = 0; i < ST_NUM_CELLS; ++i)
        if (!isfinite(prob[i]) || prob[i] < 0.0f || prob[i] > 1.0f)
            return ST_ERR_BAD_PARAM;

    /* Selection sort of the top ST_MAX_CANDIDATES over 529 entries. This is
     * 529 * 8 comparisons — cheaper and far more predictable than sorting the
     * whole array, and it never allocates. */
    {
        uint8_t taken[ST_NUM_CELLS];
        memset(taken, 0, sizeof(taken));
        for (k = 0; k < ST_MAX_CANDIDATES; ++k) {
            int best = -1;
            for (i = 0; i < ST_NUM_CELLS; ++i) {
                if (taken[i]) continue;
                if (best < 0 || prob[i] > prob[best]) best = i;
            }
            if (best < 0) break;
            if (prob[best] < cfg->accept_threshold) break;
            taken[best] = 1u;
            out->cell_id[n]    = (uint16_t)best;
            out->confidence[n] = prob[best];
            n++;
        }
    }

    if (n == 0) {
        /* True catastrophic failure: nothing anywhere in the sky cleared the
         * bar. We report it rather than silently returning the argmax, so the
         * caller can decide whether to widen the search or wait for a better
         * frame. */
        out->n = 0;
        out->rejected = 1u;
        return ST_OK;
    }

    /* Star-richness tie-break: if the runner-up is within tie_margin of the
     * leader and is easier to solve downstream, promote it. Confidence in the
     * localization is the primary driver; richness only breaks near-ties. */
    if (richness && n >= 2 && cfg->tie_margin > 0.0f) {
        if (out->confidence[0] - out->confidence[1] <= cfg->tie_margin) {
            float r0 = richness(out->cell_id[0], user);
            float r1 = richness(out->cell_id[1], user);
            if (r1 > r0) {
                uint16_t ct = out->cell_id[0];
                float    cf = out->confidence[0];
                out->cell_id[0]    = out->cell_id[1];
                out->confidence[0] = out->confidence[1];
                out->cell_id[1]    = ct;
                out->confidence[1] = cf;
            }
        }
    }

    /* Confident enough to hand the matcher exactly one cell. */
    if (out->confidence[0] >= cfg->confident_threshold) n = 1;

    out->n = (uint8_t)n;
    out->rejected = 0u;
    return ST_OK;
}
