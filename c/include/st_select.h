/* st_select.h — turning 529 probabilities into a shortlist for the matcher.
 *
 * This implements the "graceful degradation of the search space" policy:
 * one cell when confident, two or three when not, and an explicit rejection
 * flag in true catastrophic failure. It deliberately never says "search the
 * whole sky" on its own; that decision belongs to the caller.
 */
#ifndef ST_SELECT_H
#define ST_SELECT_H

#include "st_types.h"

typedef struct {
    /* A cell is a candidate if its probability exceeds this. */
    float accept_threshold;
    /* If the best cell exceeds this, we return it alone and stop. */
    float confident_threshold;
    /* If the best cell is below accept_threshold, set rejected = 1. */
    /* Star richness of the current frame, used as the tie-breaker: when two
     * candidates are within tie_margin of each other in probability, prefer
     * the one the caller's richness callback says is easier to solve.
     * Set to 0 to disable tie-breaking. */
    float tie_margin;
} st_select_cfg_t;

/* Sensible starting values; tune these on held-out data, not by feel. */
void st_select_default_cfg(st_select_cfg_t *cfg);

/* Optional callback returning a "solvability" score for a cell — higher is
 * easier to solve downstream (more bright stars, fewer near-duplicate
 * triangles). Pass NULL to skip tie-breaking. `user` is passed through. */
typedef float (*st_richness_fn)(uint16_t cell_id, void *user);

/* Build the shortlist.
 *   prob    : ST_NUM_CELLS sigmoid outputs
 *   out     : filled with up to ST_MAX_CANDIDATES cells, best first
 * Returns ST_OK. Check out->rejected and out->n. */
st_status_t st_select_cells(const float *prob,
                            const st_select_cfg_t *cfg,
                            st_richness_fn richness, void *user,
                            st_shortlist_t *out);

#endif /* ST_SELECT_H */
