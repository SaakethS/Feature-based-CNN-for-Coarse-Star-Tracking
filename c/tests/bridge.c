/* Host-only ctypes bridge; production sources remain dependency-free C99.
 * Static detector workspace means calls must be serialized (not thread-safe). */
#include "st_centroid.h"
#include "st_feature.h"
#include "st_select.h"
#include "st_pipeline.h"
#include <string.h>
int st_bridge_cells(void);
int st_bridge_bins(void);
int st_bridge_detect(const uint8_t *, float, float *, int *);
int st_bridge_select(const float *, float, float, int *);
int st_bridge_pipeline(const uint8_t *, const float *, float, float *, float *, int *);
int st_bridge_bins(void) { return ST_NUM_BINS; }
int st_bridge_cells(void) { return ST_NUM_CELLS; }
int st_bridge_detect(const uint8_t *img, float threshold, float *xyz, int *counts)
{
    static st_centroid_ws_t ws;
    st_star_t stars[ST_MAX_BLOBS];
    int n = 0, used, i;
    st_status_t rc = st_detect_stars(img, &ws, threshold, stars, ST_MAX_BLOBS, &n);
    counts[0] = n; counts[1] = 0;
    if (rc != ST_OK) return (int)rc;
    if (n < ST_MIN_STARS) return ST_ERR_TOO_FEW;
    used = st_select_brightest(stars, n, ST_MAX_STARS);
    for (i = 0; i < used; ++i) {
        xyz[3*i] = stars[i].x; xyz[3*i+1] = stars[i].y;
        xyz[3*i+2] = stars[i].flux;
    }
    counts[1] = used;
    return ST_OK;
}
int st_bridge_select(const float *p, float accept, float confident, int *ids)
{
    st_select_cfg_t cfg; st_shortlist_t out; int i;
    st_select_default_cfg(&cfg); cfg.accept_threshold=accept;
    cfg.confident_threshold=confident;
    if (st_select_cells(p, &cfg, NULL, NULL, &out) != ST_OK) return -1;
    for (i=0; i<out.n; ++i) ids[i]=(int)out.cell_id[i];
    return (int)out.n;
}
int st_bridge_pipeline(const uint8_t *img, const float *camera, float threshold,
                       float *prob, float *hist, int *counts)
{
    static st_pipeline_t p; st_shortlist_t sl; st_status_t rc;
    st_pipeline_init(&p);
    p.cam.fx=camera[0]; p.cam.fy=camera[1]; p.cam.cx=camera[2];
    p.cam.cy=camera[3]; p.cam.k1=camera[4]; p.cam.k2=camera[5];
    p.thresh_sigma=threshold;
    rc=st_pipeline_run(&p,img,&sl);
    counts[0]=p.last_n_detected; counts[1]=p.last_n_used;
    if (rc != ST_OK) return (int)rc;
    memcpy(prob,p.prob,sizeof(p.prob)); memcpy(hist,p.hist,sizeof(p.hist));
    return ST_OK;
}
