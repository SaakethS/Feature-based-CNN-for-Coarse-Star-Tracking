/* st_config.h — compile-time limits for the flight pipeline.
 *
 * Everything in this pipeline is statically sized. There is no malloc()
 * anywhere in the C code, so the worst-case memory footprint is known at
 * compile time and cannot fragment or fail in orbit.
 *
 * If you change any value here you MUST regenerate model_params.h and
 * re-run the cross-check, because several of these constants (ST_MAX_STARS,
 * ST_NUM_BINS, ST_NUM_CELLS) are baked into the trained model.
 */
#ifndef ST_CONFIG_H
#define ST_CONFIG_H

/* ---- Image geometry ---------------------------------------------------- */
/* Detector size in pixels. Set these to your actual sensor. */
#define ST_IMG_W 1280
#define ST_IMG_H 960

/* ---- Star detection ---------------------------------------------------- */
/* Maximum number of connected bright regions ("blobs") we will track while
 * scanning the image. Beyond this, extra blobs are discarded. */
#define ST_MAX_BLOBS 512

/* A blob must contain at least this many pixels to count as a star. Rejects
 * single-pixel hot pixels and most cosmic-ray hits. */
#define ST_MIN_BLOB_PIXELS 3

/* A blob larger than this is rejected as stray light / a planet / smear. */
#define ST_MAX_BLOB_PIXELS 400

/* Flood-fill stack depth. A blob can never legitimately need more than a few
 * hundred entries given ST_MAX_BLOB_PIXELS, so a small fixed stack is enough
 * and a blob that overflows it is by definition one we would reject anyway.
 * Sizing this to the image (rather than to the blob limit) would cost several
 * megabytes of static RAM for no benefit. */
#define ST_FILL_STACK 4096

/* Detection threshold: a pixel is "bright" if
 *     value > background_median + ST_THRESH_SIGMA * background_noise_sigma
 * This is the number the training-time domain randomization must also vary. */
#define ST_THRESH_SIGMA 5.0f

/* ---- Feature extraction ------------------------------------------------ */
/* We keep only the N brightest detected stars. The pair count grows as
 * N*(N-1)/2, so this constant is what bounds the runtime cost of the whole
 * feature stage. N = 24 -> 276 pairs. */
#define ST_MAX_STARS 24
#define ST_MAX_PAIRS ((ST_MAX_STARS * (ST_MAX_STARS - 1)) / 2)

/* Number of histogram bins in the pairwise-angular-distance feature.
 * Must match the training pipeline exactly. */
#define ST_NUM_BINS 25

/* ---- Sky partition ----------------------------------------------------- */
/* Number of sky cells the classifier chooses among (GLAS convention). */
#define ST_NUM_CELLS 529

/* Maximum number of candidate cells returned to the pattern matcher. If more
 * than this clear the confidence threshold, we are in a degraded state and
 * return the top ST_MAX_CANDIDATES only. */
#define ST_MAX_CANDIDATES 8

/* ---- Network sizing ---------------------------------------------------- */
/* Hidden layer widths. Must match the trained model in model_params.h. */
#define ST_H1 96
#define ST_H2 96

#endif /* ST_CONFIG_H */
