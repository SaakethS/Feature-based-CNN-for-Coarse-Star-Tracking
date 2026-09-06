/* pipeline_main.c — run the whole flight path on a real image, and time it.
 *
 * Usage:  st_pipeline_test <image.pgm> [fx]
 *
 * The image must be a binary (P5) 8-bit PGM of exactly ST_IMG_W x ST_IMG_H.
 * python/crosscheck.py writes one for you.
 *
 * This is the program to run on the Pi 5 to get a real timing number, and the
 * program that proves the C centroider recovers the star positions the
 * training simulator assumed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "st_pipeline.h"

/* One static instance. sizeof(st_pipeline_t) is dominated by the centroid
 * workspace: two arrays of one entry per pixel. */
static st_pipeline_t g_pipe;
static uint8_t g_img[ST_IMG_W * ST_IMG_H];

static int read_pgm(const char *path, uint8_t *img)
{
    FILE *f = fopen(path, "rb");
    int w, h, maxv, c;
    if (!f) { perror(path); return -1; }
    if (fscanf(f, "P5") == EOF) { fclose(f); return -1; }
    /* Skip whitespace and any comment lines before each numeric field. */
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); }
        else if (c == ' ' || c == '\n' || c == '\t' || c == '\r') continue;
        else { ungetc(c, f); break; }
    }
    if (fscanf(f, "%d %d %d", &w, &h, &maxv) != 3) { fclose(f); return -1; }
    fgetc(f);   /* single whitespace byte after the header */
    if (w != ST_IMG_W || h != ST_IMG_H || maxv != 255) {
        fprintf(stderr, "expected %dx%d 8-bit, got %dx%d maxval %d\n",
                ST_IMG_W, ST_IMG_H, w, h, maxv);
        fclose(f);
        return -1;
    }
    if (fread(img, 1, (size_t)w * (size_t)h, f) != (size_t)w * (size_t)h) {
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    st_shortlist_t sl;
    double t0, t1;
    int i;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <image.pgm> [fx]\n", argv[0]);
        return 1;
    }
    if (read_pgm(argv[1], g_img) != 0) return 1;

    st_pipeline_init(&g_pipe);
    if (argc >= 3) {
        float fx = (float)atof(argv[2]);
        g_pipe.cam.fx = fx;
        g_pipe.cam.fy = fx;
    }

    printf("pipeline state: %.1f KiB static\n", sizeof(g_pipe) / 1024.0);

    t0 = now_ms();
    {
        st_status_t rc = st_pipeline_run(&g_pipe, g_img, &sl);
        t1 = now_ms();
        if (rc != ST_OK) {
            fprintf(stderr, "pipeline returned %d\n", rc);
            return 1;
        }
    }

    printf("background %.1f  sigma %.2f\n",
           g_pipe.last_background, g_pipe.last_sigma);
    printf("detected %d stars, used %d\n",
           g_pipe.last_n_detected, g_pipe.last_n_used);
    /* Emit the centroids actually used, so crosscheck.py can match them
     * against the positions the renderer drew and measure the bias. */
    printf("STARS %d\n", g_pipe.last_n_used);
    for (i = 0; i < g_pipe.last_n_used; ++i)
        printf("S %.6f %.6f %.6f\n", (double)g_pipe.stars[i].x,
               (double)g_pipe.stars[i].y, (double)g_pipe.stars[i].flux);

    printf("histogram:");
    for (i = 0; i < ST_NUM_BINS; ++i) printf(" %.4f", g_pipe.hist[i]);
    printf("\n");

    if (sl.rejected) {
        printf("REJECTED: no cell cleared the accept threshold\n");
    } else {
        printf("shortlist (%d cells):", sl.n);
        for (i = 0; i < sl.n; ++i)
            printf(" %u(%.3f)", (unsigned)sl.cell_id[i],
                   (double)sl.confidence[i]);
        printf("\n");
    }
    printf("elapsed %.2f ms\n", t1 - t0);

    /* Repeat runs give a stable per-frame number once the caches are warm. */
    t0 = now_ms();
    for (i = 0; i < 20; ++i) st_pipeline_run(&g_pipe, g_img, &sl);
    t1 = now_ms();
    printf("mean over 20 runs: %.2f ms/frame\n", (t1 - t0) / 20.0);
    return 0;
}
