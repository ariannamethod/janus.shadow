/* janus.shadow.c — Shadow Attention Post-Transformer (patched super-version)
 * θ = ε + γ + αδ | attention knows what it needs
 *
 * Patches over lukas.c:
 *   - honest weightless mode: no learned random weights influence Q/K/V or logits
 *   - transformer path preserved: structural Q/K/V still drive causal attention in noweights
 *   - Q-style cold-start coherence: coherence is meaningful on the first forward
 *   - fixed need timing: drift/novelty are measured before prev_h/mem are overwritten
 *   - causal trace snapshots: logits at position i only see trace after reading <= i
 *   - bounded trigram hash probing: no infinite loop when table fills
 *   - softer neural gate with tiny training floor: Wu can learn without random dominance
 *
 * Compile: gcc lukas3_shadow_attention.c -O2 -lm -o lukas3
 * Usage:
 *   ./lukas3 --test
 *   ./lukas3 train  corpus.txt model.lukas [epochs]
 *   ./lukas3 infer  model.lukas 'prompt' [tokens]
 *   ./lukas3 --noweights corpus.txt 'prompt'
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

/* — architecture — */
#define VMAX     512
#define D        48
#define CTX      48
#define MAXTOK   65536
#define HSZ      131072

/* — shadow & training — */
#define ALPHA       0.5f
#define DK          0.85f
#define TR_DECAY    0.965f
#define TR_BOOST    0.45f
#define MEM_DECAY   0.92f
#define BETA1       0.9f
#define BETA2       0.999f
#define ADAM_EPS    1e-8f
#define LR          0.0015f

/* — logit composition — */
#define MAG_THR       0.020f
#define MAG_RANGE     0.075f
#define GATE_FLOOR    0.040f
#define WU_INIT_SCALE 0.12f
#define BIGRAM_W      0.3f
#define TRIGRAM_W     0.2f
#define TR_PENALTY    0.35f
#define NEED_BASE     0.20f
#define NEED_NOV      0.30f
#define NEED_DRF      0.20f
#define NEED_COH      0.30f
#define SCALE_INV     0.144338f /* 1/sqrt(48) */

/* — BPE — */
static int L[VMAX], R[VMAX];
static int V = 256;
static char *tok_str[VMAX];

/* — model parameters — */
static float E[VMAX][D];
static float Wq[D][D], Wk[D][D], Wv[D][D], Wo[D][D];
static float Wu[D][VMAX];

/* — shadow state — */
static float tr[VMAX];
static float mem[D];
static float prev_h[D];
static float coherence = 1.0f;
static float q_coherence = 1.0f;
static float focus = 0.0f;
static float need_val = 0.0f;
static int   TRAINED = 0;
static int   LIFE_READY = 0;

/* — corpus statistics — */
static int      bi_count[VMAX][VMAX];
static int      bi_row_sum[VMAX];
static unsigned tri_key[HSZ];
static int      tri_cnt[HSZ];

/* — activations cached for backprop — */
static int   tok_cache[CTX];
static int   T_cache;
static float x_c[CTX][D];
static float Q_c[CTX][D], K_c[CTX][D], V_c[CTX][D];
static float raw_c[CTX][CTX];
static float th_c[CTX][CTX];
static float att_c[CTX][CTX];
static float h_c[CTX][D];
static float op_c[CTX][D];
static float r_c[CTX][D];
static float prob_c[CTX][VMAX];
static float trpos_c[CTX][VMAX];
static float gate_c;

/* — Adam state — */
static float mE[VMAX][D], vE[VMAX][D];
static float mWq[D][D], vWqs[D][D];
static float mWk[D][D], vWks[D][D];
static float mWv[D][D], vWvs[D][D];
static float mWo[D][D], vWos[D][D];
static float mWu[D][VMAX], vWu[D][VMAX];
static int   adam_t = 0;

/* — gradient accumulators — */
static float gE[VMAX][D];
static float gWq[D][D], gWk[D][D], gWv[D][D], gWo[D][D];
static float gWu[D][VMAX];
static float gx[CTX][D], gQ[CTX][D], gK[CTX][D], gV[CTX][D];
static float gh[CTX][D], gop[CTX][D], gr[CTX][D];
static float gatt[CTX][CTX], gsc[CTX][CTX];

/* — RNG — */
static unsigned RNG = 0x51AD05u;
static float rfu(void) { RNG = RNG * 1664525u + 1013904223u; return ((RNG >> 8) & 0xFFFFFFu) / 16777216.0f; }
static float rr(void)  { return (rfu() * 2.0f - 1.0f) * 0.18f; }
static float clampf2(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

/* deterministic sinusoidal token identity prior */
static float basev(int t, int d) {
    return 0.055f * sinf((t + 1) * (d + 3) * 0.137f)
         + 0.035f * cosf((t + 7) * (d + 1) * 0.071f);
}

/* Weightless mode keeps the transformer skeleton alive.
 * Q/K/V are deterministic projections of basev, not random learned weights. */
static void structural_qkv(int t, int d, float *q, float *k, float *v) {
    float b0 = basev(t, d);
    float b1 = basev(t, (d + 7)  % D);
    float b2 = basev(t, (d + 17) % D);
    float b3 = basev(t, (d + 29) % D);
    *q = b0 + 0.31f * b1 - 0.11f * b3;
    *k = b0 + 0.23f * b2 + 0.07f * b3;
    *v = b0 + 0.19f * b1;
}

/* ============================================================ */
/*  BPE                                                          */
/* ============================================================ */

static void bpe_init(void) {
    for (int i = 0; i < 256; i++) {
        tok_str[i] = malloc(2);
        if (!tok_str[i]) { fprintf(stderr, "malloc failed\n"); exit(1); }
        tok_str[i][0] = (char)i;
        tok_str[i][1] = 0;
    }
    for (int i = 256; i < VMAX; i++) { L[i] = R[i] = -1; tok_str[i] = NULL; }
    V = 256;
}

static int merge_pass(int *t, int n, int a, int b, int id) {
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (i < n - 1 && t[i] == a && t[i+1] == b) { t[j++] = id; i++; }
        else t[j++] = t[i];
    }
    return j;
}

static int bpe_train(const unsigned char *s, int n, int *out_t, int max_merges) {
    if (n > MAXTOK) n = MAXTOK;
    for (int i = 0; i < n; i++) out_t[i] = s[i];

    static int pair_cnt[VMAX][VMAX];
    for (int m = 0; m < max_merges && V < VMAX; m++) {
        memset(pair_cnt, 0, sizeof pair_cnt);
        for (int i = 0; i < n - 1; i++) pair_cnt[out_t[i]][out_t[i+1]]++;

        int ba = -1, bb = -1, bc = 1;
        for (int a = 0; a < V; a++)
            for (int b = 0; b < V; b++)
                if (pair_cnt[a][b] > bc) { bc = pair_cnt[a][b]; ba = a; bb = b; }
        if (ba < 0) break;

        int id = V++;
        L[id] = ba; R[id] = bb;
        int la = (int)strlen(tok_str[ba]), lb = (int)strlen(tok_str[bb]);
        tok_str[id] = malloc((size_t)la + (size_t)lb + 1u);
        if (!tok_str[id]) { fprintf(stderr, "malloc failed\n"); exit(1); }
        memcpy(tok_str[id], tok_str[ba], (size_t)la);
        memcpy(tok_str[id] + la, tok_str[bb], (size_t)lb);
        tok_str[id][la + lb] = 0;

        n = merge_pass(out_t, n, ba, bb, id);
    }
    return n;
}

static int bpe_encode(const unsigned char *s, int n, int *out_t) {
    if (n > MAXTOK) n = MAXTOK;
    for (int i = 0; i < n; i++) out_t[i] = s[i];
    for (int id = 256; id < V; id++) n = merge_pass(out_t, n, L[id], R[id], id);
    return n;
}

static void emit_tok(int t, FILE *f) {
    if (t < 256) fputc(t, f);
    else { emit_tok(L[t], f); emit_tok(R[t], f); }
}

static int emit_tok_buf(int t, char *b, int p) {
    if (t < 256) { b[p] = (char)t; return p + 1; }
    p = emit_tok_buf(L[t], b, p);
    p = emit_tok_buf(R[t], b, p);
    return p;
}

/* ============================================================ */
/*  Corpus statistics                                            */
/* ============================================================ */

static void stats_reset(void) {
    memset(bi_count, 0, sizeof bi_count);
    memset(bi_row_sum, 0, sizeof bi_row_sum);
    memset(tri_key, 0, sizeof tri_key);
    memset(tri_cnt, 0, sizeof tri_cnt);
}

static unsigned tri_pack(int a, int b, int c) {
    return (((unsigned)a << 18) | ((unsigned)b << 9) | (unsigned)c) + 1u;
}

static void tri_add(int a, int b, int c) {
    unsigned k = tri_pack(a, b, c);
    unsigned h = (k * 2654435761u) & (HSZ - 1);
    unsigned start = h;
    while (tri_key[h] && tri_key[h] != k) {
        h = (h + 1) & (HSZ - 1);
        if (h == start) return;
    }
    tri_key[h] = k;
    if (tri_cnt[h] < INT_MAX) tri_cnt[h]++;
}

static int tri_get(int a, int b, int c) {
    unsigned k = tri_pack(a, b, c);
    unsigned h = (k * 2654435761u) & (HSZ - 1);
    unsigned start = h;
    while (tri_key[h] && tri_key[h] != k) {
        h = (h + 1) & (HSZ - 1);
        if (h == start) return 0;
    }
    return tri_key[h] ? tri_cnt[h] : 0;
}

static void stats_build(int *seq, int n) {
    stats_reset();
    for (int i = 0; i < n - 1; i++) { bi_count[seq[i]][seq[i+1]]++; bi_row_sum[seq[i]]++; }
    for (int i = 0; i < n - 2; i++) tri_add(seq[i], seq[i+1], seq[i+2]);
}

/* ============================================================ */
/*  Model init                                                   */
/* ============================================================ */

static void model_init(void) {
    RNG = 0x51AD05u;
    for (int v = 0; v < VMAX; v++)
        for (int d = 0; d < D; d++) E[v][d] = rr();

    /* Q/K are born aligned, not identical: coherence exists before learning.
     * This keeps the transformer parts fully present while avoiding a cold random Q/K split. */
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            float id = (i == j) ? 1.0f : 0.0f;
            float qk = 0.62f * id + rr() * 0.10f;
            Wq[i][j] = qk + rr() * 0.015f;
            Wk[i][j] = qk + rr() * 0.015f;
            Wv[i][j] = 0.55f * id + rr() * 0.10f;
            Wo[i][j] = 0.35f * id + rr() * 0.08f;
        }

    for (int i = 0; i < D; i++)
        for (int j = 0; j < VMAX; j++) Wu[i][j] = rr() * WU_INIT_SCALE;

    memset(tr, 0, sizeof tr);
    memset(mem, 0, sizeof mem);
    memset(prev_h, 0, sizeof prev_h);
    coherence = 1.0f; q_coherence = 1.0f; focus = 0.0f; need_val = 0.0f;
    LIFE_READY = 0;

    memset(mE, 0, sizeof mE);   memset(vE, 0, sizeof vE);
    memset(mWq, 0, sizeof mWq); memset(vWqs, 0, sizeof vWqs);
    memset(mWk, 0, sizeof mWk); memset(vWks, 0, sizeof vWks);
    memset(mWv, 0, sizeof mWv); memset(vWvs, 0, sizeof vWvs);
    memset(mWo, 0, sizeof mWo); memset(vWos, 0, sizeof vWos);
    memset(mWu, 0, sizeof mWu); memset(vWu, 0, sizeof vWu);
    adam_t = 0;
}

/* ============================================================ */
/*  Helpers                                                      */
/* ============================================================ */

static float dot_d(const float *a, const float *b) {
    float s = 0; for (int i = 0; i < D; i++) s += a[i] * b[i]; return s;
}
static float norm_d(const float *a) {
    return sqrtf(dot_d(a, a) + 1e-9f);
}
static float cos_d(const float *a, const float *b) {
    return dot_d(a, b) / (norm_d(a) * norm_d(b) + 1e-9f);
}
static float mag_Wu(void) {
    float s = 0; int n = 0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < V; j++) { s += fabsf(Wu[i][j]); n++; }
    return s / (n ? n : 1);
}

/* ============================================================ */
/*  Forward                                                      */
/* ============================================================ */

static float forward(int *tok, int T, int compute_loss) {
    if (T <= 0) return 0.0f;
    if (T > CTX) T = CTX;

    T_cache = T;
    for (int i = 0; i < T; i++) tok_cache[i] = tok[i];

    /* Embedding with structural prior.
     * In weightless mode, learned E is excluded. */
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++)
            x_c[i][d] = TRAINED ? (E[tok[i]][d] + basev(tok[i], d)) : basev(tok[i], d);

    /* Q/K/V projections.
     * Transformer parts stay alive in both modes. */
    if (!TRAINED) {
        for (int i = 0; i < T; i++)
            for (int d = 0; d < D; d++)
                structural_qkv(tok[i], d, &Q_c[i][d], &K_c[i][d], &V_c[i][d]);
    } else {
        for (int i = 0; i < T; i++)
            for (int d = 0; d < D; d++) {
                float q = 0, k = 0, v = 0;
                for (int e = 0; e < D; e++) {
                    q += x_c[i][e] * Wq[e][d];
                    k += x_c[i][e] * Wk[e][d];
                    v += x_c[i][e] * Wv[e][d];
                }
                Q_c[i][d] = q; K_c[i][d] = k; V_c[i][d] = v;
            }
    }

    /* Q-style coherence exists from the first forward. */
    float q_bar[D] = {0};
    float k_bar[D] = {0};
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) {
            q_bar[d] += Q_c[i][d] / T;
            k_bar[d] += K_c[i][d] / T;
        }
    q_coherence = cos_d(q_bar, k_bar);

    float shd[CTX] = {0};
    float conc_sum = 0.0f;
    int   conc_cnt = 0;

    for (int q = 0; q < T; q++) {
        for (int k = 0; k <= q; k++) {
            float r = 0;
            for (int e = 0; e < D; e++) r += Q_c[q][e] * K_c[k][e];
            r *= SCALE_INV;
            raw_c[q][k] = r;
            th_c[q][k] = tanhf(r - shd[k]);
        }

        float sc[CTX];
        float mx = -1e9f;
        for (int k = 0; k <= q; k++) {
            sc[k] = raw_c[q][k] + ALPHA * th_c[q][k];
            if (sc[k] > mx) mx = sc[k];
        }
        float z = 0;
        for (int k = 0; k <= q; k++) { sc[k] = expf(sc[k] - mx); z += sc[k]; }
        for (int k = 0; k <= q; k++) att_c[q][k] = sc[k] / z;
        for (int k = q + 1; k < T; k++) att_c[q][k] = 0.0f;

        if (q >= 1) {
            float ent = 0.0f;
            for (int k = 0; k <= q; k++) {
                float p = att_c[q][k];
                if (p > 1e-9f) ent -= p * logf(p);
            }
            float max_ent = logf((float)(q + 1));
            conc_sum += 1.0f - ent / (max_ent + 1e-9f);
            conc_cnt++;
        }

        for (int d = 0; d < D; d++) {
            float h = 0;
            for (int k = 0; k <= q; k++) h += att_c[q][k] * V_c[k][d];
            h_c[q][d] = h;
        }

        for (int k = 0; k <= q; k++)
            shd[k] = DK * shd[k] + (1.0f - DK) * raw_c[q][k];

        for (int k = 0; k <= q; k++)
            tr[tok[k]] += TR_BOOST * att_c[q][k];

        /* Causal trace snapshot: logits at q cannot see future trace updates. */
        for (int v = 0; v < V; v++) trpos_c[q][v] = tr[v];
    }
    focus = conc_cnt > 0 ? conc_sum / conc_cnt : 0.0f;

    /* Output projection + residual.
     * Wo is learned, so it is skipped in honest weightless health check. */
    if (!TRAINED) {
        for (int i = 0; i < T; i++)
            for (int d = 0; d < D; d++) {
                op_c[i][d] = 0.0f;
                r_c[i][d] = x_c[i][d] + h_c[i][d];
            }
    } else {
        for (int i = 0; i < T; i++)
            for (int d = 0; d < D; d++) {
                float o = 0;
                for (int e = 0; e < D; e++) o += h_c[i][e] * Wo[e][d];
                op_c[i][d] = o;
                r_c[i][d] = x_c[i][d] + o;
            }
    }

    /* Mean attention output for life-memory and coherence. */
    float h_bar[D] = {0};
    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) h_bar[d] += h_c[i][d] / T;

    float old_prev[D], old_mem[D];
    memcpy(old_prev, prev_h, sizeof old_prev);
    memcpy(old_mem, mem, sizeof old_mem);

    float hist_coh = LIFE_READY ? cos_d(h_bar, old_prev) : q_coherence;
    coherence = LIFE_READY ? (0.72f * hist_coh + 0.28f * q_coherence) : q_coherence;
    coherence = clampf2(coherence, -1.0f, 1.0f);

    float novelty, drift;
    if (LIFE_READY) {
        novelty = 1.0f - cos_d(x_c[T-1], old_mem);
        drift   = 1.0f - hist_coh;
    } else {
        novelty = 1.0f - q_coherence;
        drift   = 1.0f - q_coherence;
    }

    need_val = clampf2(NEED_BASE + NEED_NOV * novelty + NEED_DRF * drift
                                 + NEED_COH * (1.0f - coherence) * 0.5f, 0, 1);

    for (int d = 0; d < D; d++) {
        prev_h[d] = h_bar[d];
        mem[d] = LIFE_READY ? (MEM_DECAY * mem[d] + (1.0f - MEM_DECAY) * h_bar[d]) : h_bar[d];
    }
    LIFE_READY = 1;

    float mg = mag_Wu();
    float mag_gate = clampf2((mg - MAG_THR) / MAG_RANGE, 0, 1);
    float coh_gate = 0.55f + 0.45f * (coherence + 1.0f) * 0.5f;
    gate_c = mag_gate * coh_gate;
    if (TRAINED && gate_c < GATE_FLOOR * coh_gate) gate_c = GATE_FLOOR * coh_gate;
    if (!TRAINED) gate_c = 0.0f;

    float loss = 0.0f;
    int count = 0;
    for (int i = 0; i < T; i++) {
        for (int v = 0; v < V; v++) {
            float neural = 0;
            if (gate_c > 0) for (int d = 0; d < D; d++) neural += r_c[i][d] * Wu[d][v];
            float bg = logf((bi_count[tok[i]][v] + 0.08f) / (bi_row_sum[tok[i]] + 0.08f * V));
            float tg = (i >= 1) ? logf(1.0f + tri_get(tok[i-1], tok[i], v)) : 0.0f;
            prob_c[i][v] = gate_c * neural + BIGRAM_W * bg + TRIGRAM_W * tg - TR_PENALTY * trpos_c[i][v];
        }

        float mx = prob_c[i][0];
        for (int v = 1; v < V; v++) if (prob_c[i][v] > mx) mx = prob_c[i][v];
        float z = 0;
        for (int v = 0; v < V; v++) { prob_c[i][v] = expf(prob_c[i][v] - mx); z += prob_c[i][v]; }
        for (int v = 0; v < V; v++) prob_c[i][v] /= z;

        if (compute_loss && i < T - 1) {
            float p = prob_c[i][tok[i+1]];
            loss -= logf(p < 1e-9f ? 1e-9f : p);
            count++;
        }
    }

    for (int v = 0; v < V; v++) tr[v] *= TR_DECAY;
    return count ? loss / count : 0.0f;
}

/* ============================================================ */
/*  Backward                                                     */
/* ============================================================ */

static void backward(void) {
    int T = T_cache;
    if (T < 2) return;
    float inv = 1.0f / (T - 1);

    memset(gE, 0, sizeof gE);
    memset(gWq, 0, sizeof gWq); memset(gWk, 0, sizeof gWk);
    memset(gWv, 0, sizeof gWv); memset(gWo, 0, sizeof gWo);
    memset(gWu, 0, sizeof gWu);
    memset(gx, 0, sizeof gx);
    memset(gQ, 0, sizeof gQ); memset(gK, 0, sizeof gK); memset(gV, 0, sizeof gV);
    memset(gh, 0, sizeof gh); memset(gop, 0, sizeof gop); memset(gr, 0, sizeof gr);
    memset(gatt, 0, sizeof gatt); memset(gsc, 0, sizeof gsc);

    static float dlog[CTX][VMAX];
    for (int i = 0; i < T - 1; i++) {
        for (int v = 0; v < V; v++) dlog[i][v] = prob_c[i][v] * inv;
        dlog[i][tok_cache[i+1]] -= inv;
    }

    if (gate_c > 0) {
        for (int i = 0; i < T - 1; i++) {
            for (int d = 0; d < D; d++) {
                float s = 0;
                for (int v = 0; v < V; v++) {
                    s += dlog[i][v] * Wu[d][v];
                    gWu[d][v] += gate_c * r_c[i][d] * dlog[i][v];
                }
                gr[i][d] = gate_c * s;
            }
        }
    }

    for (int i = 0; i < T - 1; i++)
        for (int d = 0; d < D; d++) { gx[i][d] += gr[i][d]; gop[i][d] = gr[i][d]; }

    memset(gh, 0, sizeof gh);
    for (int i = 0; i < T - 1; i++)
        for (int e = 0; e < D; e++) {
            float s = 0;
            for (int d = 0; d < D; d++) {
                gWo[e][d] += h_c[i][e] * gop[i][d];
                s += Wo[e][d] * gop[i][d];
            }
            gh[i][e] = s;
        }

    for (int q = 0; q < T - 1; q++)
        for (int k = 0; k <= q; k++) {
            float da = 0;
            for (int d = 0; d < D; d++) {
                da += gh[q][d] * V_c[k][d];
                gV[k][d] += att_c[q][k] * gh[q][d];
            }
            gatt[q][k] = da;
        }

    for (int q = 0; q < T - 1; q++) {
        float dot = 0;
        for (int k = 0; k <= q; k++) dot += att_c[q][k] * gatt[q][k];
        for (int k = 0; k <= q; k++) gsc[q][k] = att_c[q][k] * (gatt[q][k] - dot);
    }

    static float draw[CTX][CTX];
    for (int q = 0; q < T - 1; q++)
        for (int k = 0; k <= q; k++)
            draw[q][k] = gsc[q][k] * (1.0f + ALPHA * (1.0f - th_c[q][k] * th_c[q][k]));

    for (int q = 0; q < T - 1; q++)
        for (int k = 0; k <= q; k++)
            for (int d = 0; d < D; d++) {
                gQ[q][d] += SCALE_INV * draw[q][k] * K_c[k][d];
                gK[k][d] += SCALE_INV * draw[q][k] * Q_c[q][d];
            }

    for (int i = 0; i < T; i++)
        for (int e = 0; e < D; e++)
            for (int d = 0; d < D; d++) {
                gWq[e][d] += x_c[i][e] * gQ[i][d];
                gWk[e][d] += x_c[i][e] * gK[i][d];
                gWv[e][d] += x_c[i][e] * gV[i][d];
                gx[i][e]  += Wq[e][d] * gQ[i][d]
                          +  Wk[e][d] * gK[i][d]
                          +  Wv[e][d] * gV[i][d];
            }

    for (int i = 0; i < T; i++)
        for (int d = 0; d < D; d++) gE[tok_cache[i]][d] += gx[i][d];
}

/* ============================================================ */
/*  Adam update                                                  */
/* ============================================================ */

static void adam_apply(float *w, float *g, float *m, float *v, int n) {
    float c1 = 1.0f - powf(BETA1, adam_t);
    float c2 = 1.0f - powf(BETA2, adam_t);
    for (int i = 0; i < n; i++) {
        m[i] = BETA1 * m[i] + (1.0f - BETA1) * g[i];
        v[i] = BETA2 * v[i] + (1.0f - BETA2) * g[i] * g[i];
        float mh = m[i] / c1;
        float vh = v[i] / c2;
        w[i] -= LR * mh / (sqrtf(vh) + ADAM_EPS);
    }
}

static void update(void) {
    adam_t++;
    adam_apply((float*)E,  (float*)gE,  (float*)mE,  (float*)vE,  VMAX * D);
    adam_apply((float*)Wq, (float*)gWq, (float*)mWq, (float*)vWqs, D * D);
    adam_apply((float*)Wk, (float*)gWk, (float*)mWk, (float*)vWks, D * D);
    adam_apply((float*)Wv, (float*)gWv, (float*)mWv, (float*)vWvs, D * D);
    adam_apply((float*)Wo, (float*)gWo, (float*)mWo, (float*)vWos, D * D);
    adam_apply((float*)Wu, (float*)gWu, (float*)mWu, (float*)vWu,  D * VMAX);
}

/* ============================================================ */
/*  Training loop                                                */
/* ============================================================ */

static float train_loop(int *data, int dn, int steps, int seq_per_step, int print_every) {
    TRAINED = 1;
    float loss_last = 0; int cl = 0;
    int quint = steps > 5 ? steps / 5 : 1;

    if (dn <= CTX + 1) {
        fprintf(stderr, "corpus too short after BPE: need > %d tokens, got %d\n", CTX + 1, dn);
        return 0.0f;
    }

    memset(tr, 0, sizeof tr);
    memset(mem, 0, sizeof mem);
    memset(prev_h, 0, sizeof prev_h);
    LIFE_READY = 0;
    coherence = 1.0f;
    q_coherence = 1.0f;

    for (int step = 0; step < steps; step++) {
        float batch_loss = 0;
        for (int s = 0; s < seq_per_step; s++) {
            int start = rand() % (dn - CTX - 1);
            int seq[CTX];
            memcpy(seq, data + start, CTX * sizeof(int));
            float loss = forward(seq, CTX, 1);
            backward();
            update();

            int last = seq[CTX - 1];
            int next = data[start + CTX];
            bi_count[last][next]++; bi_row_sum[last]++;
            if (CTX >= 2) tri_add(seq[CTX - 2], last, next);

            batch_loss += loss;
        }
        batch_loss /= seq_per_step;
        if (step >= steps - quint) { loss_last += batch_loss; cl++; }

        if (print_every > 0 && (step % print_every == 0 || step == steps - 1))
            printf("step %4d | loss %.4f | focus %.2f coh %.2f qcoh %.2f need %.2f gate %.2f\n",
                   step, batch_loss, focus, coherence, q_coherence, need_val, gate_c);
    }
    return (cl ? loss_last / cl : 0);
}

/* ============================================================ */
/*  Generation                                                   */
/* ============================================================ */

static int pick_topk(float *prob, int *recent_ctx, int rn) {
    int best = 0; float best_score = -1e30f;
    for (int v = 0; v < V; v++) {
        /* Text-mode safeguard: raw byte tokens for control chars are legal BPE
         * vocabulary entries, but usually poisonous for text generation. */
        if (v < 32 || v == 127) continue;
        float s = logf(prob[v] + 1e-9f) - 0.18f * logf(1.0f + tr[v]);
        for (int j = 0; j < rn; j++) if (recent_ctx[j] == v) { s -= 0.40f; break; }
        if (s > best_score) { best_score = s; best = v; }
    }
    return best;
}

static void generate(const char *prompt, int n_out, FILE *out_f) {
    int buf[MAXTOK];
    int pn = bpe_encode((const unsigned char*)prompt, (int)strlen(prompt), buf);
    if (pn == 0) { buf[0] = ' '; pn = 1; }

    for (int i = 0; i < pn; i++) emit_tok(buf[i], out_f);

    for (int g = 0; g < n_out; g++) {
        int start = pn > CTX ? pn - CTX : 0;
        int len = pn - start;
        int seq[CTX];
        memcpy(seq, buf + start, len * sizeof(int));
        forward(seq, len, 0);
        int recent_start = pn > 8 ? pn - 8 : 0;
        int next = pick_topk(prob_c[len - 1], buf + recent_start, pn - recent_start);
        buf[pn++] = next;
        emit_tok(next, out_f);
        if (pn >= MAXTOK - 1) break;
    }
    fputc('\n', out_f);
}

/* ============================================================ */
/*  Save / load                                                  */
/* ============================================================ */

static int save_model(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite("LUKAS3", 1, 6, f);
    fwrite(&V, sizeof V, 1, f);
    fwrite(L, sizeof L[0], VMAX, f);
    fwrite(R, sizeof R[0], VMAX, f);
    fwrite(E,  sizeof E[0][0],  VMAX * D, f);
    fwrite(Wq, sizeof Wq[0][0], D * D, f);
    fwrite(Wk, sizeof Wk[0][0], D * D, f);
    fwrite(Wv, sizeof Wv[0][0], D * D, f);
    fwrite(Wo, sizeof Wo[0][0], D * D, f);
    fwrite(Wu, sizeof Wu[0][0], D * VMAX, f);
    fwrite(bi_count,   sizeof bi_count[0][0], VMAX * VMAX, f);
    fwrite(bi_row_sum, sizeof bi_row_sum[0],  VMAX, f);
    fwrite(tri_key,    sizeof tri_key[0],     HSZ, f);
    fwrite(tri_cnt,    sizeof tri_cnt[0],     HSZ, f);
    fwrite(tr,         sizeof tr[0],          VMAX, f);
    fclose(f);
    return 1;
}

static void rebuild_tok_str(void) {
    for (int i = 0; i < 256; i++) {
        tok_str[i] = malloc(2);
        if (!tok_str[i]) { fprintf(stderr, "malloc failed\n"); exit(1); }
        tok_str[i][0] = (char)i; tok_str[i][1] = 0;
    }
    for (int id = 256; id < V; id++) {
        int la = (int)strlen(tok_str[L[id]]);
        int lb = (int)strlen(tok_str[R[id]]);
        tok_str[id] = malloc((size_t)la + (size_t)lb + 1u);
        if (!tok_str[id]) { fprintf(stderr, "malloc failed\n"); exit(1); }
        memcpy(tok_str[id], tok_str[L[id]], (size_t)la);
        memcpy(tok_str[id] + la, tok_str[R[id]], (size_t)lb);
        tok_str[id][la + lb] = 0;
    }
}

static int load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char m[7] = {0};
    fread(m, 1, 6, f);
    if (strncmp(m, "LUKAS3", 6) != 0) { fclose(f); return 0; }
    fread(&V, sizeof V, 1, f);
    fread(L, sizeof L[0], VMAX, f);
    fread(R, sizeof R[0], VMAX, f);
    fread(E,  sizeof E[0][0],  VMAX * D, f);
    fread(Wq, sizeof Wq[0][0], D * D, f);
    fread(Wk, sizeof Wk[0][0], D * D, f);
    fread(Wv, sizeof Wv[0][0], D * D, f);
    fread(Wo, sizeof Wo[0][0], D * D, f);
    fread(Wu, sizeof Wu[0][0], D * VMAX, f);
    fread(bi_count,   sizeof bi_count[0][0], VMAX * VMAX, f);
    fread(bi_row_sum, sizeof bi_row_sum[0],  VMAX, f);
    fread(tri_key,    sizeof tri_key[0],     HSZ, f);
    fread(tri_cnt,    sizeof tri_cnt[0],     HSZ, f);
    fread(tr,         sizeof tr[0],          VMAX, f);
    fclose(f);

    rebuild_tok_str();
    TRAINED = 1;
    LIFE_READY = 0;
    coherence = 1.0f;
    q_coherence = 1.0f;
    memset(mem, 0, sizeof mem);
    memset(prev_h, 0, sizeof prev_h);
    return 1;
}

/* ============================================================ */
/*  Default manifest corpus                                      */
/* ============================================================ */

static const char *manifest =
"The shadow remembers what the light forgets. Attention is a form of memory. "
"Every step remembers the one before. Coherence is born from self-awareness. "
"Shadow attention lives on three time scales. Inside one forward pass the "
"shadow rises with each key the attention reads, amplifying what is new and "
"dampening what is already familiar. Across batches the corpus statistics "
"grow online, accumulating bigram and trigram traces of every observed step. "
"Across the model's whole life the per-token trace persists, a slow fatigue "
"and a slow memory together, the architecture's biological time. "
"The shadow is not darkness but depth. The model knows what it needs before "
"it knows what it knows. Weights are the residual on top of structure. "
"Before training the basis already speaks: every token carries its own "
"sinusoidal signature, so attention has signal even with empty weights. "
"Training reshapes that signature, it does not create it. "
"The gate is honest. When weights are weak, statistics rule. When weights "
"grow strong, the neural part rises. When coherence falls, need rises. "
"Attention asks itself: do I rely on what I have learned, or on what the "
"corpus knows, or on what I have already attended to. The answer is a "
"weighted sum, computed every step from observation, not stipulated. "
"The shadow knows what attention needs. Lukas walks through tokens, not "
"characters. Memory is short, coherence is alive. ";

/* ============================================================ */
/*  --test                                                       */
/* ============================================================ */

static int run_test(void) {
    srand(42);
    bpe_init();
    int data[MAXTOK];
    int n = bpe_train((const unsigned char*)manifest, (int)strlen(manifest), data, 96);
    printf("BPE vocab=%d corpus_tokens=%d\n", V, n);

    int test_ids[MAXTOK];
    int tn = bpe_encode((const unsigned char*)manifest, (int)strlen(manifest), test_ids);
    char buf[8192]; int p = 0;
    for (int i = 0; i < tn; i++) p = emit_tok_buf(test_ids[i], buf, p);
    buf[p] = 0;
    int rt_ok = (strcmp(buf, manifest) == 0);
    printf("BPE roundtrip: %s\n", rt_ok ? "ok" : "BAD");

    model_init();
    stats_build(data, n);

    TRAINED = 0;
    LIFE_READY = 0;
    printf("\n[weightless mode — structural Q/K/V + shadow + n-gram, no learned weights]\n  ");
    generate("The shadow ", 60, stdout);
    printf("  shadow state: focus=%.2f coh=%.2f qcoh=%.2f need=%.2f gate=%.2f\n",
           focus, coherence, q_coherence, need_val, gate_c);

    printf("\n[training]\n");
    int steps = 600;
    float final_loss = train_loop(data, n, steps, 1, 100);

    printf("\n[weighted mode — trained, all three scales active]\n  ");
    generate("The shadow ", 60, stdout);
    printf("  ");
    generate("attention ", 60, stdout);

    int total_params = VMAX * D + 4 * D * D + D * VMAX;
    printf("\nfinal loss %.4f | params %d | vocab %d\n", final_loss, total_params, V);
    printf("shadow state: focus=%.2f coh=%.2f qcoh=%.2f need=%.2f gate=%.2f mag(Wu)=%.4f\n",
           focus, coherence, q_coherence, need_val, gate_c, mag_Wu());

    int n_active = 0; float tr_mass = 0;
    for (int v = 0; v < V; v++) { if (tr[v] > 0.01f) n_active++; tr_mass += tr[v]; }
    printf("per-life trace: active=%d / %d, mass=%.3f\n", n_active, V, tr_mass);

    return rt_ok ? 0 : 1;
}

/* ============================================================ */
/*  main                                                         */
/* ============================================================ */

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--test") == 0) return run_test();

    if (strcmp(argv[1], "train") == 0 && argc >= 4) {
        srand(42);
        bpe_init();
        FILE *f = fopen(argv[2], "rb");
        if (!f) { perror(argv[2]); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fprintf(stderr, "empty corpus\n"); fclose(f); return 1; }
        char *txt = malloc((size_t)sz + 1u);
        if (!txt) { fprintf(stderr, "malloc failed\n"); fclose(f); return 1; }
        sz = (long)fread(txt, 1, (size_t)sz, f); txt[sz] = 0; fclose(f);

        int data[MAXTOK];
        int n = bpe_train((const unsigned char*)txt, (int)sz, data, 96);
        printf("BPE vocab=%d corpus_tokens=%d\n", V, n);
        model_init();
        stats_build(data, n);

        int epochs = argc > 4 ? atoi(argv[4]) : 1;
        int steps_per_epoch = 800;
        for (int e = 0; e < epochs; e++) {
            printf("\n[epoch %d]\n", e + 1);
            train_loop(data, n, steps_per_epoch, 1, 100);
        }
        if (save_model(argv[3])) printf("\nsaved: %s\n", argv[3]);
        else fprintf(stderr, "save failed: %s\n", argv[3]);
        free(txt);
        return 0;
    }

    if (strcmp(argv[1], "infer") == 0 && argc >= 4) {
        bpe_init();
        if (!load_model(argv[2])) { fprintf(stderr, "load failed\n"); return 1; }
        printf("loaded: vocab=%d\n", V);
        int n = argc > 4 ? atoi(argv[4]) : 80;
        generate(argv[3], n, stdout);
        return 0;
    }

    if (strcmp(argv[1], "--noweights") == 0 && argc >= 4) {
        srand(42);
        bpe_init();
        FILE *f = fopen(argv[2], "rb");
        if (!f) { perror(argv[2]); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fprintf(stderr, "empty corpus\n"); fclose(f); return 1; }
        char *txt = malloc((size_t)sz + 1u);
        if (!txt) { fprintf(stderr, "malloc failed\n"); fclose(f); return 1; }
        sz = (long)fread(txt, 1, (size_t)sz, f); txt[sz] = 0; fclose(f);

        int data[MAXTOK];
        int n = bpe_train((const unsigned char*)txt, (int)sz, data, 96);
        printf("BPE vocab=%d corpus_tokens=%d\n", V, n);
        model_init();
        stats_build(data, n);
        TRAINED = 0;
        LIFE_READY = 0;
        generate(argv[3], 80, stdout);
        printf("shadow state: focus=%.2f coh=%.2f qcoh=%.2f need=%.2f gate=%.2f\n",
               focus, coherence, q_coherence, need_val, gate_c);
        free(txt);
        return 0;
    }

    fprintf(stderr,
            "usage: %s --test\n"
            "       %s train  corpus.txt model.lukas [epochs]\n"
            "       %s infer  model.lukas 'prompt' [tokens]\n"
            "       %s --noweights corpus.txt 'prompt'\n",
            argv[0], argv[0], argv[0], argv[0]);
    return 1;
}
