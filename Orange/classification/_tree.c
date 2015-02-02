#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _MSC_VER
	#include "err.h"
	#define ASSERT(x) if (!(x)) err(1, "%s:%d", __FILE__, __LINE__)
#else
	#define ASSERT(x) if(!(x)) exit(1)
	#define log2f(x) log((double) (x)) / log(2.0)
#endif // _MSC_VER

struct SimpleTreeNode {
	int type, children_size, split_attr;
	float split;
	struct SimpleTreeNode **children;

	float *dist;  /* classification */
	float n, sum; /* regression */
};

struct Args {
	int minInstances, maxDepth;
	float maxMajority, skipProb;

	int type, *attr_split_so_far;
	// PDomain domain;
	// PRandomGenerator randomGenerator;
};

enum { DiscreteNode, ContinuousNode, PredictorNode };
enum { Classification, Regression };

int compar_attr;

/* This function uses the global variable compar_attr.
 * Examples with unknowns are larger so that, when sorted, they appear at the bottom.
 */
int 
compar_examples(const void *ptr1, const void *ptr2)
{
	float *e1, *e2;

	e1 = (float *)ptr1;
	e2 = (float *)ptr2;
	if (isnan(e1[compar_attr]))
		return 1;
	if (isnan(e2[compar_attr]))
		return -1;
	
	if (e1[compar_attr] < e2[compar_attr])
		return -1;
	if (e1[compar_attr] > e2[compar_attr])
		return 1;
	return 0;
}

float
entropy(float *xs, int size)
{
	float *ip, *end, sum, e;

	for (ip = xs, end = xs + size, e = 0.0, sum = 0.0; ip != end; ip++)
		if (*ip > 0.0) {
			e -= *ip * log2f(*ip);
			sum += *ip;
		}

	return sum == 0.0 ? 0.0 : e / sum + log2f(sum);
}

int
test_min_examples(float *attr_dist, int attr_vals, struct Args *args)
{
	int i;

	for (i = 0; i < attr_vals; i++) {
		if (attr_dist[i] > 0.0 && attr_dist[i] < args->minInstances)
			return 0;
	}
	return 1;
}

float
gain_ratio_c(float *examples, float *ys, float *ws, int size, int M, int cls_vals, int attr, float cls_entropy, struct Args *args, float *best_split)
{
	float *ex, *ex_end, *ex_next, *y, *w;
	int i, cls, minInstances, size_known;
	float score, *dist_lt, *dist_ge, *attr_dist, best_score, size_weight;

	/* minInstances should be at least 1, otherwise there is no point in splitting */
	minInstances = args->minInstances < 1 ? 1 : args->minInstances;

	/* allocate space */
	ASSERT(dist_lt = (float *)calloc(cls_vals, sizeof *dist_lt));
	ASSERT(dist_ge = (float *)calloc(cls_vals, sizeof *dist_ge));
	ASSERT(attr_dist = (float *)calloc(2, sizeof *attr_dist));

	/* sort */
	compar_attr = attr;
	qsort(examples, size, M * sizeof(float), compar_examples);

	/* compute gain ratio for every split */
	size_known = size;
	size_weight = 0.0;
	for (ex = examples, ex_end = examples + size * M, y = ys, w = ws; ex < ex_end; ex += M, y++, w++) {
		if (isnan(ex[attr])) {
			size_known = (ex - examples) / M;
			break;
		}
		if (!isnan(*y))
			dist_ge[(int)*y] += *w;
		size_weight += *w;
	}

	attr_dist[1] = size_weight;
	best_score = -INFINITY;

	for (ex = examples, ex_end = ex + (size_known - minInstances) * M, ex_next = ex + M, i = 0; ex < ex_end; ex += M, ex_next++, i++) {
		if (!isnan(ys[i])) {
			cls = (int)ys[i];
			dist_lt[cls] += ws[i];
			dist_ge[cls] -= ws[i];
		}
		attr_dist[0] += ws[i];
		attr_dist[1] -= ws[i];

		if (ex[attr] == ex_next[attr] || i + 1 < minInstances)
			continue;

		/* gain ratio */
		score = (attr_dist[0] * entropy(dist_lt, cls_vals) + attr_dist[1] * entropy(dist_ge, cls_vals)) / size_weight;
		score = (cls_entropy - score) / entropy(attr_dist, 2);


		if (score > best_score) {
			best_score = score;
			*best_split = (ex[attr] + ex_next[attr]) / 2.0;
		}
	}

	printf("C %d %f\n", attr, best_score);

	/* cleanup */
	free(dist_lt);
	free(dist_ge);
	free(attr_dist);

	return best_score;
}
