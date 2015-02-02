#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
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

	for (ex = examples, ex_end = ex + (size_known - minInstances) * M, ex_next = ex + M, i = 0; ex < ex_end; ex += M, ex_next += M, i++) {
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

		printf("score: %f\n", score);

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

float
gain_ratio_d(float *examples, float *ys, float *ws, int size, int M, int cls_vals, int *attr_vals, int attr, float cls_entropy, struct Args *args)
{
	float *ex, *ex_end;
	int i, cls_val, attr_val;
	float score, size_weight, size_attr_known, size_attr_cls_known, attr_entropy, *cont, *attr_dist, *attr_dist_cls_known;

	/* allocate space */
	ASSERT(cont = (float *)calloc(cls_vals * attr_vals[attr], sizeof(float)));
	ASSERT(attr_dist = (float *)calloc(attr_vals[attr], sizeof(float)));
	ASSERT(attr_dist_cls_known = (float *)calloc(attr_vals[attr], sizeof(float)));

	/* contingency matrix */
	size_weight = 0.0;
	for (ex = examples, ex_end = examples + size * M, i = 0; ex < ex_end; ex += M, i++) {
		if (!isnan(ex[attr])) {
			attr_val = ex[attr];
			attr_dist[attr_val] += ws[i];
			if (!isnan(ys[i])) {
				cls_val = ys[i];
				attr_dist_cls_known[attr_val] += ws[i];
				cont[attr_val * cls_vals + cls_val] += ws[i];
			}
		}
		size_weight += ws[i];
	}

	/* min examples in leaves */
	if (!test_min_examples(attr_dist, attr_vals[attr], args)) {
		score = -INFINITY;
		goto finish;
	}

	size_attr_known = size_attr_cls_known = 0.0;
	for (i = 0; i < attr_vals[attr]; i++) {
		size_attr_known += attr_dist[i];
		size_attr_cls_known += attr_dist_cls_known[i];
	}

	/* gain ratio */
	score = 0.0;
	for (i = 0; i < attr_vals[attr]; i++)
		score += attr_dist_cls_known[i] * entropy(cont + i * cls_vals, cls_vals);
	attr_entropy = entropy(attr_dist, attr_vals[attr]);

	if (size_attr_cls_known == 0.0 || attr_entropy == 0.0 || size_weight == 0.0) {
		score = -INFINITY;
		goto finish;
	}

	score = (cls_entropy - score / size_attr_cls_known) / attr_entropy * ((float)size_attr_known / size_weight);

	printf("D %d %f\n", attr, score);

finish:
	free(cont);
	free(attr_dist);
	free(attr_dist_cls_known);
	return score;
}

float
mse_c(float *examples, float *ys, float *ws, int size, int M, int attr, float cls_mse, struct Args *args, float *best_split)
{
	float *ex, *ex_end, *ex_next;
	int i, minInstances, size_known;
	float size_attr_known, size_weight, cls_val, cls_score, best_score, size_attr_cls_known, score;

	struct Variance {
		double n, sum, sum2;
	} var_lt = {0.0, 0.0, 0.0}, var_ge = {0.0, 0.0, 0.0};

	/* minInstances should be at least 1, otherwise there is no point in splitting */
	minInstances = args->minInstances < 1 ? 1 : args->minInstances;

	/* sort */
	compar_attr = attr;
	qsort(examples, size, M * sizeof(float), compar_examples);

	/* compute mse for every split */
	size_known = size;
	size_attr_known = 0.0;
	for (ex = examples, ex_end = examples + size * M, i = 0; ex < ex_end; ex += M, i++) {
		if (isnan(ex[attr])) {
			size_known = (ex - examples) / M;
			break;
		}
		if (!isnan(ys[i])) {
			cls_val = ys[i];
			var_ge.n += ws[i];
			var_ge.sum += ws[i] * cls_val;
			var_ge.sum2 += ws[i] * cls_val * cls_val;
		}
		size_attr_known += ws[i];
	}

	/* count the remaining examples with unknown values */
	size_weight = size_attr_known;
	for (ex_end = examples + size * M; ex < ex_end; ex += M)
		size_weight += ws[i];

	size_attr_cls_known = var_ge.n;
	best_score = -INFINITY;

	for (ex = examples, ex_end = ex + (size_known - minInstances) * M, ex_next = ex + M, i = 0; ex < ex_end; ex += M, ex_next += M, i++) {
		if (!isnan(ys[i])) {
			cls_val = ys[i];
			var_lt.n += ws[i];
			var_lt.sum += ws[i] * cls_val;
			var_lt.sum2 += ws[i] * cls_val * cls_val;

			/* this calculation might be numarically unstable - fix */
			var_ge.n -= ws[i];
			var_ge.sum -= ws[i] * cls_val;
			var_ge.sum2 -= ws[i] * cls_val * cls_val;
		}

		if (ex[attr] == ex_next[attr] || i + 1 < minInstances)
			continue;

		/* compute mse */
		score = var_lt.sum2 - var_lt.sum * var_lt.sum / var_lt.n;
		score += var_ge.sum2 - var_ge.sum * var_ge.sum / var_ge.n;

		score = (cls_mse - score / size_attr_cls_known) / cls_mse * (size_attr_known / size_weight);

		if (score > best_score) {
			best_score = score;
			*best_split = (ex[attr] + ex_next[attr]) / 2.0;
		}
	}

	printf("C %d %f\n", attr, best_score);
	return best_score;
}


struct SimpleTreeNode *
make_predictor(struct SimpleTreeNode *node)
{
	node->type = PredictorNode;
	node->children_size = 0;
	return node;
}


struct SimpleTreeNode *
build_tree(float *examples, float *ys, float *ws, int size, int M, int cls_vals, int *domain, int depth, struct SimpleTreeNode *parent, struct Args *args)
{
	int i, best_attr;
	float cls_entropy, cls_mse, best_score, score, size_weight, best_split, split;
	struct SimpleTreeNode *node;
	float *ex, *ex_end;

	ASSERT(node = (struct SimpleTreeNode *)malloc(sizeof *node));

	if (args->type == Classification) {
		ASSERT(node->dist = (float *)calloc(cls_vals, sizeof(float *)));

		if (size == 0) {
			assert(parent);
			node->type = PredictorNode;
			node->children_size = 0;
			memcpy(node->dist, parent->dist, cls_vals * sizeof *node->dist);
			return node;
		}

		/* class distribution */
		size_weight = 0.0;
		for (i = 0; i < size; i++)
			if (!isnan(ys[i])) {
				node->dist[(int)ys[i]] += ws[i];
				size_weight += ws[i];
			}

		/* stopping criterion: majority class */
		for (i = 0; i < cls_vals; i++)
			if (node->dist[i] / size_weight >= args->maxMajority)
				return make_predictor(node);

		cls_entropy = entropy(node->dist, cls_vals);

	} else {
		float n, sum, sum2, cls_val;

		assert(args->type == Regression);
		if (size == 0) {
			assert(parent);
			node->type = PredictorNode;
			node->children_size = 0;
			node->n = parent->n;
			node->sum = parent->sum;
			return node;
		}

		n = sum = sum2 = 0.0;
		for (i = 0; i < size; i++) {
			if (!isnan(ys[i])) {
				cls_val = ys[i];
				n += ws[i];
				sum += ws[i] * cls_val;
				sum2 += ws[i] * cls_val * cls_val;
			}
		}

		node->n = n;
		node->sum = sum;
		cls_mse = (sum2 - sum * sum / n) / n;

		if (cls_mse < 1e-5) {
			return make_predictor(node);
		}
	}

	/* stopping criterion: depth exceeds limit */
	if (depth == args->maxDepth)
		return make_predictor(node);

	/* score attributes */
	best_score = -INFINITY;
	for (i = 0; i < M; i++) {
		if (!args->attr_split_so_far[i]) {
			/* select random subset of attributes */
			if ((double)rand() / (double)RAND_MAX < args->skipProb)
				continue;

			if (domain[i] == 0) { // INTVAR
				/*
				score = args->type == Classification ?
				  gain_ratio_d(examples, size, i, cls_entropy, args) :
				  mse_d(examples, size, i, cls_mse, args);
				if (score > best_score) {
					best_score = score;
					best_attr = i;
				}
				*/
				assert(0);
			} else if (domain[i] == 1) { // FLOATVAR
				/*
				score = args->type == Classification ?
				  gain_ratio_c(examples, size, i, cls_entropy, args, &split) :
				  mse_c(examples, size, i, cls_mse, args, &split);
				*/
				gain_ratio_c(examples, ys, ws, size, M, cls_vals, i, cls_entropy, args, &split);
				if (score > best_score) {
					best_score = score;
					best_split = split;
					best_attr = i;
				}
			}
		}
	}

#if 0

	if (best_score == -INFINITY)
		return make_predictor(node, examples, size, args);

	if (args->domain->attributes->at(best_attr)->varType == TValue::INTVAR) {
		struct Example *child_examples, *child_ex;
		int attr_vals;
		float size_known, *attr_dist;

		/* printf("* %2d %3s %3d %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_score); */

		attr_vals = args->domain->attributes->at(best_attr)->noOfValues(); 

		node->type = DiscreteNode;
		node->split_attr = best_attr;
		node->children_size = attr_vals;

		ASSERT(child_examples = (struct Example *)calloc(size, sizeof *child_examples));
		ASSERT(node->children = (SimpleTreeNode **)calloc(attr_vals, sizeof *node->children));
		ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

		/* attribute distribution */
		size_known = 0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!ex->example->values[best_attr].isSpecial()) {
				attr_dist[ex->example->values[best_attr].intV] += ex->weight;
				size_known += ex->weight;
			}

		args->attr_split_so_far[best_attr] = 1;

		for (i = 0; i < attr_vals; i++) {
			/* create a new example table */
			for (ex = examples, ex_end = examples + size, child_ex = child_examples; ex < ex_end; ex++) {
				if (ex->example->values[best_attr].isSpecial()) {
					*child_ex = *ex;
					child_ex->weight *= attr_dist[i] / size_known;
					child_ex++;
				} else if (ex->example->values[best_attr].intV == i) {
					*child_ex++ = *ex;
				}
			}

			node->children[i] = build_tree(child_examples, child_ex - child_examples, depth + 1, node, args);
		}
					
		args->attr_split_so_far[best_attr] = 0;

		free(attr_dist);
		free(child_examples);
	} else {
		struct Example *examples_lt, *examples_ge, *ex_lt, *ex_ge;
		float size_lt, size_ge;

		/* printf("* %2d %3s %3d %f %f\n", depth, args->domain->attributes->at(best_attr)->get_name().c_str(), size, best_split, best_score); */

		assert(args->domain->attributes->at(best_attr)->varType == TValue::FLOATVAR);

		ASSERT(examples_lt = (struct Example *)calloc(size, sizeof *examples));
		ASSERT(examples_ge = (struct Example *)calloc(size, sizeof *examples));

		size_lt = size_ge = 0.0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!ex->example->values[best_attr].isSpecial())
				if (ex->example->values[best_attr].floatV < best_split)
					size_lt += ex->weight;
				else
					size_ge += ex->weight;

		for (ex = examples, ex_end = examples + size, ex_lt = examples_lt, ex_ge = examples_ge; ex < ex_end; ex++)
			if (ex->example->values[best_attr].isSpecial()) {
				*ex_lt = *ex;
				*ex_ge = *ex;
				ex_lt->weight *= size_lt / (size_lt + size_ge);
				ex_ge->weight *= size_ge / (size_lt + size_ge);
				ex_lt++;
				ex_ge++;
			} else if (ex->example->values[best_attr].floatV < best_split) {
				*ex_lt++ = *ex;
			} else {
				*ex_ge++ = *ex;
			}

		/*
		 * Check there was an actual reduction of size in the the two subsets.
		 * This test fails when all best_attr's (the only attr) values  are
		 * the same (and equal best_split) so the data is split in 0 | n size
		 * subsets and recursing would lead to an infinite recursion.
		 */
		if ((ex_lt - examples_lt) < size && (ex_ge - examples_ge) < size) {
			node->type = ContinuousNode;
			node->split_attr = best_attr;
			node->split = best_split;
			node->children_size = 2;
			ASSERT(node->children = (SimpleTreeNode **)calloc(2, sizeof *node->children));

			node->children[0] = build_tree(examples_lt, ex_lt - examples_lt, depth + 1, node, args);
			node->children[1] = build_tree(examples_ge, ex_ge - examples_ge, depth + 1, node, args);
		} else {
			node = make_predictor(node, examples, size, args);
		}

		free(examples_lt);
		free(examples_ge);
	}

#endif
	return node;
}
