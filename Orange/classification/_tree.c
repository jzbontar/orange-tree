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
	#define log2(x) log((double) (x)) / log(2.0)
#endif // _MSC_VER

struct Args {
	int minInstances, maxDepth;
	float maxMajority, skipProb;

	int type, *attr_split_so_far, num_attrs, cls_vals, *attr_vals, *domain;
};

struct SimpleTreeNode {
	int type, children_size, split_attr;
	float split;
	struct SimpleTreeNode **children;

	float *dist;  /* classification */
	float n, sum; /* regression */
};

struct Example {
	double *x, y, weight;
};

enum { DiscreteNode, ContinuousNode, PredictorNode };
enum { Classification, Regression };
enum { IntVar, FloatVar };

int compar_attr;

/* This function uses the global variable compar_attr.
 * Examples with unknowns are larger so that, when sorted, they appear at the bottom.
 */
int
compar_examples(const void *ptr1, const void *ptr2)
{
	double x1, x2;

	x1 = ((struct Example *)ptr1)->x[compar_attr];
	x2 = ((struct Example *)ptr2)->x[compar_attr];
	if (isnan(x1))
		return 1;
	if (isnan(x2))
		return -1;
	if (x1 < x2)
		return -1;
	if (x1 > x2)
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
gain_ratio_c(struct Example *examples, int size, int attr, float cls_entropy, struct Args *args, float *best_split)
{
	struct Example *ex, *ex_end, *ex_next;
	int i, cls, cls_vals, minInstances, size_known;
	float score, *dist_lt, *dist_ge, *attr_dist, best_score, size_weight;

	cls_vals = args->cls_vals;

	/* minInstances should be at least 1, otherwise there is no point in splitting */
	minInstances = args->minInstances < 1 ? 1 : args->minInstances;

	/* allocate space */
	ASSERT(dist_lt = (float *)calloc(cls_vals, sizeof *dist_lt));
	ASSERT(dist_ge = (float *)calloc(cls_vals, sizeof *dist_ge));
	ASSERT(attr_dist = (float *)calloc(2, sizeof *attr_dist));

	/* sort */
	compar_attr = attr;
	qsort(examples, size, sizeof(struct Example), compar_examples);

	/* compute gain ratio for every split */
	size_known = size;
	size_weight = 0.0;
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (isnan(ex->x[attr])) {
			size_known = ex - examples;
			break;
		}
		if (!isnan(ex->y))
			dist_ge[(int)ex->y] += ex->weight;
		size_weight += ex->weight;
	}

	attr_dist[1] = size_weight;
	best_score = -INFINITY;

	for (ex = examples, ex_end = ex + size_known - minInstances, ex_next = ex + 1, i = 0; ex < ex_end; ex++, ex_next++, i++) {
		if (!isnan(ex->y)) {
			cls = ex->y;
			dist_lt[cls] += ex->weight;
			dist_ge[cls] -= ex->weight;
		}
		attr_dist[0] += ex->weight;
		attr_dist[1] -= ex->weight;

		if (ex->x[attr] == ex_next->x[attr] || i + 1 < minInstances)
			continue;

		/* gain ratio */
		score = (attr_dist[0] * entropy(dist_lt, cls_vals) + attr_dist[1] * entropy(dist_ge, cls_vals)) / size_weight;
		score = (cls_entropy - score) / entropy(attr_dist, 2);


		if (score > best_score) {
			best_score = score;
			*best_split = (ex->x[attr] + ex_next->x[attr]) / 2.0;
		}
	}

	/* printf("C %s %f\n", args->domain->attributes->at(attr)->get_name().c_str(), best_score); */

	/* cleanup */
	free(dist_lt);
	free(dist_ge);
	free(attr_dist);

	return best_score;
}

float
gain_ratio_d(struct Example *examples, int size, int attr, float cls_entropy, struct Args *args)
{
	struct Example *ex, *ex_end;
	int i, cls_vals, attr_vals, attr_val, cls_val;
	float score, size_weight, size_attr_known, size_attr_cls_known, attr_entropy, *cont, *attr_dist, *attr_dist_cls_known;

	cls_vals = args->cls_vals;
	attr_vals = args->attr_vals[attr];

	/* allocate space */
	ASSERT(cont = (float *)calloc(cls_vals * attr_vals, sizeof(float *)));
	ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof(float *)));
	ASSERT(attr_dist_cls_known = (float *)calloc(attr_vals, sizeof(float *)));

	/* contingency matrix */
	size_weight = 0.0;
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!isnan(ex->x[attr])) {
			attr_val = ex->x[attr];
			attr_dist[attr_val] += ex->weight;
			if (!isnan(ex->y)) {
				cls_val = ex->y;
				attr_dist_cls_known[attr_val] += ex->weight;
				cont[attr_val * cls_vals + cls_val] += ex->weight;
			}
		}
		size_weight += ex->weight;
	}

	/* min examples in leaves */
	if (!test_min_examples(attr_dist, attr_vals, args)) {
		score = -INFINITY;
		goto finish;
	}

	size_attr_known = size_attr_cls_known = 0.0;
	for (i = 0; i < attr_vals; i++) {
		size_attr_known += attr_dist[i];
		size_attr_cls_known += attr_dist_cls_known[i];
	}

	/* gain ratio */
	score = 0.0;
	for (i = 0; i < attr_vals; i++)
		score += attr_dist_cls_known[i] * entropy(cont + i * cls_vals, cls_vals);
	attr_entropy = entropy(attr_dist, attr_vals);

	if (size_attr_cls_known == 0.0 || attr_entropy == 0.0 || size_weight == 0.0) {
		score = -INFINITY;
		goto finish;
	}

	score = (cls_entropy - score / size_attr_cls_known) / attr_entropy * ((float)size_attr_known / size_weight);

	/* printf("D %d %f\n", attr, score); */

finish:
	free(cont);
	free(attr_dist);
	free(attr_dist_cls_known);
	return score;
}

float
mse_c(struct Example *examples, int size, int attr, float cls_mse, struct Args *args, float *best_split)
{
	struct Example *ex, *ex_end, *ex_next;
	int i, minInstances, size_known;
	float size_attr_known, size_weight, cls_val, best_score, size_attr_cls_known, score;

	struct Variance {
		double n, sum, sum2;
	} var_lt = {0.0, 0.0, 0.0}, var_ge = {0.0, 0.0, 0.0};

	/* minInstances should be at least 1, otherwise there is no point in splitting */
	minInstances = args->minInstances < 1 ? 1 : args->minInstances;

	/* sort */
	compar_attr = attr;
	qsort(examples, size, sizeof(struct Example), compar_examples);

	/* compute mse for every split */
	size_known = size;
	size_attr_known = 0.0;
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (isnan(ex->x[attr])) {
			size_known = ex - examples;
			break;
		}
		if (!isnan(ex->y)) {
			cls_val = ex->y;
			var_ge.n += ex->weight;
			var_ge.sum += ex->weight * cls_val;
			var_ge.sum2 += ex->weight * cls_val * cls_val;
		}
		size_attr_known += ex->weight;
	}

	/* count the remaining examples with unknown values */
	size_weight = size_attr_known;
	for (ex_end = examples + size; ex < ex_end; ex++)
		size_weight += ex->weight;

	size_attr_cls_known = var_ge.n;
	best_score = -INFINITY;

	for (ex = examples, ex_end = ex + size_known - minInstances, ex_next = ex + 1, i = 0; ex < ex_end; ex++, ex_next++, i++) {
		if (!isnan(ex->y)) {
			cls_val = ex->y;
			var_lt.n += ex->weight;
			var_lt.sum += ex->weight * cls_val;
			var_lt.sum2 += ex->weight * cls_val * cls_val;

			/* this calculation might be numarically unstable - fix */
			var_ge.n -= ex->weight;
			var_ge.sum -= ex->weight * cls_val;
			var_ge.sum2 -= ex->weight * cls_val * cls_val;
		}

		if (ex->x[attr] == ex_next->x[attr] || i + 1 < minInstances)
			continue;

		/* compute mse */
		score = var_lt.sum2 - var_lt.sum * var_lt.sum / var_lt.n;
		score += var_ge.sum2 - var_ge.sum * var_ge.sum / var_ge.n;

		score = (cls_mse - score / size_attr_cls_known) / cls_mse * (size_attr_known / size_weight);

		if (score > best_score) {
			best_score = score;
			*best_split = (ex->x[attr] + ex_next->x[attr]) / 2.0;
		}
	}

	/* printf("C %s %f\n", args->domain->attributes->at(attr)->get_name().c_str(), best_score); */
	return best_score;
}

float
mse_d(struct Example *examples, int size, int attr, float cls_mse, struct Args *args)
{
	int attr_vals;
	float *attr_dist, score, cls_val, size_attr_cls_known, size_attr_known, size_weight;
	struct Example *ex, *ex_end;

	struct Variance {
		float n, sum, sum2;
	} *variances, *v, *v_end;

	attr_vals = args->attr_vals[attr];

	ASSERT(variances = (struct Variance *)calloc(attr_vals, sizeof *variances));
	ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

	size_weight = size_attr_cls_known = size_attr_known = 0.0;
	for (ex = examples, ex_end = examples + size; ex < ex_end; ex++) {
		if (!isnan(ex->x[attr])) {
			attr_dist[(int)ex->x[attr]] += ex->weight;
			size_attr_known += ex->weight;

			if (!isnan(ex->y)) {
				cls_val = ex->y;
				v = variances + (int)ex->x[attr];
				v->n += ex->weight;
				v->sum += ex->weight * cls_val;
				v->sum2 += ex->weight * cls_val * cls_val;
				size_attr_cls_known += ex->weight;
			}
		}
		size_weight += ex->weight;
	}

	/* minimum examples in leaves */
	if (!test_min_examples(attr_dist, attr_vals, args)) {
		score = -INFINITY;
		goto finish;
	}

	score = 0.0;
	for (v = variances, v_end = variances + attr_vals; v < v_end; v++)
		if (v->n > 0.0)
			score += v->sum2 - v->sum * v->sum / v->n;
	score = (cls_mse - score / size_attr_cls_known) / cls_mse * (size_attr_known / size_weight);

	if (size_attr_cls_known <= 0.0 || cls_mse <= 0.0 || size_weight <= 0.0)
		score = 0.0;

finish:
	free(attr_dist);
	free(variances);

	return score;
}

struct SimpleTreeNode *
make_predictor(struct SimpleTreeNode *node, struct Example *examples, int size, struct Args *args)
{
	node->type = PredictorNode;
	node->children_size = 0;
	return node;
}

struct SimpleTreeNode *
build_tree_(struct Example *examples, int size, int depth, struct SimpleTreeNode *parent, struct Args *args)
{
	int i, cls_vals, best_attr;
	float cls_entropy, cls_mse, best_score, score, size_weight, best_split, split;
	struct SimpleTreeNode *node;
	struct Example *ex, *ex_end;

	cls_vals = args->cls_vals;

	ASSERT(node = (struct SimpleTreeNode *)malloc(sizeof *node));

	if (args->type == Classification) {
		ASSERT(node->dist = (float *)calloc(cls_vals, sizeof(float)));

		if (size == 0) {
			assert(parent);
			node->type = PredictorNode;
			node->children_size = 0;
			memcpy(node->dist, parent->dist, cls_vals * sizeof *node->dist);
			return node;
		}

		/* class distribution */
		size_weight = 0.0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!isnan(ex->y)) {
				node->dist[(int)ex->y] += ex->weight;
				size_weight += ex->weight;
			}

		/* stopping criterion: majority class */
		for (i = 0; i < cls_vals; i++)
			if (node->dist[i] / size_weight >= args->maxMajority)
				return make_predictor(node, examples, size, args);

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
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!isnan(ex->y)) {
				cls_val = ex->y;
				n += ex->weight;
				sum += ex->weight * cls_val;
				sum2 += ex->weight * cls_val * cls_val;
			}

		node->n = n;
		node->sum = sum;
		cls_mse = (sum2 - sum * sum / n) / n;

		if (cls_mse < 1e-5) {
			return make_predictor(node, examples, size, args);
		}
	}

	/* stopping criterion: depth exceeds limit */
	if (depth == args->maxDepth)
		return make_predictor(node, examples, size, args);

	/* score attributes */
	best_score = -INFINITY;

	for (i = 0; i < args->num_attrs; i++) {
		if (!args->attr_split_so_far[i]) {
			/* select random subset of attributes */
			if ((double)rand() / (double)RAND_MAX < args->skipProb)
				continue;

			if (args->domain[i] == IntVar) {
				score = args->type == Classification ?
				  gain_ratio_d(examples, size, i, cls_entropy, args) :
				  mse_d(examples, size, i, cls_mse, args);
				if (score > best_score) {
					best_score = score;
					best_attr = i;
				}
			} else if (args->domain[i] == FloatVar) {
				score = args->type == Classification ?
				  gain_ratio_c(examples, size, i, cls_entropy, args, &split) :
				  mse_c(examples, size, i, cls_mse, args, &split);
				if (score > best_score) {
					best_score = score;
					best_split = split;
					best_attr = i;
				}
			}
		}
	}

	if (best_score == -INFINITY)
		return make_predictor(node, examples, size, args);

	if (args->domain[best_attr] == IntVar) {
		struct Example *child_examples, *child_ex;
		int attr_vals;
		float size_known, *attr_dist;

		printf("* %2d %3d %3d %f\n", depth, best_attr, size, best_score);

		attr_vals = args->attr_vals[best_attr];

		node->type = DiscreteNode;
		node->split_attr = best_attr;
		node->children_size = attr_vals;

		ASSERT(child_examples = (struct Example *)calloc(size, sizeof *child_examples));
		ASSERT(node->children = (struct SimpleTreeNode **)calloc(attr_vals, sizeof *node->children));
		ASSERT(attr_dist = (float *)calloc(attr_vals, sizeof *attr_dist));

		/* attribute distribution */
		size_known = 0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!isnan(ex->x[best_attr])) {
				attr_dist[(int)ex->x[best_attr]] += ex->weight;
				size_known += ex->weight;
			}

		args->attr_split_so_far[best_attr] = 1;

		for (i = 0; i < attr_vals; i++) {
			/* create a new example table */
			for (ex = examples, ex_end = examples + size, child_ex = child_examples; ex < ex_end; ex++) {
				if (isnan(ex->x[best_attr])) {
					*child_ex = *ex;
					child_ex->weight *= attr_dist[i] / size_known;
					child_ex++;
				} else if ((int)ex->x[best_attr] == i) {
					*child_ex++ = *ex;
				}
			}

			node->children[i] = build_tree_(child_examples, child_ex - child_examples, depth + 1, node, args);
		}
					
		args->attr_split_so_far[best_attr] = 0;

		free(attr_dist);
		free(child_examples);
	} else {
		struct Example *examples_lt, *examples_ge, *ex_lt, *ex_ge;
		float size_lt, size_ge;

		printf("* %2d %3d %3d %f %f\n", depth, best_attr, size, best_split, best_score);

		assert(args->domain[best_attr] == FloatVar);

		ASSERT(examples_lt = (struct Example *)calloc(size, sizeof *examples));
		ASSERT(examples_ge = (struct Example *)calloc(size, sizeof *examples));

		size_lt = size_ge = 0.0;
		for (ex = examples, ex_end = examples + size; ex < ex_end; ex++)
			if (!isnan(ex->x[best_attr])) {
				if (ex->x[best_attr] < best_split)
					size_lt += ex->weight;
				else
					size_ge += ex->weight;
			}

		for (ex = examples, ex_end = examples + size, ex_lt = examples_lt, ex_ge = examples_ge; ex < ex_end; ex++)
			if (isnan(ex->x[best_attr])) {
				*ex_lt = *ex;
				*ex_ge = *ex;
				ex_lt->weight *= size_lt / (size_lt + size_ge);
				ex_ge->weight *= size_ge / (size_lt + size_ge);
				ex_lt++;
				ex_ge++;
			} else if (ex->x[best_attr] < best_split) {
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
			ASSERT(node->children = (struct SimpleTreeNode **)calloc(2, sizeof *node->children));

			node->children[0] = build_tree_(examples_lt, ex_lt - examples_lt, depth + 1, node, args);
			node->children[1] = build_tree_(examples_ge, ex_ge - examples_ge, depth + 1, node, args);
		} else {
			node = make_predictor(node, examples, size, args);
		}

		free(examples_lt);
		free(examples_ge);
	}

	return node;
}

struct SimpleTreeNode *
build_tree(double *x, double *y, double *w, int size, struct Args *args)
{
	struct Example *examples;
	struct SimpleTreeNode *tree;
	int i;

	/* create a tabel with pointers to examples */
	ASSERT(examples = (struct Example *)calloc(size, sizeof *examples));
	for (i = 0; i < size; i++) {
		examples[i].x = x + i * args->num_attrs;
		examples[i].y = y[i];
		examples[i].weight = w[i];
	}
	ASSERT(args->attr_split_so_far = (int *)calloc(args->num_attrs, sizeof(int)));
	tree = build_tree_(examples, size, 0, NULL, args);
	free(examples);
	free(args->attr_split_so_far);
	return tree;
}

void
destroy_tree(struct SimpleTreeNode *node, int type)
{
    int i;

    if (node->type != PredictorNode) {
        for (i = 0; i < node->children_size; i++)
            destroy_tree(node->children[i], type);
        free(node->children);
    }
    if (type == Classification)
        free(node->dist);
    free(node);
}

void
predict_classification_(double *x, struct SimpleTreeNode *node, struct Args *args, double *p)
{
    int i;

    while (node->type != PredictorNode) {
		if (isnan(x[node->split_attr])) {
            for (i = 0; i < node->children_size; i++) {
                predict_classification_(x, node->children[i], args, p);
            }
			return;
        } else if (node->type == DiscreteNode) {
            node = node->children[(int)x[node->split_attr]];
        } else {
            assert(node->type == ContinuousNode);
            node = node->children[x[node->split_attr] >= node->split];
        }
	}
	for (i = 0; i < args->cls_vals; i++) {
		p[i] += node->dist[i];
	}
}

void
predict_classification(double *x, int size, struct SimpleTreeNode *node, struct Args *args, double *p)
{
	int i, j;
	double *xx, *pp;

	for (i = 0; i < size; i++) {
		xx = x + i * args->num_attrs;
		pp = p + i * args->cls_vals;
		predict_classification_(xx, node, args, pp);
		double sum = 0;
		for (j = 0; j < args->cls_vals; j++) {
			sum += pp[j];
		}
		for (j = 0; j < args->cls_vals; j++) {
			pp[j] /= sum;
		}
	}
}

void
predict_regression_(double *x, struct SimpleTreeNode *node, struct Args *args, double *sum, double *n)
{
    int i;

    while (node->type != PredictorNode) {
		if (isnan(x[node->split_attr])) {
            for (i = 0; i < node->children_size; i++) {
                predict_regression_(x, node->children[i], args, sum, n);
            }
            return;
        } else if (node->type == DiscreteNode) {
            assert(x[node->split_attr] < node->children_size);
            node = node->children[(int)x[node->split_attr]];
        } else {
            assert(node->type == ContinuousNode);
            node = node->children[x[node->split_attr] > node->split];
        }
    }

    *sum += node->sum;
    *n += node->n;
}

void
predict_regression(double *x, int size, struct SimpleTreeNode *node, struct Args *args, double *p)
{
	int i;
	double sum, n;

	for (i = 0; i < size; i++) {
		sum = n = 0;
		predict_regression_(x + i * args->num_attrs, node, args, &sum, &n);
		p[i] = sum / n;
	}
}

void
save_tree_(struct SimpleTreeNode *node, struct Args *args, FILE *stream)
{
	int i;

	fprintf(stream, "{ %d %d ", node->type, node->children_size);

	if (node->type != PredictorNode)
		fprintf(stream, "%d %f ", node->split_attr, node->split);

	for (i = 0; i < node->children_size; i++)
		save_tree_(node->children[i], args);

	if (args->type == Classification) {
		for (i = 0; i < args->cls_vals; i++)
			fprintf(stream, "%d ", (int)node->dist[i]);
	} else {
		assert(args->type == Regression);
		fprintf(stream, "%f %f ", node->n, node->sum);
	}
	fprintf(stream, "} ");
}
