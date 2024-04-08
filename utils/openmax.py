import numpy as np
import pickle

from utils.evt_fitting import query_weibull
from utils.evt_fitting import weibull_tailfitting
from utils.openmax_utils import *


NCLASSES = 16
ALPHA_RANK = NCLASSES
WEIBULL_TAIL_SIZE = NCLASSES
LABELS = range(NCLASSES)


def create_model(model, dataset, fold):
    logits_output, softmax_output = get_activations(dataset, model)
    _, y_data = dataset.get_all()
    correct_index = get_correct_classified(y_data, softmax_output)

    logits_correct = logits_output[correct_index]
    y_correct = y_data[correct_index]
    y_correct = np.argmax(y_correct, axis=1)

    av_map = {}

    for label in LABELS:
        av_map[label] = logits_correct[y_correct == label]

    feature_mean = []
    feature_distance = []

    for label in LABELS:
        mean = compute_mean_vector(av_map[label])
        distance = compute_distance_dict(mean, av_map[label])
        feature_mean.append(mean)
        feature_distance.append(distance)

    build_weibull(mean=feature_mean, distance=feature_distance,
                  tail=WEIBULL_TAIL_SIZE, fold=fold)


def build_weibull(mean, distance, tail, fold):
    model_path = f'models/weibull_model_{fold}.pkl'
    weibull_model = {}

    for label in LABELS:
        weibull_model[label] = {}
        weibull = weibull_tailfitting(
            mean[label], distance[label], tailsize=tail)
        weibull_model[label] = weibull

    with open(model_path, 'wb') as file:
        pickle.dump(weibull_model, file)


def recalibrate_scores(weibull_model, activation_vector, alpharank=ALPHA_RANK, distance_type='eucos'):
    ranked_list = activation_vector.argsort().ravel()[::-1]
    alpha_weights = [
        ((alpharank+1) - i) / float(alpharank) for i in range(1, alpharank+1)
    ]
    ranked_alpha = np.zeros(NCLASSES)

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_scores = []
    openmax_scores_u = []

    for label in LABELS:
        weibull = query_weibull(label, weibull_model, distance_type)
        av_distance = compute_distance(
            weibull[1], activation_vector.ravel())
        wscore = weibull[2][0].w_score(av_distance)
        modified_score = activation_vector[0][label] * \
            (1 - wscore*ranked_alpha[label])
        openmax_scores += [modified_score]
        openmax_scores_u += [activation_vector[0][label] - modified_score]

    openmax_scores = np.asarray(openmax_scores)
    openmax_scores_u = np.asarray(openmax_scores_u)

    openmax_probab, prob_u = compute_openmax_probability(
        openmax_scores, openmax_scores_u)
    return openmax_probab, prob_u


def compute_openmax_probability(openmax_scores, openmax_scores_u):
    e_k = np.exp(openmax_scores)
    e_u = np.exp(np.sum(openmax_scores_u))
    openmax_arr = np.concatenate((e_k, e_u), axis=None)
    total_denominator = np.sum(openmax_arr)
    prob_k = e_k / total_denominator
    prob_u = e_u / total_denominator
    res = np.concatenate((prob_k, prob_u), axis=None)
    return res, prob_u


def compute_openmax(activation_vector, fold):
    model_path = f'models/weibull_model_{fold}.pkl'
    with open(model_path, 'rb') as file:
        weibull_model = pickle.load(file)
    openmax, prob_u = recalibrate_scores(weibull_model, activation_vector)
    return openmax, prob_u
