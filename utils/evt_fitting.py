import numpy as np
import libmr


def weibull_tailfitting(mean, distance, tailsize, distance_type='eucos'):
    weibull_model = {}
    distance_scores = np.array(distance[distance_type])
    meantrain_vec = np.array(mean)

    weibull_model[f'distances_{distance_type}'] = distance_scores
    weibull_model['mean_vec'] = meantrain_vec
    weibull_model['weibull_model'] = []

    mr = libmr.MR()

    tailtofit = sorted(distance_scores)[-tailsize:]
    mr.fit_high(tailtofit, len(tailtofit))
    weibull_model['weibull_model'] += [mr]
    return weibull_model


def query_weibull(label, weibull_model, distance_type='eucos'):
    category_weibull = []
    category_weibull += [weibull_model[label][f'distances_{distance_type}']]
    category_weibull += [weibull_model[label]['mean_vec']]
    category_weibull += [weibull_model[label]['weibull_model']]
    return category_weibull
