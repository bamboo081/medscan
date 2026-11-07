import numpy as np
from medscan.evaluation import metrics_from_probs


def test_patient_aggregation():
    probs = np.array([[.9,.1],[.7,.3],[.2,.8],[.1,.9]])
    m = metrics_from_probs(probs, [0,0,1,1], ['a','b'], ['p1','p1','p2','p2'])
    assert m['macro_f1'] == 1 and m['patient_macro_f1'] == 1
    assert m['patients'] == 2
