import cmlreaders
import ptsa
import ramutils
from sklearn.externals import joblib
from utils import*
import tensorflow as tf



rhino_root = '/Volumes/RHINO'
import pandas as pd

def load_data(subject, session, experiement):

    subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
    dataset = joblib.load(subject_dir)


    dataset_enc = select_phase(dataset)
    dataset_enc['X'] = normalize_sessions(dataset_enc['X'], dataset_enc['session'])   # select only encoding data


    train_mask = dataset_enc['session'] == session


    train_x, train_y = pd.DataFrame(dataset_enc['X'][~train_mask,:]), pd.DataFrame(dataset_enc['y'][~train_mask])
    test_x, test_y = pd.DataFrame(dataset_enc['X'][train_mask,:]), pd.DataFrame(dataset_enc['y'][train_mask])

    return (train_x,train_y), (test_x,test_y)




def normalize_sessions(pow_mat, event_sessions):
    sessions = np.unique(event_sessions)
    for sess in sessions:
        sess_event_mask = (event_sessions == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat


def get_sessions(subject, experiement):

    subject_dir = rhino_root + '/scratch/tphan/joint_classifier/FR1/' + subject + '/dataset.pkl'
    dataset = joblib.load(subject_dir)
    dataset_enc = select_phase(dataset)
    return np.unique(dataset_enc['session'])




# def train_input_fn(features, labels, batch_size):
#     """An input function for training"""
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#
#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#     # Return the dataset.
#     return dataset
#
#
# def eval_input_fn(features, labels, batch_size):
#     """An input function for evaluation or prediction"""
#     features=dict(features)
#     if labels is None:
#         # No labels, use only features.
#         inputs = features
#     else:
#         inputs = (features, labels)
#
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)
#
#     # Batch the examples
#     assert batch_size is not None, "batch_size must not be None"
#     dataset = dataset.batch(batch_size)
#
#     # Return the dataset.
#     return dataset
