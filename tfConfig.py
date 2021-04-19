import json
import os


def config(worker_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.pop('TF_CONFIG', None)
    tf_config = {
        'cluster': {
            'worker': ['localhost:138', 'localhost:101']
        },
        'task': {'type': 'worker', 'index': worker_index}
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
