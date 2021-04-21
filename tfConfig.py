import json
import os


def config(worker_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.pop('TF_CONFIG', None)
    os.environ['TF_CONFIG'] = json.dumps({
        "cluster": {
            "chief": ["ip:12345"],
            "worker": ["ip:23456"]
        },
        "task": {"type": "chief", "index": 0}
    })
