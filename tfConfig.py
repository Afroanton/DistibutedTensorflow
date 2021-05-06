import json
import os


def config():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.pop('TF_CONFIG', None)
    os.environ['TF_CONFIG'] = json.dumps({
        "cluster": {
            "chief": ["192.168.1.115:12345"],
            "worker": ["192.168.1.187:23456", "192.168.1.122:34567", "192.168.1.116:45678"]
        },
        "task": {"type": "chief", "index": 0}
    })
