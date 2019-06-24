import __init_path__
import env
import os

def get_kitti_config_1():
    config = {'train_sequences': ['00',
                                  '01',
                                  '02',
                                  '03',
                                  '04',
                                  '05',
                                  '06',
                                  '07'
                                  ],
              'val_sequences': [],
              'test_sequences': ['08',
                                 '09',
                                 '10'
                                 ],
              }
    return config


def get_kitti_config_2():
    config = {'train_sequences': ['00',
                                  '02',
                                  '08',
                                  '09'
                                  ],
              'val_sequences': [],
              'test_sequences': ['03',
                                 '04',
                                 '05',
                                 '06',
                                 '07',
                                 '10'
                                 ],
              'id': 1,
              }
    return config


def get_discoman_iros_1_config():
    config = {
        "train_sequences": [
                            "000001"
                             ],
        "val_sequences": ["000002",
                          ],
        "test_sequences": ["000044"
                           ],
         }
    return config


def get_discoman_debug_config():
    config = {
        "train_sequences": [
                            "000001"
                             ],
        "val_sequences": ["000001",
                          ],
        "test_sequences": ["000001"
                           ],
         }
    return config


def get_tum_config():
    config = {"id": 3}
    return config