import dill
from multiprocessing import Pool

FORMAT_STR = '{:<25} {}'


def limit_resources(allow_growth=True,
                    per_process_gpu_memory_fraction=0.33,
                    cuda_visible_devices=0,
                    cpu_count=8,
                    cpu_threads=16):
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto(
        device_count={'CPU': cpu_count},
        intra_op_parallelism_threads=cpu_threads,
        inter_op_parallelism_threads=cpu_threads)

    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    session = tf.Session(config=config)
    K.tensorflow_backend.set_session(session)

    print(FORMAT_STR.format('CUDA visible devices:', cuda_visible_devices))
    print(FORMAT_STR.format('Available GPUs:', ', '.join(K.tensorflow_backend._get_available_gpus())))
    print(FORMAT_STR.format('Allow growth:', allow_growth))
    print(FORMAT_STR.format('GPU memory fraction:', per_process_gpu_memory_fraction))
    print(FORMAT_STR.format('Number of CPU:', cpu_count))
    print(FORMAT_STR.format('Number of CPU threads:', cpu_threads))
    return session


def fix_random_seed(random_seed=42):
    import torch
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    from numpy.random import seed
    seed(random_seed)

    import tensorflow
    tensorflow.set_random_seed(random_seed)

    import random
    random.seed(random_seed)

    print()
    print(FORMAT_STR.format('Random seed:', random_seed))


def set_warning_options():
    import numpy as np
    np.seterr(all='raise')


def set_computation(random_seed=42, *args, **kwargs):
    print('\n' + '='*98 + '\n')
    print('Computation settings:\n')

    session = limit_resources(*args, **kwargs)
    fix_random_seed(random_seed)
    set_warning_options()

    print('\n' + '='*98 + '\n\n')
    return session


def apply_dumped_function(dumped_function, args, kwargs):
    target_function = dill.loads(dumped_function)
    res = target_function(*args, **kwargs)
    return res


def dump_function(target_function, *args, **kwargs):
    dumped_function = dill.dumps(target_function)
    return apply_dumped_function, (dumped_function, args, kwargs)


def with_gpu(f, **computation_kwargs):
    def gpu_wrapper(*args, **kwargs):
        session = set_computation(**computation_kwargs)
        f(*args, **kwargs)
        session.close()
    return gpu_wrapper


def make_memory_safe(f, **computation_kwargs):
    def multiprocessing_wrapper(*args, **kwargs):
        f_with_gpu = with_gpu(f, **computation_kwargs)
        with Pool(1) as pool:
            pool.apply(*dump_function(f_with_gpu, *args, **kwargs))
    return multiprocessing_wrapper
