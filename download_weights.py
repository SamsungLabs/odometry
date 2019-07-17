import os
import gdown
import subprocess
from tqdm import trange
from pathlib import Path


if __name__ == '__main__':

    url = list()
    destination = list()

    url.append('https://drive.google.com/uc?id=1mjb4ioDRH8ViGbui52stSUDwhkGrDXy8')
    destination.append(Path(os.path.realpath(__file__)).parent/'weights/struct2depth_model_kitti.tar.gz')

    url.append(' https://drive.google.com/uc?id=11SzYIezaF8yaIVKAml7kPdqgncna2vj7')
    destination.append(Path(os.path.realpath(__file__)).parent/'weights/pwcnet.ckpt-595000.data-00000-of-00001')

    url.append(' https://drive.google.com/uc?id=1guw6rpVRsO9OfKnuKGGeUY0kpNfJf4yy')
    destination.append(Path(os.path.realpath(__file__)).parent/'weights/pwcnet.ckpt-595000.index')

    url.append(' https://drive.google.com/uc?id=1w8DgWut4APWZpprGxPvCbvmg8sJZ11-u')
    destination.append(Path(os.path.realpath(__file__)).parent/'weights/pwcnet.ckpt-595000.meta')

    for i in trange(len(url)):
        gdown.download(url[i], destination[i].as_posix(), quiet=False)

    subprocess.run(f'tar -C {destination[0].parent.as_posix()} -xf {destination[0]}', shell=True, check=True)
    subprocess.run([f'rm {destination[0].as_posix()}'], shell=True, check=True)
