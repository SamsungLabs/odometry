import os
import subprocess
import cv2

from .image_utils import save_image


def create_image_filename(index):
    return str(index).zfill(6)


def get_video_duration(video_path):
    filestat = subprocess.Popen(["ffprobe", video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    for byte_string in filestat.stdout.readlines():
        string = str(byte_string).lower()[2:-1]
        if 'duration' in string:
            duration_string = string.split()[1][:-1].replace('.', ':')
            h, m, s, ms = map(float, duration_string.split(':'))
            ms /= 100
            video_duration = ((h * 60 + m) * 60 + s) + ms
            return video_duration
    return None


def get_fps(video_path, image_count):
    video_duration = get_video_duration(video_path)
    if video_duration is not None:
        fps = (image_count - 1) / video_duration
        return fps
    return None


def parse_video(video_path, image_dir):
    os.makedirs(image_dir)

    count = 0
    video = cv2.VideoCapture(self.video_path)
    while video.isOpened():
        success, image = video.read()
        if not success:
            break
        image_filepath = os.path.join(image_dir, create_image_filename(count))
        save_image(image, image_filepath)
        count += 1

    print('Total: {} frames'.format(count))
    video.release()

    return get_fps(video_path, count)
