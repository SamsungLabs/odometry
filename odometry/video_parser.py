import os
import subprocess
import cv2

class VideoParser():
    def __init__(self,
                 image_manager,
                 video_path):
        self.image_manager = image_manager
        self.video_path = video_path
        self.num_frames = 0

        self.video_duration = -1
        self._get_video_duration()

    def _get_video_duration(self):
        filestat = subprocess.Popen(["ffprobe", self.video_path],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        for byte_string in filestat.stdout.readlines():
            string = str(byte_string).lower()[2:-1]
            if 'duration' in string:
                duration_string = string.split()[1][:-1].replace('.', ':')
                h, m, s, ms = map(float, duration_string.split(':'))
                ms /= 100
                self.video_duration = ((h * 60 + m) * 60 + s) + ms

    @property
    def fps(self):
        if self.video_duration:
            return (self.num_frames - 1) / self.video_duration
        return None

    def parse(self):
        self._get_video_duration()

        video = cv2.VideoCapture(self.video_path)
        while video.isOpened():
            success, image = video.read()
            if not success:
                break
            self.image_manager.save_image(image)
            print(self.image_manager.num_images, end='\r')
            self.num_frames += 1

        print('Total: {} frames'.format(self.num_frames))
        print('Saved: {} frames'.format(self.image_manager.num_images))
        video.release()

    def __repr__(self):
        return 'VideoParser(image_manager={}, video_path={}, video_duration={}, num_frames={})'.format(
            self.image_manager, self.video_path, self.video_duration, self.num_frames)
