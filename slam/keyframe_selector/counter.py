from slam.keyframe_selector.base_keyframe_selector import BaseKeyfameSelector


class CounterKeyFrameSelector(BaseKeyfameSelector):
    def __init__(self, period):
        self.period = period

    def is_key_frame(self, keyframe, frame, index):
        return index % self.period == 0
