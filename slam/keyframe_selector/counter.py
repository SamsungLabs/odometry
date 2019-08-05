from slam.keyframe_selector.base_keyframe_selector import BaseKeyfameSelector


class CounterKeyFrameSelector(BaseKeyfameSelector):
    def is_key_frame(self, keyframe, frame, index):
        return not bool(index % 10)
