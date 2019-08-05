from slam.keyframe_selector.base_keyframe_selector import BaseKeyfameSelector


class CounterKeyFrameSelector(BaseKeyfameSelector):
    def is_key_frame(self, keyframe, frame, index):
        new_keyframe = not bool(index % 10)
        kidnapped = False
        return new_keyframe, kidnapped
