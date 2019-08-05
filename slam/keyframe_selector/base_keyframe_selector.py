class BaseKeyfameSelector:
    def is_key_frame(self, keyframe, frame, index) -> bool:
        raise RuntimeError('Not implemented')
