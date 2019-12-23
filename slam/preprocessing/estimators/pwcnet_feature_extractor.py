from .pwcnet_estimator import PWCNetEstimator


class PWCNetFeatureExtractor(PWCNetEstimator):

    def __init__(self, *args, **kwargs):
        super(PWCNetFeatureExtractor, self).__init__(name='PWCNetExtractor',
                                                     *args,
                                                     **kwargs)

    def get_nn_opts(self):
        nn_opts = super(PWCNetFeatureExtractor, self).get_nn_opts()
        nn_opts['ret_feat'] = True
        return nn_opts

    def _convert_model_output_to_prediction(self, output):
        return output

    def _run_model_inference(self, model_input):
        return self.model.return_features(model_input, batch_size=1, verbose=False)
