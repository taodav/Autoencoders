


class AutoencoderEvaluator:
    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model


    def print_decoded_with_image(self, image, decoded):
        raise NotImplementedError