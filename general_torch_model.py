import torch
import numpy as np
import torch.nn as nn


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict

    def predict_ensemble(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            output = self.model(image)
            output.zero_()
            for i in range(10):
                output += self.model(image)
                self.num_queries += image.size(0)
        _, predict = torch.max(output.data, 1)
        return predict

    def get_num_queries(self):
        return self.num_queries

    def get_gradient(self, loss):
        loss.backward()
