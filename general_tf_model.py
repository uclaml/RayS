import torch
import numpy as np
import torch.nn as nn

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

class GeneralTFModel(nn.Module):
    def __init__(self, model_logits, x_input, sess, n_class=10, im_mean=None, im_std=None):
        super(GeneralTFModel, self).__init__()
        self.model_logits = model_logits
        self.x_input = x_input
        self.sess = sess
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image_tf = np.moveaxis(image.cpu().numpy(), 1, 3)
        logits = self.sess.run(self.model_logits, {self.x_input: image_tf})
        return torch.from_numpy(logits).cuda()

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
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        self.num_queries += image.size(0)

        image_tf = np.moveaxis(image.cpu().numpy(), 1, 3)
        logits = self.sess.run(self.model_logits, {self.x_input: image_tf})
        
        return torch.from_numpy(logits).cuda()

    def predict_label(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict
