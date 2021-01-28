import numpy as np
from scipy.special import softmax
import torch
from torch.autograd import Variable


def create_context_batch(end_idx, image_vectors, model, device, start_idx=0, exact_batch_size=None):

    if not exact_batch_size:
        exact_batch_size = end_idx

    contexts = np.zeros((exact_batch_size, 2048))
    for im_idx in range(start_idx, start_idx + exact_batch_size):
        e = np.zeros((7, 7))
        context = np.zeros(2048)
        for row in range(7):
            for col in range(7):
                a = Variable(torch.FloatTensor(image_vectors[im_idx, :, row, col])).to(device)
                e[row, col] = model.attention(a).detach().cpu().numpy()
        alpha = softmax(e.flatten()).reshape((7, 7))
        for row in range(7):
            for col in range(7):
                context += alpha[row, col] * image_vectors[im_idx, :, row, col]
        contexts[im_idx - start_idx, :] = context
    return contexts
