import torch
import torch.nn as nn
from external.googlenet.googlenet import googlenet
# from .. import external
        
def convert_to_bow(question_encoding):
    bow_freq_encoding = torch.sum(question_encoding)
    bow_binary_encoding = torch.where(bow_freq_encoding>0.0, torch.tensor(1.0), torch.tensor(0.0))
    return bow_binary_encoding

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super(SimpleBaselineNet, self).__init__()
	    ############ 2.2 TODO
        # import ipdb; ipdb.set_trace()
        # in_dim = 1024 + 5717 // 1000 + (5717->1024)
        # hid_dim = 
        # out_dim = 
        self.reduce_to_low_dim = nn.Linear(5717, 1024)
        self.googlenet = googlenet(pretrained=True)
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.4,inplace=True),
        #     nn.Linear(hid_dim, out_dim),
        # )

	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        img_feat = self.googlenet(image)
        bow_encoding = convert_to_bow(question_encoding)
        question_low_dim = self.reduce_to_low_dim(bow_encoding)
        # pred = self.classifier(torch.cat(img_feat, question_low_dim), dim=1)
        return pred

	    ############
        # raise NotImplementedError()
