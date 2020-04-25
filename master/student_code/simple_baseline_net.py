import torch
import torch.nn as nn
from external.googlenet.googlenet import googlenet
# from .. import external
        
def convert_to_bow(question_encoding):
    # import ipdb; ipdb.set_trace()
    bow_freq_encoding = torch.sum(question_encoding, dim=1)
    bow_binary_encoding = torch.where(bow_freq_encoding>0.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    return bow_binary_encoding

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()
	    ############ 2.2 TODO
        # import ipdb; ipdb.set_trace()
        # in_dim = 1024 + 5717 // 1000 + (5717->1024)
        # hid_dim = 
        # out_dim = 
        self.googlenet = googlenet(pretrained=True)
        # for param in self.googlenet.parameters():

        # self.branch_img_feats = googlenet(pretrained=True)
        self.word_embeddings = nn.Sequential(nn.Linear(5747, 1024))
        in_dim = 1024 + 1000
        hid_dim = 1024
        out_dim = 5217
        self.softmax_layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.ReLU(),
            # nn.Dropout(0.4,inplace=True),
            # nn.Linear(hid_dim, out_dim),
        )

	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        # import ipdb; ipdb.set_trace()
        img_feat = self.googlenet(image)
        bow_encoding = convert_to_bow(question_encoding)
        question_low_dim = self.word_embeddings(bow_encoding)
        concat_vec = torch.cat((img_feat.transpose(1,0), question_low_dim.transpose(1,0))).transpose(1,0)
        pred = self.softmax_layer(concat_vec)
        return pred

	    ############
        # raise NotImplementedError()
