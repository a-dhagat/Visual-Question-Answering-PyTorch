from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision import transforms
# from vqa_dataset import VqaDataset
# device = torch.device("cuda")
class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=True):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250 # 250  # Steps

        # import ipdb; ipdb.set_trace()
        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()
        # import ipdb; ipdb.set_trace()
        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation
        self._batch_size = batch_size

        self.invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        self.train_dataset = train_dataset

        # self.val_dataset = val_dataset

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, writer, current_step):
        ############ 2.8 TODO
        # Should return your validation accuracy
        # val_batch_size = 16
        import numpy as np
        # len_dataloader = np.arange(len(self._val_dataset_loader))
        # random_data = np.random.choice(len_dataloader, 1000, replace=False)
        # import ipdb; ipdb.set_trace()
        count = 0
        iters = 0
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            # self._model.eval()
            # if iters==20:
            #     break
            # iters += 1

            img = batch_data['image'].cuda()
            question = batch_data['question'].cuda()
            gt_answer = batch_data['answer'].cuda()
            pred = self._model(img, question)

            gt_ans_sum = torch.sum(gt_answer,dim=1)
            # gt = torch.argmax(gt_ans_sum, dim=1)
            gt = torch.where(gt_ans_sum>1.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

            pred_ans = torch.argmax(pred,dim=1)
            predicted_answer = torch.zeros(gt.size()).cuda() # TODO
            predicted_answer.scatter_(1,pred_ans.view(-1,1),1.0)
            ground_truth_answer = gt.cuda() # TODO

            count += (predicted_answer * ground_truth_answer).sum()
            # print(count)
            # for pred_ans, gt_ans in zip(predicted_answer, ground_truth_answer):
            #     if pred_ans == gt_ans and gt_ans is not 5216:
            #         count += 1



        ############

            if self._log_validation:
                ############ 2.9 TODO
                # you probably want to plot something here
                # if iters==1:
                idx = np.random.choice(self._batch_size,1)
                binary_question = batch_data['question'][idx]
                word_question = ''
                for i in binary_question.squeeze():
                    word = torch.argmax(i).detach().cpu().numpy().item()
                    # print(word)
                    if word<5746:
                        word_question += ' ' + (list(self.train_dataset.question_word_to_id_map.keys())[word])
                writer.add_text('question: ', word_question, current_step)
                
                gt_answer = torch.argmax(gt[idx].squeeze()).detach().cpu().numpy().item()
                if gt_answer<5216:
                    gt_answer_word = list(self.train_dataset.answer_to_id_map.keys())[gt_answer]
                    writer.add_text('answer: ', gt_answer_word, current_step)

                img = batch_data['image'][idx]
                img = self.invTrans(img.squeeze())
                writer.add_image('image: ', img.detach().cpu().numpy(), current_step)
                # pass

                ############
        accuracy = count/(self._batch_size*len(self._val_dataset_loader))
        # accuracy = count/(self._batch_size*20)
        return accuracy
        # raise NotImplementedError()

    def train(self):
        writer = SummaryWriter('./runs/run_full')
        self.loss_obj = nn.CrossEntropyLoss(size_average=True).cuda()
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            # import ipdb; ipdb.set_trace()

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                # import ipdb; ipdb.set_trace()
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                img = batch_data['image'].cuda()
                question = batch_data['question'].cuda()
                gt_answer = batch_data['answer'].cuda()
                pred = self._model(img, question)

                # ground_truth_answer_vote = torch.zeros((self._batch_size,5217))
                gt_ans_sum = torch.sum(gt_answer,dim=1)
                gt = torch.argmax(gt_ans_sum, dim=1)
                # for j,i in enumerate(gt):
                #     ground_truth_answer_vote[j][i]=1.0

                predicted_answer = pred.cuda() # TODO
                ground_truth_answer = gt.cuda() # TODO

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    writer.add_scalar('train_loss: ',loss.item(), current_step)
                    torch.save(self._model.state_dict(), "./saved_model_full/full_model_at_epoch_" + str(epoch) + "_currentstep_" + str(current_step) + ".pth")

                    ############

                # import ipdb; ipdb.set_trace()
                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate(writer, current_step)
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    writer.add_scalar('valid_acc: ',val_accuracy, current_step)

                    ############
