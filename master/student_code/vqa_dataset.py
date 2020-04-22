from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from PIL import Image
import torch
from torchvision import transforms


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None, max_list_length=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        if transform is not None:
            self._transform = transform
        else:
            self._transform = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        
        self.question_ids = self._vqa.getQuesIds()
        self.fixed_str = '000000000000'

        self._pre_encoder = pre_encoder
        self._cache_location = cache_location

        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass
        
        # import ipdb; ipdb.set_trace()

        # Create the question map if necessary
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            question_sentences = []
            # question_ids = self._vqa.getQuesIds()

            for question_id in self.question_ids:
                question_sentences.append(self._vqa.qqa[question_id]['question'])

            word_list = self._create_word_list(question_sentences)
            self.question_word_to_id_map = self._create_id_map(word_list, question_word_list_length)
            ############
            # raise NotImplementedError()
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            answer_sentence_list = []
            for question_id in self.question_ids:
                answer_list = self._vqa.qa[question_id]['answers']
                for item in answer_list:
                    answer_sentence_list.append(item['answer'])
            
            self.answer_to_id_map = self._create_id_map(answer_sentence_list, answer_list_length)
            ############
            # raise NotImplementedError()
        else:
            self.answer_to_id_map = answer_to_id_map

        # import pdb; pdb.set_trace()


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        """
        https://machinelearningmastery.com/clean-text-machine-learning-python/
        """
        import string
        table = str.maketrans('','',string.punctuation)
        # import ipdb; ipdb.set_trace()
        word_list = []
        if type(sentences) == list:
            for sentence in sentences:
                words = sentence.split(" ")
                word_list += [word.translate(table).lower() for word in words]
        else:
            words = sentences.split(" ")
            word_list += [word.translate(table).lower() for word in words]
        ############
        # raise NotImplementedError()
        return word_list


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO
        from collections import Counter
        
        # import pdb; pdb.set_trace()
        word_rank_list = Counter(word_list).most_common(max_list_length)
        
        id_map = {}
        for idx, (word,_) in enumerate(word_rank_list):
            id_map[word] = idx

        ############
        # raise NotImplementedError()
        return id_map


    def __len__(self):
        ############ 1.8 TODO
        # return len(self._vqa.imgToQA)
        return len(self._vqa.questions['questions'])
        # return len(self._vqa.getQuesIds())
        ############
        # raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        image_id_from_idx = self._vqa.questions['questions'][idx]['image_id']
        question_id_from_idx = self._vqa.questions['questions'][idx]['question_id']
        question_sentence = self._vqa.questions['questions'][idx]['question']
        answer_sentences = [ans['answer'] for ans in self._vqa.qa[question_id_from_idx]['answers']]

        # import ipdb; ipdb.set_trace()

        ############

        # if self._cache_location is not None and self._pre_encoder is not None:
        #     ############ 3.2 TODO
        #     # implement your caching and loading logic here

        #     ############
        #     raise NotImplementedError()
        # else:
        ############ 1.9 TODO
        # load the image from disk, apply self._transform (if not None)
        # import ipdb; ipdb.set_trace()
        image_id_from_idx_string = self.fixed_str + str(image_id_from_idx)
        truncated_image_id_from_idx = image_id_from_idx_string[-12:]
        img_file_path = self._image_dir + '/' + self._image_filename_pattern.format(truncated_image_id_from_idx)
        image = Image.open(img_file_path)
        image = self._transform(image)
        ############
        # raise NotImplementedError()

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        question_encoding = torch.zeros(26,5746)
        answer_encoding = torch.zeros(10,5216)

        question_word_list = self._create_word_list(question_sentence)
        for idx, word in enumerate(question_word_list):
            if idx >= self._max_question_length:
                break
            map_idx = self.question_word_to_id_map[word]
            question_encoding[idx][map_idx] = 1

        # answer_sentence_list = self._create_word_list(answer_sentences)
        for idx, answer in enumerate(answer_sentences):
            map_idx = self.answer_to_id_map[answer]
            answer_encoding[idx][map_idx] = 1
        
        data = {'image':image, 'question':question_encoding, 'answer':answer_encoding}
        return data

        ############
        # raise NotImplementedError()
