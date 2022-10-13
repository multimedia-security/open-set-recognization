''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import copy
import logging

import numpy as np
import torch
import torch.utils.data as td
from torch.autograd import Variable
from tqdm import tqdm
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.other_tools.display as nii_warn
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger('iCARL')


class IncrementalLoader(td.Dataset):
    def __init__(self, data, classes, active_classes, transform=None, cuda=False,
                 oversampling=True):
        oversampling = not oversampling
        self.len = 0
        # sort_index = np.argsort(labels)
        self.f_load_data = data.load_data
        self.m_input_dirs = data.input_dirs
        self.m_input_dims = data.input_dims
        self.m_input_exts = data.input_exts
        self.m_input_all_dim = data.input_all_dim
        self.m_output_dirs = data.output_dirs
        self.m_output_exts = data.output_exts
        self.m_output_dims = data.output_dims

        self.data = data.seq_info
        self.m_input_reso = data.input_reso
        self.labels = data.labels
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.active_classes = active_classes
        self.limited_classes = {}
        self.total_classes = 18
        self.means = {}
        self.cuda = cuda
        # self.weights = np.zeros(self.total_classes * self.class_size)
        self.indices = {}
        self.__class_indices()
        self.ori_indices = self.indices.copy()
        self.over_sampling = oversampling
        # f(label) = new_label. We do this to ensure labels are in increasing order. For example, even if first increment chooses class 1,5,6, the training labels will be 0,1,2
        self.indexMapper = {}
        self.no_transformation = False
        self.transformLabels()
        # model_id = 'wav2vec2-base-960h'
        # self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        # self.model = Wav2Vec2Model.from_pretrained(model_id)
        # self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def get_audio_embeddings(self, audio):
        with torch.no_grad():
            input_values = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
            hidden_states = self.model(input_values).last_hidden_state
            return hidden_states

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((len(self.labels), max(self.labels) + 1))
        b[np.arange(len(self.labels)), self.labels] = 1
        self.labels = b

    def __class_indices(self):
        cur = 0
        for temp in range(0, self.total_classes):
            # print(np.argwhere(np.array(self.labels) == temp))
            cur_len = len(np.argwhere(np.array(self.labels) == temp))
            self.indices[temp] = (cur, cur + cur_len)
            if temp in self.active_classes:
                self.len += cur_len
            cur += cur_len

    def add_class(self, n):
        logger.debug("Adding Class %d", n)
        if n in self.active_classes:
            logger.debug('class {} have existed'.format(n))
            return
        # Mapping each new added classes to new label in increasing order; we switch the label so that the resulting confusion matrix is always in order
        # regardless of order of classes used for incremental training.
        indices = len(self.indexMapper)
        if not n in self.indexMapper:
            self.indexMapper[n] = indices
        self.active_classes.append(n)
        # self.len += self.indices[n][1] - self.indices[n][0]
        self.update_length()

    def set_detect_novel(self, exist, new):
        self.active_classes = []
        self.indices = self.ori_indices.copy()
        for c in range(exist):
            self.active_classes.append(c)
        self.active_classes.append(new)
        for i in self.indices.keys():
            if i not in range(exist) and i != new:
                self.indices[i] = (self.indices[i][0], self.indices[i][0])
        self.update_length()

    def update_length(self):
        '''
        Function to compute length of the active elements of the data. 
        :return: 
        '''

        len_var = 0
        for a in self.active_classes:
            len_var += self.indices[a][1] - self.indices[a][0]
        self.len = len_var

        return

    def limit_class(self, n, k):
        if k == 0:
            logger.warning("Removing class %d", n)
            self.remove_class(n)
            self.update_length()
            return False
        if k > self.indices[n][1] - self.indices[n][0]:
            k = self.indices[n][1] - self.indices[n][0]
        if n in self.limited_classes:
            self.limited_classes[n] = k
            # Remove this line; this turns off oversampling
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.update_length()
            return False
        else:
            if not self.over_sampling:
                self.indices[n] = (self.indices[n][0], self.indices[n][0] + k)
            self.limited_classes[n] = k
            self.update_length()
            return True

    def limit_class_and_sort(self, n, k, model, cur_len):
        ''' This function should only be called the first time a class is limited. To change the limitation, 
        call the limiClass(self, n, k) function 
        
        :param n: Class to limit
        :param k: No of exemplars to keep 
        :param model: Features extracted from this model for sorting. 
        :return: 
        '''

        if self.limit_class(n, k):
            start = self.indices[n][0]
            end = self.indices[n][1]
            buff = []
            # index = random.sample(range(start, end), k)
            # for i in index:
            #     buff.append(self.data[i])
            # np.zeros(np.array(self.data[start:end]).shape)
            images = []
            # Get input features of all the images of the class
            for ind in range(start, end):
                img = self.getdata(self.data[ind])
                if "torch" in str(type(img)):
                    img = img.numpy()
                images.append(img)

            # images = images

            # tensor_data = zip(images, lengths)
            # for i in range(len(images)):
            #     # print(images[i].shape)
            #     if lengths[i] < max_l:
            #         img_buff = images[i].repeat(max_l // lengths[i] + 1, axis=0)
            #         images[i] = img_buff[0:max_l]
            # st = time.time()
            # images = torch.tensor(images)
            # et = time.time()
            # print('cost time {}'.format(et - st))
            # collate_fn = nii_collate_fn.customize_collate
            kwargs = {'num_workers': 4, 'pin_memory': True}
            # if args.cuda else {}
            img_l_dataloader = torch.utils.data.DataLoader(images, batch_size=16, shuffle=False, drop_last=True,**kwargs)

            # data_tensor = torch.tensor(images)

            # if self.cuda:
            #     data_tensor = data_tensor.cuda()

            # Get features
            # for data, y, target, datalength in data_tensor:
            features = torch.zeros([len(images), cur_len], dtype=torch.float)
            for idx, data in enumerate(tqdm(img_l_dataloader)):
                features[idx * 16: (idx + 1) * 16] = model.forward(Variable(data.cuda()))
            # features = torch.tensor(np.array(features).reshape(len(features)))
            features_copy = copy.deepcopy(features.data)
            features = features.cuda()
            mean = torch.mean(features, 0, True)
            list_of_selected = []

            # Select exemplars
            for exmp_no in range(0, min(k, end - start)):
                if exmp_no > 0:
                    to_add = torch.sum(features_copy[0:exmp_no], dim=0).unsqueeze(0)  # 对每个样本特征求和
                    if self.cuda:
                        to_add = to_add.cuda()
                    features_temp = (features + Variable(to_add)) / (exmp_no + 1) - mean
                else:
                    features_temp = features - mean
                features_norm = torch.norm(features_temp.data, 2, dim=1)
                if self.cuda:
                    features_norm = features_norm.cpu()
                arg_min = np.argmin(features_norm.numpy())
                if arg_min in list_of_selected:
                    assert False
                list_of_selected.append(arg_min)
                buff.append(self.data[start + arg_min])
                features_copy[exmp_no] = features.data[arg_min]
                features[arg_min] = features[arg_min] + 10000000

                # logger.debug("Arg Min: %d", arg_min)
            print("Exmp shape", len(buff[0:min(k, end - start)]))
            self.data[start:start + min(k, end - start)] = buff[0:min(k, end - start)]

        self.update_length()

    def remove_class(self, n):
        while n in self.active_classes:
            self.active_classes.remove(n)
        self.update_length()

    def __len__(self):
        return self.len

    def get_start_index(self, n):
        '''
        
        :param n: 
        :return: Returns starting index of classs n
        '''
        return self.indices[n][0]

    def getIndexElem(self, bool):
        self.no_transformation = bool

    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        assert (index < len(self.data))
        assert (index < self.len)

        length = 0
        temp_a = 0
        old_len = 0
        for a in self.active_classes:
            temp_a = a
            old_len = length
            length += self.indices[a][1] - self.indices[a][0]
            if length > index:
                break
        base = self.indices[temp_a][0]
        incre = index - old_len
        if temp_a in self.limited_classes:
            incre = incre % self.limited_classes[temp_a]
        index = base + incre
        img = self.getdata(self.data[index])
        # img = self.get_audio_embeddings(img.reshape(img.shape[0]))
        # input_values = self.processor(img.squeeze(-1), sampling_rate=16000, return_tensors="pt").input_values
        # INFERENCE
        # retrieve logits & take argmax
        # logits = self.model(input_values).logits
        # img = self.pad(img)

        return img, self.labelsNormal[index]

    # 填充音频
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len) + 1
        padded_x = x.repeat(num_repeats, axis=0)[0:max_len]
        # images[i] = img_buff[0:max_l]
        # padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

    def getdata(self, tmp_seq_info):
        """ __getitem__(self, idx):
        Return input, output

        For test set data, output can be None
        """
        # try:
        #     tmp_seq_info = self.m_seq_info[idx]
        # except IndexError:
        #     nii_warn.f_die("Sample %d is not in seq_info" % (idx))

        # file_name
        file_name = tmp_seq_info.seq_name

        # For input data
        input_reso = self.m_input_reso[0]
        seq_len = int(tmp_seq_info.length // input_reso)
        s_idx = int(tmp_seq_info.start_pos // input_reso)
        e_idx = s_idx + seq_len

        # in case the input length not account using tmp_seq_info.seq_length
        if seq_len < 0:
            seq_len = 0
            s_idx = 0
            e_idx = 0

        input_dim = self.m_input_all_dim
        in_data = np.zeros([seq_len, 1], dtype=nii_dconf.h_dtype)
        s_dim = 0
        e_dim = 0

        # loop over each feature type
        for t_dir, t_ext, t_dim, t_res in \
                zip(self.m_input_dirs, self.m_input_exts, \
                    [self.m_input_dims[0]], self.m_input_reso):
            e_dim = s_dim + t_dim

            # get file path and load data
            file_path = file_name
            try:
                tmp_d = self.f_load_data(file_path, t_dim)
            except IOError:
                nii_warn.f_die("Cannot find %s" % (file_path))

            # write data
            if t_res < 0:
                # if this is for input data not aligned with output
                # make sure that the input is in shape (seq_len, dim)
                #  f_load_data should return data in shape (seq_len, dim)
                if tmp_d.ndim == 1:
                    in_data = np.expand_dims(tmp_d, axis=1)
                elif tmp_d.ndim == 2:
                    in_data = tmp_d
                else:
                    nii_warn.f_die("Default IO cannot handle %s" % (file_path))
            elif tmp_d.shape[0] == 1:
                # input data has only one frame, duplicate
                if tmp_d.ndim > 1:
                    in_data[:, s_dim:e_dim] = tmp_d[0, :]
                elif t_dim == 1:
                    in_data[:, s_dim] = tmp_d
                else:
                    nii_warn.f_die("Dimension wrong %s" % (file_path))
            else:
                # normal case
                if tmp_d.ndim > 1:
                    # write multi-dimension data
                    in_data[:, s_dim:e_dim] = tmp_d[s_idx:e_idx, :]
                elif t_dim == 1:
                    # write one-dimension data
                    in_data[:, s_dim] = tmp_d[s_idx:e_idx]
                else:
                    nii_warn.f_die("Dimension wrong %s" % (file_path))
            s_dim = e_dim

        # load output data
        if self.m_output_dirs:
            output_reso = self.m_output_reso[0]
            seq_len = int(tmp_seq_info.seq_length() // output_reso)
            s_idx = int(tmp_seq_info.seq_start_pos() // output_reso)
            e_idx = s_idx + seq_len

            out_dim = self.m_output_all_dim
            out_data = np.zeros([seq_len, out_dim], \
                                dtype=nii_dconf.h_dtype)
            s_dim = 0
            e_dim = 0
            for t_dir, t_ext, t_dim in zip(self.m_output_dirs, \
                                           self.m_output_exts, \
                                           self.m_output_dims):
                e_dim = s_dim + t_dim
                # get file path and load data
                file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
                try:
                    tmp_d = self.f_load_data(file_path, t_dim)
                except IOError:
                    nii_warn.f_die("Cannot find %s" % (file_path))

                if tmp_d.shape[0] == 1:
                    if tmp_d.ndim > 1:
                        out_data[:, s_dim:e_dim] = tmp_d[0, :]
                    elif t_dim == 1:
                        out_data[:, s_dim] = tmp_d
                    else:
                        nii_warn.f_die("Dimension wrong %s" % (file_path))
                else:
                    if tmp_d.ndim > 1:
                        out_data[:, s_dim:e_dim] = tmp_d[s_idx:e_idx, :]
                    elif t_dim == 1:
                        out_data[:, s_dim] = tmp_d[s_idx:e_idx]
                    else:
                        nii_warn.f_die("Dimension wrong %s" % (file_path))
                s_dim = s_dim + t_dim
        else:
            out_data = []
        in_data = self.pad(in_data)

        # post processing if necessary
        # in_data, out_data, tmp_seq_info, idx = self.f_post_data_process(
        #     in_data, out_data, tmp_seq_info, idx)
        # print('00000000000000000')

        # return data
        # assert isinstance(tmp_seq_info.length, object)
        return in_data
