''' Incremental-Classifier Learning 
 '''

from __future__ import print_function

import argparse
import ast
import importlib
import logging
import time

from sklearn.manifold import TSNE
from torch.autograd import Variable
from torchvision import transforms
from torch import Tensor
from models.wide_proto_resnet import Wide_ResNet
from models.proto_type import DceLoss

import torch
import torch.utils.data as td

import data_handler
import experiment as ex
import model
import plotter as plt
import trainer
from torchkeras import summary
import config
import core_scripts.other_tools.display as nii_warn
import core_scripts.data_io.default_data_io as nii_dset
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.config_parse.config_parse as nii_config_parse
import core_scripts.config_parse.arg_parse as nii_arg_parse
import core_scripts.op_manager.op_manager as nii_op_wrapper
import core_scripts.nn_manager.nn_manager as nii_nn_wrapper
import core_scripts.startup_config as nii_startup
import core_scripts.data_io.customize_collate_fn as nii_collate_fn
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as mplt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.x_vector_Indian_LID import X_vector
import core_scripts.data_io.seq_info as nii_seq_tk


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(num_repeats, axis=0)[0:max_len]
    # images[i] = img_buff[0:max_l]
    # padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = mplt.figure()
    ax = mplt.subplot(111)
    for i in range(data.shape[0]):
        mplt.text(data[i, 0], data[i, 1], str(label[i]),
                  color=mplt.cm.Set1(label[i] / 10.),
                  fontdict={'weight': 'bold', 'size': 9})
    mplt.xticks([])
    mplt.yticks([])
    mplt.title(title)
    return fig


logger = logging.getLogger('iCARL')

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 2.0). Note that lr is decyed by args.gamma parameter args.schedule ')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60, 70, 80],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--random-init', action='store_true', default=False,
                    help='Initialize model for next increment using previous weights if false and random weights otherwise')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss and only uses the cross entropy loss. See "Distilling Knowledge in Neural Networks" by Hinton et.al for details')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding algorithm and do random instance selection instead')
parser.add_argument('--seeds', type=int, nargs='+', default=[12345],
                    help='Seeds values to be used; seed introduces randomness by changing order of classes')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet44",
                    help='model type to be used. Example : resnet32, resnet20, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="./output",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--unstructured-size', type=int, default=0,
                    help='Leftover parameter of an unreported experiment; leave it at 0')
parser.add_argument('--alphas', type=float, nargs='+', default=[0.7],
                    help='Weight given to new classes vs old classes in the loss; high value of alpha will increase perfomance on new classes at the expense of older classes. Dynamic threshold moving makes the system more robust to changes in this parameter')
parser.add_argument('--decay', type=float, default=0.005, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=7, help='How many classes to add in each increment')
parser.add_argument('--T', type=float, default=3, help='Tempreture used for softening the targets')
parser.add_argument('--memory-budgets', type=int, nargs='+', default=[200],
                    help='How many images can we store at max. 0 will result in fine-tuning')

parser.add_argument('--epochs-class', type=int, default=50, help='Number of epochs for each increment')

parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--no-nl', action='store_true', default=False,
                    help='No Normal Loss. Only uses the distillation loss to train the new model on old classes ('
                         'Normal loss is used for new classes however')
parser.add_argument('--shuffle', action='store_false', default=True)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--input_dim', type=int, default=60)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--classes', type=int, default=13, help='最开始有多少类')
parser.add_argument('--feat_dim', type=int, default=4, help='每类原型个数')
parser.add_argument('--add_class', type=str,
                    default='{0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]}')
# {0: [0, 1, 2, 3, 4, 5, 6], 1: [7], 2: [8], 3: [9], 4: [10], 5: [11], 6: [12], 7: [13], 8: [14], 9: [15], 10: [16],
#  11: [17], 12: [18], 13: [19]}
# {0: [0, 1, 2, 3,4,5,6,7,8,9], 1: [10], 2: [11], 3: [12], 4: [13], '
#                             '5: [14], 6: [15], 7: [16], 8: [17]}
# 0: [0, 1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10], 2: [11, 12, 13, 14], 3: [15, 16, 17, 18, 19]
# 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# {0: [0, 1, 2, 3,4,5,6,7,8,9], 1: [10,11], 2:[12,13], 3: [14,15],4:[16,17]}
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.cuda.set_device(args.local_rank)
# dist.init_process_group(backend='nccl')
# dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

transforms = transforms.Compose([
    lambda x: pad(x),
    lambda x: Tensor(x)
])

# Checks to make sure parameters are sane
if args.step_size < 2:
    print("Step size of 1 will result in no learning;")
    assert False

# Run an experiment corresponding to every seed value
for seed in args.seeds:
    # Run an experiment corresponding to every alpha value
    for at in args.alphas:
        args.alpha = at
        # Run an experiment corresponding to every memory budget
        for m in args.memory_budgets:
            args.memory_budget = m
            # In LwF, memory_budget is 0 (See the paper "Learning without Forgetting" for details).
            if args.lwf:
                args.memory_budget = 0

            # Fix the seed.
            args.seed = seed
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            prj_conf = importlib.import_module('config')

            # Loader used for training data
            params = {'batch_size': args.batch_size,
                      'shuffle': args.shuffle,
                      'num_workers': args.num_workers}

            # Load file list and create data loader
            trn_lst = nii_list_tool.read_list_from_text(prj_conf.proto_train_dir)
            trn_set = nii_dset.NIIDataSetLoader(
                'train',
                args,
                transforms,
                prj_conf.trn_eval_set_name,
                trn_lst,
                prj_conf.input_dirs,
                prj_conf.proto_train_dir,
                prj_conf.input_exts,
                prj_conf.input_dims,
                prj_conf.input_reso,
                prj_conf.input_norm,
                prj_conf.output_dirs,
                prj_conf.output_exts,
                prj_conf.output_dims,
                prj_conf.output_reso,
                prj_conf.output_norm,
                './',
                truncate_seq=prj_conf.truncate_seq,
                min_seq_len=prj_conf.minimum_len,
                save_mean_std=True,
                wav_samp_rate=prj_conf.wav_samp_rate)

            val_lst = nii_list_tool.read_list_from_text(prj_conf.proto_dev_dir)
            val_set = nii_dset.NIIDataSetLoader(
                'dev',
                args,
                transforms,
                prj_conf.val_eval_set_name,
                val_lst,
                prj_conf.val_input_dirs,
                prj_conf.proto_dev_dir,
                prj_conf.input_exts,
                prj_conf.input_dims,
                prj_conf.input_reso,
                prj_conf.input_norm,
                prj_conf.output_dirs,
                prj_conf.output_exts,
                prj_conf.output_dims,
                prj_conf.output_reso,
                prj_conf.output_norm,
                './',

                truncate_seq=prj_conf.truncate_seq,
                min_seq_len=prj_conf.minimum_len,
                save_mean_std=False,
                wav_samp_rate=prj_conf.wav_samp_rate)

            nov_set = nii_dset.NIIDataSetLoader(
                'train',
                args,
                transforms,
                prj_conf.trn_eval_set_name,
                trn_lst,
                prj_conf.input_dirs,
                prj_conf.proto_train_dir,
                prj_conf.input_exts,
                prj_conf.input_dims,
                prj_conf.input_reso,
                prj_conf.input_norm,
                prj_conf.output_dirs,
                prj_conf.output_exts,
                prj_conf.output_dims,
                prj_conf.output_reso,
                prj_conf.output_norm,
                './',

                truncate_seq=prj_conf.truncate_seq,
                min_seq_len=prj_conf.minimum_len,
                save_mean_std=True,
                wav_samp_rate=prj_conf.wav_samp_rate)

            train_dataset_loader = data_handler.IncrementalLoader(trn_set,
                                                                  args.classes, [0, 1],
                                                                  cuda=args.cuda, oversampling=not args.upsampling,
                                                                  )
            # Loader for test data.
            test_dataset_loader = data_handler.IncrementalLoader(val_set,
                                                                 args.classes,
                                                                 [0, 1],
                                                                 cuda=args.cuda, oversampling=not args.upsampling,
                                                                 )
            novel_dataset_loader = data_handler.IncrementalLoader(nov_set,
                                                                  args.classes,
                                                                  [0, 1],
                                                                  cuda=args.cuda, oversampling=not args.upsampling,
                                                                  )
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_loader, drop_last=True)

            kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}
            collate_fn = nii_collate_fn.customize_collate

            # Iterator to iterate over training data.
            train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                                         **kwargs)
            # Iterator to iterate over test data
            test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=128, shuffle=True,
                                                        **kwargs)

            # Get the required model
            myModel = Wide_ResNet(28, 5, 0.3, args.classes, args.feat_dim)
            dce = DceLoss(args.classes, args.feat_dim)
            # myModel = X_vector(args.input_dim, args.classes)
            model_para = torch.load(
                '/home/hadoop/桌面/代码/中文伪造音频检测/incremental-learning-autoencoders_detect_novel-icral/model_save_中文增量_decay_0.005_seed_12345_class_14/model_0_epoch_4.pt')
            myModel.load_state_dict(model_para['net'], strict=False)
            dce.load_state_dict(model_para['dce'], strict=False)
            # dce.modefied_centers()
            # train_iterator.dataset.data = model_para['data']
            # train_iterator.dataset.indices = model_para['indices']
            # myModel.linear3 = torch.nn.Linear(50, args.classes + 1)
            # myModel.linear3.weight.data[:-1] = myModel.linear1.weight.data
            # myModel.linear1 = myModel.linear3
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    myModel = torch.nn.DataParallel(myModel)
                    myModel.cuda()
                else:
                    myModel.cuda()
                    dce.cuda()

            # Define an experiment.
            my_experiment = ex.experiment(args.name, args)

            # Adding support for logging. A .log is generated with all the logs. Logs are also stored in a temp file one directory
            # before the code repository
            logger = logging.getLogger('iCARL')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(my_experiment.path + ".log")
            fh.setLevel(logging.DEBUG)

            fh2 = logging.FileHandler("../temp.log")
            fh2.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            fh2.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(fh2)
            logger.addHandler(ch)

            # Define the optimizer used in the experiment
            optimizer = torch.optim.SGD([{'params': myModel.parameters()}, {'params': dce.parameters()}], args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.decay, nesterov=True)

            # Trainer object used for training
            my_trainer = trainer.Trainer(train_iterator, test_iterator, trn_set, myModel, dce, args, optimizer)

            # # load model
            # model_fro_fc = {}
            # model_para = torch.load('model_save_增量_16_3_decay_0.01_seed_12345/model_0_epoch_19.pt')
            # # for para in model_para['model']:
            # #     if para == 'module.fc.fc1.weight':
            # #         break
            # #     else:
            # #         model_fro_fc[para] = model_para['model'][para]
            # my_trainer.model.load_state_dict(model_para['model'])
            # my_trainer.model.eval()
            # with torch.no_grad():
            #     for idx, (data, target) in enumerate(test_iterator):
            #         data = data.cuda()
            #         dim_out, _ = myModel(Variable(data))
            #         tsne = TSNE(n_components=2, init='pca', random_state=0)
            #         t0 = time.time()
            #         result = tsne.fit_transform(dim_out.detach().cpu().numpy().reshape(args.batch_size, 128 * 8 * 51))
            #         fig = plot_embedding(result, target.numpy(),
            #                              't-SNE embedding of the digits (time %.2fs)'
            #                              % (time.time() - t0))
            #         mplt.show(fig)
            # Parameters for storing the results
            x = []
            y = []
            y1 = []
            train_y = []
            higher_y = []
            y_scaled = []
            y_grad_scaled = []
            nmc_ideal_cum = []
            model_dir = 'model_save_中文增量_decay_{}_seed_{}_class_{}'.format(args.decay, seed, args.classes + 1)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            # Initilize the evaluators used to measure the performance of the system.
            t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args)

            # Loop that incrementally adds more and more classes
            add_class = ast.literal_eval(args.add_class)
            logger.info(add_class)
            cur_all_classes = 0
            cur_len = 0
            for class_group in range(len(add_class.keys())):
                logger.info('SEED: {} MEMORY_BUDGET: {}  CLASS_GROUP: {}'.format(seed, m, class_group))
                cur_len += len(add_class[class_group])
                # Add new classes to the train, train_nmc, and test iterator
                my_trainer.increment_classes(class_group, add_class)
                my_trainer.update_frozen_model()
                my_trainer.updata_criterion(class_group, add_class)
                # epoch = 0
                # epochs = 1
                # if class_group == 0:
                #     epochs = 1
                # else:
                #     epochs = args.epochs_class
                cur_all_classes += len(add_class[class_group])

                # Running epochs_class epochs
                # for epoch in range(0, args.epochs_class):
                #     my_trainer.update_lr(epoch)
                #     start_time = time.time()
                #     # my_trainer.train(epoch, class_group, add_class)
                #     if epoch % args.log_interval == (args.log_interval - 1):
                #         # tError = t_classifier.evaluate(my_trainer.model, train_iterator, my_trainer.dce)
                #         # logger.debug("*********CURRENT EPOCH********** : %d", epoch)
                #         # logger.debug("Train Classifier: %0.2f", tError)
                #         # logger.debug("Test Classifier: %0.2f",
                #         #              t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.dce))
                #         # my_trainer.detect_novel(novel_dataset_loader, add_class, class_group)
                #
                #         # 保存模型
                #         my_trainer.setup_training(cur_len)
                #         save_mode_path = os.path.join(model_dir, 'model_{}_epoch_{}.pt'.format(class_group, epoch))
                #         save_parameter = {
                #             'net': my_trainer.model.state_dict(),
                #             'dce': my_trainer.dce.state_dict(),
                #             'data': train_dataset_loader.data,
                #             'indices': train_dataset_loader.indices
                #         }
                #         # # save_mode_path_sum = os.path.join(model_dir, 'model_epoch_sum_{}.pt'.format(epoch))
                #         torch.save(save_parameter, save_mode_path)
                #         # f_sum = {'f_sum_max_all': my_trainer.f_sum_max_all,
                #         #          'f_sum_sec_all': my_trainer.f_sum_sec_all,
                #         #          'nb_correct': my_trainer.nb_correct
                #         #          }
                #         # torch.save(f_sum, save_mode_path_sum)
                #         print('{}.pt have saved!'.format(save_mode_path))
                #     end_time = time.time()
                #     logger.info('epoch {} cost time: {}'.format(epoch, end_time - start_time))
                #
                # Evaluate the learned classifier
                # my_trainer.setup_training(cur_len)
                my_experiment.store_json()
                # Finally, plotting the results;
                my_plotter = plt.Plotter()
                tcMatrix = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, cur_all_classes,
                                                             my_trainer.dce)
                # Plotting the confusion matrices
                my_plotter.plotMatrix(class_group, my_experiment.path + "tcMatrix", tcMatrix)
