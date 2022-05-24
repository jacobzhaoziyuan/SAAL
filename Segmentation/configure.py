import argparse

data_path_train = '../data/GrayData'
result_path = '../result'

def get_arguments():

    parser = argparse.ArgumentParser(description='PyTorch UNet Training')

    # Model
    parser.add_argument('--num_classes', type=int, default=3,
                        help='output channel of network')
    parser.add_argument('--weight_path', type=str, default=None, help='pre-trained weights')
    parser.add_argument('--pre_train_type', type=str, default=None, help='load pre-trained weights or not')
    
    # load clustering results
    parser.add_argument('--cluster_npy_path', type=str, default=None, help='cluster npy path')
    parser.add_argument('--cluster_idx_path', type=str, default=None, help='cluster idx path')
    
    # Datasets
    parser.add_argument("--data-path-train", type=str, default=data_path_train,
                        help="Path to the images.")
    parser.add_argument("--img_size", type=list, default=[256,256])
    parser.add_argument("--select_type", default='random',type=str,help='random/cluster')
    parser.add_argument("--select_num", default = 100, type=int, help= 'number of selected samples')
    parser.add_argument("--train_num", default = 1600, type=int, help='number of samples for train from training dataset')
    parser.add_argument("--result_path", type=str, default=result_path,
                        help="Path to the results.")
    parser.add_argument("--load_idx", type=str, default=None,
                        help="Path to the sample index.")

    # Optimization options
    parser.add_argument('--batch-size', type=int,  default=8, help='batch size')
    parser.add_argument('--num-epochs', type=int,  default=50, help='maximum epoch number to train')
    parser.add_argument('--Adam', action='store_true')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument('--lr', type=float,  default=0.001, help='maximum epoch number to train')
    parser.add_argument("--change-optim", action="store_true")
    parser.add_argument('-Tmax', '--lr-rampdown-epochs', default=200, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--eta-min', default=0., type=float)
    
    # loss function options
    parser.add_argument('--loss', default='ce', type=str, help='ce/dice/ce_dice' )
    
    # trainig round
    parser.add_argument('--iteration_num', default=0, type=int, help='iteration number')
    parser.add_argument('--active_epochs', default=2, type=int, help='active epochs')
    parser.add_argument('--aug_samples', default=35, type=int, help='augmentation samples')
    parser.add_argument('--round', default=10, type=int, help='training several rounds for average')

    # Miscs
    parser.add_argument('--manual-seed', type=int, default=1111, help='random seed')
    parser.add_argument('--gpu', type=str, default='2', help='GPU to use')

    return parser.parse_args()
