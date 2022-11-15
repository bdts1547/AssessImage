from argparse import Namespace

# Config TRACER
def getConfig():
   
    parser = Namespace()
    parser.actiont='inference'
    parser.exp_num='0'
    parser.dataset='upload/'
    parser.data_path='data/'

    # Model parameter settings
    parser.arch='5'
    parser.channels = [24, 40, 112, 320]
    parser.RFB_aggregated_channel=[32, 64, 128]
    parser.frequency_radius = 16
    parser.denoise = 0.93
    parser.gamma = 0.1

    # Training parameter settings
    parser.img_size = 512
    parser.batch_size = 1
    parser.epochs = 100
    parser.lr = 5e-5
    parser.optimizer = 'Adam'
    parser.weight_decay = 1e-4
    parser.criterion = 'API'
    parser.scheduler = 'Reduce'
    parser.aug_ver = 2
    parser.lr_factor = 0.1
    parser.clipping = 2.0
    parser.patience = 5
    parser.model_path = 'results/'
    parser.seed = 42
    parser.save_map = True


    # Hardware settings
    parser.multi_gpu = False
    parser.num_workers = 4


    # cfg = parser.parse_args()

    return parser


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)