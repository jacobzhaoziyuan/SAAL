import os
import shutil
from datetime import datetime
import pytz

class models_genesis_config:
    model = "Unet2D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes" # not use
    scale = 32
    input_rows = 256
    input_cols = 256
    input_deps = 1
    nb_class = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_dir = "../SSLModel/Reuslts/pretrained_weights"
    timenow = datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), '%Y-%m-%d_%H-%M-%S')
    model_path = os.path.join(model_dir,timenow)
    print('Model path: ',model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    logs_path = os.path.join(model_path, "Logs")
    print('log path: ',logs_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        
    shotdir = os.path.join(model_path, 'snapshot')
    print('snapshot path: ',shotdir)
    if not os.path.exists(shotdir):
        os.makedirs(shotdir)
    
    # model pre-training
    verbose = 1
    weights = os.path.join(model_path,'ISIC_Unsup.pt')
    batch_size = 1
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 0.01
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
