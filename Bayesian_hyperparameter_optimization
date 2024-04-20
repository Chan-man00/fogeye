import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image
import cv2
from utils_stereofog import calculate_model_results
import numpy as np
from skopt import BayesSearchCV
#from skopt import hp
from bayes_opt import BayesianOptimization, UtilityFunction
import sklearn
# SK Learn imports
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# Pandas import
import pandas as pd


# Define the Internal method for optimization
# Function to be optimized
def train_network(lr, batch_size, beta1,  n_layers_D, ngf, ndf, dropout_rate,  pool_size): # lr_policy, norm, netG, netD, init_type,gan_mode, n_epochs, 
    # Convert continuous variables to appropriate format
    batch_size = int(batch_size)
    
    #dataroot = ".\\datasets\\stereofog_images\\04-04-24_augmented"
    dataroot = ".\\datasets\\stereofog_images\\4-15-24-hdr"
    n_layers_D = int(n_layers_D)
    ngf = int(ngf)
    ndf = int(ndf)
    pool_size = int(pool_size)
    lr = float(lr)
    beta1 = float(beta1)
    dropout_rate = float(dropout_rate)
    norm = 'batch'
    lr_policy = 'linear'
    gan_mode = 'vanilla'
    init_type = 'normal'
    netG = 'resnet_9blocks'
    netD = 'n_layers'
    n_epochs = 100
    n_epochs = int(n_epochs)

        # --results_path .\\results\\optimizing\\optimizing_results 
    # Command to run the Python script
    command = f"python train.py --dataroot {dataroot} --no_html --name optimizing --batch_size {batch_size} --model pix2pix --direction BtoA --lr {lr} --n_epochs {n_epochs} --n_epochs_decay 0 --beta1 {beta1} --init_type {init_type} --gan_mode {gan_mode} --netD n_layers --n_layers_D {n_layers_D} --ngf {ngf} --ndf {ndf} --lr_policy {lr_policy} --norm {norm} --netD {netD} --netG {netG} --pool_size {pool_size} --dropout_rate {round(dropout_rate,4)}"
    
    # Execute the command and capture output
    
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True, timeout=None)
    '''with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        proc.wait()
        stdout, stderr = proc.communicate() '''

    # Extract the metric you want to optimize. Assume it's printed as 'Final loss: X.XX'
    # This is just a placeholder. You must modify your training script to output the necessary metric.
    output = result.stdout
    
    #print(output)
    final_loss = float(output.split('G_L1\': ')[-1][:8].strip())

    # Minimizing the loss (hence negative as BayesOpt maximizes the target by default)
    return -final_loss


# bayes_opt requires this to be a dictionary.
bds = {'dropout_rate': (0.1, 0.9),
		'ngf': (16, 96),
        'ndf': (16, 128),
		'n_layers_D': (1, 6),
        'beta1': (.1, 1.0),
        'lr': (0.0000001, .0005),
        'pool_size': (1, 200),
        'batch_size': (1, 100),
		}


"""'netG': ('resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128'),
		'netD': ("basic", "n_layers", "pixel"),
         '--norm': ("batch", "instance", "none"),
        'lr_policy': ('linear', 'step', 'plateau', 'cosine'),
        'gan_mode': ('vanilla', 'lsgan', 'wgangp'),
        'n_epochs': (100),
        'lr_decay_iters': (10, 100),
		'init_type': ('normal', 'xavier', 'kaiming', 'orthogonal')"""

# Create a BayesianOptimization optimizer and optimize the function
optimizer = BayesianOptimization(f = train_network,
                                 pbounds = bds,
                                 random_state = 7)


optimizer.maximize(init_points = 5, n_iter = 25)
best_score = optimizer.max
print(best_score)


'''learning_rate_space = hp.uniform('learning_rate', 0.01, 1)
num_hidden_units_space = hp.quniform('num_hidden_units', 10, 100, 1)
search_space = [learning_rate_space, num_hidden_units_space]
optimizer = BayesSearchCV(
    estimator = create_model(),
    search_spaces: list,
    optimizer_kwargs: Any | None = None,
    n_iter: int = 50,
    scoring: Any | None = None,
    fit_params: Any | None = None,
    n_jobs: int = 1,
    n_points: int = 1,
    refit: bool = True,
    cv: Any | None = None,
    verbose: int = 0,
    pre_dispatch: str = '2*n_jobs',
    random_state: Any | None = None,
    error_score: str = 'raise',
    return_train_score: bool = False
)'''


############################## Better Way to do it ##########################
'''def main(args):
    start_params = args.start_params
    redis_port = args.redis_port

    params_queue = RedisQueue('params-' + args.task, port=redis_port)
    params_queue.clear()
    results_queue = RedisQueue('results-' + args.task, port=redis_port)
    results_queue.clear()
    hyper_parameters = args.hyper_parameters
    print('hyper_parameters:', hyper_parameters)
    hp_db = redis.Redis(port=redis_port)
    hp_db.set('hp-{}-{}'.format(args.task, args.size), hyper_parameters)
    print('start_params:', start_params)
    print('hyper_parameters key:', 'hp-{}-{}'.format(args.task, args.size))
    print('params queue:', 'params-' + args.task)
    print('results queue:', 'results-' + args.task)
    sys.stdout.flush()

    def train_and_eval(kd_alpha, kd_mse_beta, ce_gama):
        kd_alpha, kd_mse_beta, ce_gama = round(kd_alpha, 1), round(kd_mse_beta, 1), round(ce_gama, 1)
        params_queue.put('{},{},{}'.format(kd_alpha, kd_mse_beta, ce_gama))
        print("waiting results for a={},b={},c={}...".format(kd_alpha, kd_mse_beta, ce_gama))
        sys.stdout.flush()
        eval_result = results_queue.get(block=True)
        print("eval_result: {} for a={},b={},c={}".format(eval_result, kd_alpha, kd_mse_beta, ce_gama))
        sys.stdout.flush()
        return round(float(eval_result), 2)

    # Bounded region of parameter space
    param_space = {'kd_alpha': (0.1, 2.0), 'kd_mse_beta': (0.1, 2.0), 'ce_gama': (0.1, 2.0)}

    optimizer = BayesianOptimization(
        f=train_and_eval,
        pbounds=param_space,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={'kd_alpha': start_params[0], 'kd_mse_beta': start_params[1], 'ce_gama': start_params[2]},
        # params={'kd_alpha': 0.5, 'kd_mse_beta': 0.2, 'ce_gama': 0.3},  # s9
        # params={'kd_alpha': 0.3, 'kd_mse_beta': 0.2, 'ce_gama': 0.8}, #s10
        lazy=False,
    )
    logger = JSONLogger(path=args.progress_log)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
    )
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)
'''



# def calculate_model_results(results_path = results_path):

#     results_path = os.path.join(results_path, 'test_latest/images')

#     # CW-SSIM implementation
#     gaussian_kernel_sigma = 1.5
#     gaussian_kernel_width = 11
#     gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

#     # Indexing the images
#     images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

#     SSIM_scores = []
#     CW_SSIM_scores = []
#     Pearson_image_correlations = []

#     for i, image in enumerate(images):


#         clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
#         fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
#         fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

#         # Calculating the Pearson correlation coefficient between the two images (https://stackoverflow.com/questions/34762661/percentage-difference-between-two-images-in-python-using-correlation-coefficient, https://mbrow20.github.io/mvbrow20.github.io/PearsonCorrelationPixelAnalysis.html)
#         # clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
#         # Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
#         # corrImAbs = np.absolute(Pearson_image_correlation)

#         # Pearson_image_correlations.append(np.mean(corrImAbs))

#         # Calculating the SSIM between the fake image and the clear image
#         (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)

#         SSIM_scores.append(SSIM_score_reconstruction)

#         # Calculating the CW-SSIM between the fake image and the clear image (https://github.com/jterrace/pyssim)
#         CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))

#         CW_SSIM_scores.append(CW_SSIM)

#         # Calculate the average values

#         mean_SSIM = np.mean(SSIM_scores)
#         mean_CW_SSIM = np.mean(CW_SSIM_scores)

#         return mean_SSIM, mean_CW_SSIM

'''mean_SSIM, mean_CW_SSIM = calculate_model_results(results_path)

print(f"Mean SSIM: {mean_SSIM:.2f}")
print(f"Mean CW-SSIM: {mean_CW_SSIM:.2f}")'''
