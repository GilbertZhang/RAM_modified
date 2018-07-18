start_step = 0

# to enable visualization, set draw to True
eval_only = False
draw = True
animate = False

no_glp = -1

# conditions
translateMnist = 1
MixedMnist = False
clutteredMnist = False
translateMnist_scale = 28
eyeCentered = 0

# pretrain
preTraining = 0
preTraining_epoch = 20000
drawReconsturction = 0

# about translation
MNIST_SIZE = 28
translated_img_size = 60  # side length of the picture

if translateMnist:
    print("TRANSLATED MNIST")
    img_size = translated_img_size
    depth = 3  # number of zooms
    sensorBandwidth = 12
    minRadius = 8  # zooms -> minRadius * 2**<depth_level>

    initLr = 1e-3
    lr_min = 1e-4
    lrDecayRate = .999
    lrDecayFreq = 200
    momentumValue = .9
    batch_size = 64

else:
    print("CENTERED MNIST")
    img_size = MNIST_SIZE
    depth = 1  # number of zooms
    sensorBandwidth = 8
    minRadius = 4  # zooms -> minRadius * 2**<depth_level>

    initLr = 1e-3
    lrDecayRate = .99
    lrDecayFreq = 200
    momentumValue = .9
    batch_size = 20


# model parameters
channels = 1                # mnist are grayscale images
totalSensorBandwidth = depth * channels * (sensorBandwidth **2)
nGlimpses = 6               # number of glimpses
loc_sd = 0.22               # std when setting the location

# network units
hg_size = 128               #
hl_size = 128               #
g_size = 256                #
cell_size = 256             #
cell_out_size = cell_size   #

# paramters about the training examples
n_classes = 10              # card(Y)

# training parameters
max_iters = 1000000
SMALL_NUM = 1e-10

# resource prellocation
mean_locs = []              # expectation of locations
sampled_locs = []           # sampled locations ~N(mean_locs[.], loc_sd)
baselines = []              # baseline, the value prediction
glimpse_images = []         # to show in window