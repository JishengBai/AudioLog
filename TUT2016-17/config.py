# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# The configuration for training the model

train_path = '~/train'
val_path = '~/val'
test_path = '~/test_example'
exp_name = "htsat_pretrain.pth" # the saved ckpt prefix name of the model 
exp_path = '~/exp'

loss_type = "clip_ce" # 

# trained from a checkpoint, or evaluate a single model 
resume_checkpoint = "./HTSAT_AudioSet_Saved_6.ckpt"

debug = False

random_seed = 970131 # 19970318 970131 12412 127777 1009 34047
batch_size = 32  # batch size per GPU x GPU number , default is 32 x 4 = 128
learning_rate = 1e-4 # 1e-4 also workable 
max_epoch = 200
early_stop_epochs = 50



# for model's design
enable_tscam = True # enbale the token-semantic layer

# for signal processing
sample_rate = 32000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 10 # audio_set 10-sec clip
window_size = 1024
hop_size = 320 # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)


# for data collection
classes_num = 25 # esc: 50 | audioset: 527 | scv2: 35
asc_classes_num = 5
patch_size = (25, 4) # deprecated
crop_size = None # int(clip_samples * 0.5) deprecated

posterior_thresh = 0.5
labels_dict = {"(object) banging": 0,
                    "(object) impact": 1,
                    "(object) rustling": 2,
                    "(object) snapping": 3,
                    "(object) squeaking": 4,
                    "bird singing": 5,
                    "brakes squeaking": 6,
                    "breathing": 7,
                    "car": 8,
                    "car passing by": 8,
                    "children": 9,
                    "children shouting": 9, 
                    "cupboard": 10,
                    "cutlery": 11,
                    "dishes": 12,
                    "drawer": 13,
                    "fan": 14,
                    "glass jingling": 15,
                    "keyboard typing": 16,
                    "large vehicle": 17,
                    "mouse clicking": 18,
                    "mouse wheeling": 19,
                    "object impact": 1,
                    "people speaking": 20,
                    "people talking": 20,
                    "people walking": 21,
                    "washing dishes": 22,
                    "water tap running ": 23,
                    "wind blowing": 24
                    }
labels_hard = ['(object) banging', '(object) impact', '(object) rustling', '(object) snapping',
               '(object) squeaking', 'bird singing', 'brakes squeaking', 'breathing', 'car', 
                'children',  'cupboard', 'cutlery', 'dishes',
               'drawer', 'fan', 'glass jingling', 'keyboard typing', 'large vehicle', 'mouse clicking',
               'mouse wheeling',  'people talking', 'people walking',
               'washing dishes', 'water tap running', 'wind blowing']

asc_labels_dict = {"city_center": 0,
                    "home": 1,
                    "office": 2,
                    "residential_area": 3,
                    }
asc_labels = ['city_center', 'home', 'office', 
            'residential_area']









# these data preparation optimizations do not bring many improvements, so deprecated
enable_token_label = False # token label
class_map_path = "class_hier_map.npy"
class_filter = None 
retrieval_index = [15382, 9202, 130, 17618, 17157, 17516, 16356, 6165, 13992, 9238, 5550, 5733, 1914, 1600, 3450, 13735, 11108, 3762, 
    9840, 11318, 8131, 4429, 16748, 4992, 16783, 12691, 4945, 8779, 2805, 9418, 2797, 14357, 5603, 212, 3852, 12666, 1338, 10269, 2388, 8260, 4293, 14454, 7677, 11253, 5060, 14938, 8840, 4542, 2627, 16336, 8992, 15496, 11140, 446, 6126, 10691, 8624, 10127, 9068, 16710, 10155, 14358, 7567, 5695, 2354, 8057, 17635, 133, 16183, 14535, 7248, 4560, 14429, 2463, 10773, 113, 2462, 9223, 4929, 14274, 4716, 17307, 4617, 2132, 11083, 1039, 1403, 9621, 13936, 2229, 2875, 17840, 9359, 13311, 9790, 13288, 4750, 17052, 8260, 14900]
token_label_range = [0.2,0.6]
enable_time_shift = False # shift time
enable_label_enhance = False # enhance hierarchical label
enable_repeat_mode = False # repeat the spectrogram / reshape the spectrogram




# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]

swin_pretrain_path = None
# "/home/Research/model_backup/pretrain/swin_tiny_c24_patch4_window8_256.pth"

# Some Deprecated Optimization in the model design, check the model code for details
htsat_attn_heatmap = False
htsat_hier_output = False 
htsat_use_max = False


# for ensemble test 

ensemble_checkpoints = []
ensemble_strides = []


# weight average folder
wa_folder = "/home/version_0/checkpoints/"
# weight average output filename
wa_model_path = "HTSAT_AudioSet_Saved_x.ckpt"

esm_model_pathes = [
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_1.ckpt",
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_2.ckpt",
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_3.ckpt",
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_4.ckpt",
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_5.ckpt",
    "/home/Research/model_backup/AudioSet/HTSAT_AudioSet_Saved_6.ckpt"
]

# for framewise localization
heatmap_dir = "/home/Research/heatmap_output"
test_file = "htsat-test-ensemble"
fl_local = False # indicate if we need to use this dataset for the framewise detection
fl_dataset = "/home/Research/desed/desed_eval.npy"  
fl_class_num = [
    "Speech", "Frying", "Dishes", "Running_water",
    "Blender", "Electric_shaver_toothbrush", "Alarm_bell_ringing",
    "Cat", "Dog", "Vacuum_cleaner"
]

# map 527 classes into 10 classes
fl_audioset_mapping = [
    [0,1,2,3,4,5,6,7],
    [366, 367, 368],
    [364],
    [288, 289, 290, 291, 292, 293, 294, 295, 296, 297],
    [369],
    [382],
    [310, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402],
    [81, 82, 83, 84, 85],
    [74, 75, 76, 77, 78, 79],
    [377]
]
