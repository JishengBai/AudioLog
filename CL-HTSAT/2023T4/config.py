# The configuration for AudioLog

train_path = '~/train'
val_path = '~/eval'
test_path = '~/test_example'
exp_name = "./htsat_pretrain.pth" # the saved ckpt prefix name of the model 
exp_path = '~/exp'

loss_type = "clip_ce" # 

# trained from a checkpoint, or evaluate a single model 
resume_checkpoint = "./HTSAT_AudioSet_Saved_6.ckpt"

debug = False

random_seed = 970131 # 19970318 970131 12412 127777 1009 34047
batch_size = 30  # batch size per GPU x GPU number , default is 32 x 4 = 128
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
fmax = 16000
shift_max = int(clip_samples * 0.5)


# for data collection
classes_num = 11 # esc: 50 | audioset: 527 | scv2: 35
asc_classes_num = 5
patch_size = (25, 4) # deprecated
crop_size = None # int(clip_samples * 0.5) deprecated

posterior_thresh = 0.5

labels_dict = {'birds_singing': 0,
'car': 1,
'people talking': 2,
'footsteps': 3,
'children voices': 4,
'wind_blowing': 5,
'brakes_squeaking': 6,
'large_vehicle': 7,
'cutlery and dishes': 8,
'metro approaching': 9,
'metro leaving': 10,
                    }
labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 
               'children voices', 'wind_blowing', 'brakes_squeaking', 
               'large_vehicle', 'cutlery and dishes', 'metro approaching', 
               'metro leaving']

asc_labels_dict = {'cafe_restaurant':0, 'city_center':1, 'grocery_store':2, 'metro_station':3,
            'residential_area':4
                    }
asc_labels = ['cafe_restaurant', 'city_center', 'grocery_store', 'metro_station',
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
