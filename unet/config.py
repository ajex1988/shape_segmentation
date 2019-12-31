

tfrecord_file_train = '/mnt/sdc/ShapeTexture/simulation_data/1219/train.tfrecord'
tfrecord_file_val = '/mnt/sdc/ShapeTexture/simulation_data/1219/val.tfrecord'
max_steps = 5000
cuda_device_id = '0'

resize_height = 512
resize_width = 512

model_dir = '/mnt/sdc/ShapeTexture/models/1219'
keep_checkpoint_max = 5
save_checkpoints_secs = 100
log_step_count_steps = 50