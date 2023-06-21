import os.path as osp
import numpy as np

from src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from src.tf_utils import reset_tf_graph
from src.general_utils import plot_3d_point_cloud


top_out_dir = './data/'
top_in_dir = './data/uniform_samples_2048/'

experiment_name = 'single_class_ae'
n_pc_points = 2048
bneck_size = 16
ae_loss = 'chamfer'
class_name = raw_input('Give me the dataset name (e.g. "test"):').lower()

syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
print(class_dir)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

train_params = default_train_params()

encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))

                      
load_pre_trained_ae = True
restore_epoch = 1000
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)

feed_pc, feed_model_names, _ = all_pc_data.full_epoch_data(shuffle=False)
reconstructions = ae.reconstruct(feed_pc)[0]
latent_codes = ae.transform(feed_pc)
np.savetxt('codes.csv', latent_codes, fmt='%0.8f', delimiter=',')
np.savetxt('names.csv', feed_model_names, fmt='%s', delimiter=',')

print(feed_model_names[0])
i = 0
plot_3d_point_cloud(reconstructions[i][:, 0],
                    reconstructions[i][:, 1],
                    reconstructions[i][:, 2], in_u_sphere=True);

