import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img,GradCAMlocal,GradCAMmta,GradCAM2d,GradCAMske
from modeling.models import TriGait_baseline
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
import os
import cv2
from torchvision import models
import argparse,pickle,numpy
import torchvision.transforms as T
from data.transform import get_transform
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def readpu(paths):
    data_list = []

    for file_name in os.listdir(paths):
        pth = os.path.join(paths, file_name)
        if pth.endswith('.pkl'):
            with open(pth, 'rb') as f:
                _ = pickle.load(f)
            f.close()
        else:
            raise ValueError('- Loader - just support .pkl !!!')
        data_list.append(_)




    double_index = np.arange(0, data_list[0].shape[0], 2)


    #data_list[0] = data_list[0][double_index, :]
    m = min(data_list[0].shape[0], data_list[1].shape[0])
    data_list[0] = data_list[0][0:m, :]
    data_list[1] = data_list[1][0:m,:, :]
    return data_list

class ResizeTransform:
    def __init__(self):
        a = 10

    def __call__(self, x):
        result = x.unsqueeze(-1)
        print(result.shape)
        result = result.repeat(1,1,1,1,44)

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]

        return result

def main(cfgs,Training):


    model = TriGait_baseline(cfgs,Training)
    model.eval()

    model.eval()
    with torch.no_grad():
        for i, (image_batch, label_batch) in enumerate(testloader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            label_batch = label_batch.long().squeeze()
            inputs = image_batch
            logits, feature = model(inputs)
            if i == 0:
                feature_bank = feature
                label_bank = label_batch
                logits_bank = logits
            else:
                feature_bank = torch.cat((feature_bank, feature))
                label_bank = torch.cat((label_bank, label_batch))
                logits_bank = torch.cat((logits_bank, logits))

    target_category = None


    # load image
    img_path = "/data/GaitData/CASIA-B-sil-pose/010/nm-01/144/"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

    inputs= readpu(img_path)

    trf_cfgs = cfgs['trainer_cfg']['transform']
    seq_trfs = get_transform(trf_cfgs)


    ipts = [np2var(np.asarray(seq_trfs[0](inputs[1])),requires_grad=False).float().unsqueeze(0),np2var(np.asarray(seq_trfs[1](inputs[0])), requires_grad=False).float().unsqueeze(0)]




    target_layers = [model.ES.ConvA3[1], model.aggre_t.out_conv,model.attention_t.dilate_conv_1,model.attention_t.dilate_conv_2,model.attention_t.dilate_conv_4]
    cam = GradCAMmta(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=ipts, target_category=target_category)
    # print(grayscale_cam[0,0,10:20,10:20])

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam[0].shape)
    sil = ipts[0][0, :]
    map = []
    i = grayscale_cam.shape[0]
    for i in range(i):
        map.append(grayscale_cam[i,:])
    map = np.concatenate(map, axis=-1)


    visualization = show_cam(map,use_rgb=True)
    plt.imshow(visualization)

    print("===========save_mta============")

    plt.savefig('mta.jpg')

    plt.close()





    target_layers = [model.tcnst2[-1].st.atten]  # 定义目标层
    cam = GradCAMske(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=ipts, target_category=target_category)
    # print(grayscale_cam[0,0,10:20,10:20])

    grayscale_cam = grayscale_cam[0, :]
    sil = ipts[0][0, :]


    map = []
    i = grayscale_cam.shape[0]
    for i in range(i):
        map.append(grayscale_cam[i, :])
    map = np.concatenate(map, axis=-1)

    visualization = show_cam(map, use_rgb=True)
    plt.imshow(visualization)

    output_directory = "/code/CodeTest/fxl/Opengait/heatmap/"

    print("===========saveske============")

    plt.savefig('ske.jpg')

    plt.close()



    target_layers = [model.fuse]  # 定义目标层
    cam = GradCAMfuse(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=ipts, target_category=target_category)
    # print(grayscale_cam[0,0,10:20,10:20])

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)
    sil = ipts[0][0, :]
    visualization = []
    visualization=show_cam(grayscale_cam,use_rgb=True)
    plt.imshow(visualization)

    output_directory = "/code/CodeTest/fxl/Opengait/heatmap/"

    print("===========savefuse============")

    plt.savefig('fuse.jpg')

    plt.show()
    
    




parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='/config/trigait/trigait_gait3d.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))

    cfgs = config_loader(opt.cfgs)

    cfgs['evaluator_cfg']['restore_hint'] = 50000
    cfgs['trainer_cfg']['restore_hint'] = 50000

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    main(cfgs,training)