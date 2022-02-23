# C:\Users\90543\AppData\Local\torch_extensions\torch_extensions
import torch
import data
import models
import optimizers
from options import TrainOptions
from util import IterationCounter
from util import Visualizer
from util import MetricTracker
from evaluation import GroupEvaluator
import subprocess
import platform
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["num_workers"] = "0"
def init_vsvars():
    vswhere_path = r"%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
    vswhere_path = os.path.expandvars(vswhere_path)
    if not os.path.exists(vswhere_path):
        raise EnvironmentError("vswhere.exe not found at: %s", vswhere_path)

    vs_path = os.popen('"{}" -latest -property installationPath'.format(vswhere_path)).read().rstrip()
    vsvars_path = os.path.join(vs_path, "VC\\Auxiliary\\Build\\vcvars64.bat")

    output = os.popen('"{}" && set'.format(vsvars_path)).read()

    for line in output.splitlines():
        pair = line.split("=", 1)
        if(len(pair) >= 2):
            os.environ[pair[0]] = pair[1]


if __name__ == '__main__': 
    if "windows" in platform.system().lower():
        init_vsvars()
        os.system("where cl.exe")
    opt = TrainOptions().parse(command='--lr 0.002 --evaluation_freq 5 --dataroot "D:/HISTORIFY/Models_Trials/my_dataset/historified" --dataset_mode imagefolder --num_gpus 1 --batch_size 2 --preprocess scale_shortside_and_crop --load_size 256 --crop_size 256 --display_freq 1600 --print_freq 480 --name church_pretrained --patch_use_aggregation False --continue_train True --evaluation_metrics swap_visualization  --checkpoints_dir "D:/HISTORIFY/Models_Trials/swapping-autoencoder-pytorch/checkpoints/" --total_nimgs 1 --display_port 2004')
    #print(opt)
           
    dataset = data.create_dataset(opt)
    tmp = opt.dataroot
    opt.dataset = dataset

    base_dir, _ = os.path.split(opt.dataroot)
    content_dir = os.path.join(base_dir,'content')
    opt.dataroot = content_dir
    opt.shuffle_dataset = False
    content_dataset = data.create_dataset(opt)
    #content_dataset.dataloader.dataset.A_paths
    content_imgs = []
    test_dir = os.path.join(base_dir, 'test_set')
    #test_dir_length = len(os.listdir(test_dir))
    opt.dataroot = test_dir
    test_set = data.create_dataset(opt)    
    opt.dataset = dataset
    opt.shuffle_dataset = True
    opt.dataroot=tmp
    iter_counter = IterationCounter(opt)
    visualizer = Visualizer(opt)
    metric_tracker = MetricTracker(opt)
    evaluators = GroupEvaluator(opt)

    model = models.create_model(opt)
    optimizer = optimizers.create_optimizer(opt, model)
    for c in range(content_dataset.length):
        content_imgs.append(next(content_dataset))
    
    while not iter_counter.completed_training():
        with iter_counter.time_measurement("data"):
            cur_data = next(dataset)
            name = os.path.split(cur_data['path_A'][1])[1]
            first_ = name.find('_')
            second_ = name.find('_',first_ + 1)
            for x in content_imgs:
                #print(x['path_A'][0],x['path_A'][1])
                #print(name[first_+1:second_])
                if os.path.split(x['path_A'][0])[1] == "image_"+name[first_+1:second_]+".png":
                    cur_data['path_A'][0] = x['path_A'][0]
                    cur_data['real_A'][0] = x['real_A'][0]
                    #import numpy
                    #import torch
                    #import cv2
                    #img1 = cv2.imread(cur_data['path_A'][0])
                    #img2 = cv2.imread(cur_data['path_A'][1])
                    #cv2.imshow(cur_data['path_A'][0]+" and "+cur_data['path_A'][1],numpy.vstack((img1,img2)))
                    #cv2.waitKey(0)
                    break
                elif os.path.split(x['path_A'][1])[1] == "image_"+name[first_+1:second_]+".png":
                    #cur_data['path_A'][0] = x['path_A'][1]
                    #cur_data['real_A'][0] = x['real_A'][1]
                    #import numpy
                    #import torch
                    #import cv2
                    #img1 = cv2.imread(cur_data['path_A'][0])
                    #img2 = cv2.imread(cur_data['path_A'][1])
                    #cv2.imshow(cur_data['path_A'][0]+" and "+cur_data['path_A'][1],numpy.vstack((img1,img2)))
                    #cv2.waitKey(0)
                    break
            if iter_counter.needs_displaying():
                print("snapshot visuals",cur_data['path_A'][0],cur_data['path_A'][1])
                visuals = optimizer.get_visuals_for_snapshot(cur_data)
                visualizer.display_current_results(visuals,
                                                   iter_counter.steps_so_far)
            metrics = evaluators.evaluate(                # bu kısmı hallet datasetin içindeki veriler yine aynı olmalı
            model, test_set, iter_counter.steps_so_far)
            metric_tracker.update_metrics(metrics, smoothe=False)

        with iter_counter.time_measurement("maintenance"):
            if iter_counter.needs_printing():
                visualizer.print_current_losses(iter_counter.steps_so_far,
                                                iter_counter.time_measurements,
                                                metric_tracker.current_metrics())

            if iter_counter.needs_displaying():
                print("snapshot visuals",cur_data['path_A'][0],cur_data['path_A'][1])
                visuals = optimizer.get_visuals_for_snapshot(cur_data)
                visualizer.display_current_results(visuals,
                                                   iter_counter.steps_so_far)

            if iter_counter.needs_evaluation():
                metrics = evaluators.evaluate(                # bu kısmı hallet datasetin içindeki veriler yine aynı olmalı
                    model, test_set, iter_counter.steps_so_far)
                metric_tracker.update_metrics(metrics, smoothe=False)

            if iter_counter.needs_saving():
                optimizer.save(iter_counter.steps_so_far)

            if iter_counter.completed_training():
                break

            iter_counter.record_one_iteration()
        
    optimizer.save(iter_counter.steps_so_far)
    print('Training finished.')
