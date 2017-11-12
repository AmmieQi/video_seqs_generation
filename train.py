import time
import os
import random
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from data.process_videos import avi2pngs
from pdb import set_trace as st
from util import html

opt = TrainOptions().parse()

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

avi2pngs(opt)
if opt.file_list != None:
     opt.dataroot = opt.dataroot + '/split'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training samples = %d' % dataset_size)

train_batch_size = opt.batchSize



model = create_model(opt)
visualizer = Visualizer(opt)

opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = opt.testroot
opt.file_list = opt.test_file_list
opt.seq_len = opt.test_seq_len
opt.pre_len = opt.test_pre_len
opt.seq_stride = opt.test_seq_stride
avi2pngs(opt)
if opt.file_list != None:
     opt.dataroot = opt.dataroot + '/split'
testing_data_loader = CreateDataLoader(opt)
test_dataset = testing_data_loader.load_data()
test_dataset_size = len(testing_data_loader)
print('#testing samples = %d' % test_dataset_size)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


total_steps = 0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += train_batch_size
        epoch_iter += train_batch_size
        model.set_input(data)
        model.optimize_parameters()	
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / train_batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if total_steps % opt.display_freq == 0 and opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        #if total_steps % opt.save_latest_freq == 0:
        #    print('saving the latest model (epoch %d, total_steps %d)' %
        #          (epoch, total_steps))
        #    model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()

    if epoch % 50 == 0:
        print('#testing samples = %d' % test_dataset_size)        
        for i, data in enumerate(test_dataset):
            model.set_input(data)
            model.test()

            visuals = model.get_all_current_visuals()
            img_path = model.get_image_paths()
            
            print('process image... %s' % img_path)
            visualizer.save_seqs(webpage, visuals, img_path)


print('#testing samples = %d' % test_dataset_size)        
for i, data in enumerate(test_dataset):
    model.set_input(data)
    model.test()

    visuals = model.get_all_current_visuals()
    img_path = model.get_image_paths()
            
    print('process image... %s' % img_path)
    visualizer.save_seqs(webpage, visuals, img_path)