import time
import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from data.process_videos import avi2pngs

opt = TrainOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1


opt.continue_train = True
opt.dataroot = opt.trainroot
opt.file_list = opt.train_file_list
avi2pngs(opt)
opt.dataroot = opt.dataroot + '/split'
training_data_loader = CreateDataLoader(opt)
train_dataset = training_data_loader.load_data()
train_dataset_size = len(training_data_loader)
print('#training samples = %d' % train_dataset_size)


opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = opt.testroot
opt.file_list = opt.test_file_list
avi2pngs(opt)
opt.dataroot = opt.dataroot + '/split'
testing_data_loader = CreateDataLoader(opt)
test_dataset = testing_data_loader.load_data()
test_dataset_size = len(testing_data_loader)
print('#testing samples = %d' % test_dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

total_steps = 0
opt.batchSize = 32
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(train_dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if total_steps % opt.display_freq == 0 and opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/train_dataset_size, opt, errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()

opt.batchSize = 1
for i, data in enumerate(test_dataset):
    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()

    print('process image... %s' % img_path)
    visualizer.save_seqs(webpage, visuals, img_path)

webpage.save()
