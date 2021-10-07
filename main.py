import argparse
import os
import logging
import time
import math

from datetime import datetime
from functools import partial
from multiprocessing import Pool, log_to_stderr

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                             'to have subdirectories named "train" and "val"; alternatively,\n' +
                             'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--process', default=4, type=int, metavar='N',
                        help='number of data loading processes (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


@pipeline_def
def create_dali_pipeline(data_dir, num_shards, shard_id):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=True,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu'
    decoder_device = 'cpu'
    device_memory_padding = 0
    host_memory_padding = 0
    crop = 224
    images = fn.decoders.image_random_crop(images,
                                           device=decoder_device, output_type=types.RGB,
                                           device_memory_padding=device_memory_padding,
                                           host_memory_padding=host_memory_padding,
                                           random_aspect_ratio=[0.8, 1.25],
                                           random_area=[0.1, 1.0],
                                           num_attempts=100)
    images = fn.resize(images,
                       device=dali_device,
                       resize_x=crop,
                       resize_y=crop,
                       interp_type=types.INTERP_TRIANGULAR)
    mirror = fn.random.coin_flip(probability=0.5)

    images = fn.crop_mirror_normalize(images,
                                      device=dali_device,
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=mirror)
    return images, labels


def main():
    global args
    args = parse()

    if not len(args.data):
        raise Exception("error: No data set provided")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

    args.world_size = 1
    if args.distributed:
        print('Inits distributed process group with gloo backend')
        # Distributed information will be passed in through environment variable WORLD_SIZE and RANK
        torch.distributed.init_process_group(backend='gloo',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if len(args.data) == 1:
        train_dir = os.path.join(args.data[0], 'train')
    else:
        train_dir = args.data[0]

    master_addr = "N/A"
    master_port = "N/A"
    if 'MASTER_ADDR' in os.environ:
        master_addr = os.environ['MASTER_ADDR']
    if 'MASTER_PORT' in os.environ:
        master_port = os.environ['MASTER_PORT']

    # Use multiple processes to mock real machine learning training
    num_shards = args.world_size * args.process
    shard_id = range(rank * args.process, (rank + 1) * args.process)

    print('Launching training script: train_dir[{}], world_size[{}], rank[{}], batch_size[{}], processes[{}], '
          'num_shards[{}], current_shard_id[{}], master_addr[{}], master_port[{}]'
          .format(train_dir, args.world_size, rank, args.batch_size, args.process, num_shards,
                  shard_id, master_addr, master_port))

    log_to_stderr(logging.DEBUG)
    pool = Pool(processes=args.process)
    dali_func = partial(dali, args.batch_size, train_dir, args.print_freq, num_shards)

    results = pool.map(dali_func, shard_id)
    total_time = 0.0
    image_per_second = 0.0
    for result in results:
        total_time += result[0]
        image_per_second += result[1]

    # TODO(lu) add a socket to receive the img/sec from all nodes in the cluster
    print("Training end: Average speed: {:3f} img/sec, Total time: {:3f} sec"
          .format(image_per_second, total_time))


def dali(batch_size, train_dir, print_freq, num_shards, shard_id):
    print('Launching training script in child process: train_dir[{}], batch_size[{}], print_freq[{}], '
          'num_shards[{}], current_shard_id[{}], starting at[{}]'
          .format(train_dir, batch_size, print_freq, num_shards, shard_id, datetime.now().time()))
    pipe = create_dali_pipeline(batch_size=batch_size,
                                num_threads=1,
                                seed=12 + shard_id,
                                data_dir=train_dir,
                                num_shards=num_shards,
                                shard_id=shard_id,
                                device_id=-99999)
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # train for one epoch
    total_duration, throughput = train(train_loader, batch_size, print_freq, shard_id)
    train_loader.reset()
    return total_duration, throughput


def train(train_loader, batch_size, print_freq, shard_id):
    batch_time = AverageMeter()
    train_loader_len = int(math.ceil(train_loader._size / batch_size))
    start = time.time()
    end = time.time()

    count = 0
    for i, data in enumerate(train_loader):
        count = count + 1
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        if i % print_freq == 0:
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if shard_id == 0:
                print('[[{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {2:.3f} ({3:.3f})\t'.format(
                    i, train_loader_len,
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time))
        # Use the time.sleep to replace the actual training logics
        time.sleep(0.5)

    total_time = time.time() - start
    throughput = train_loader._size / total_time
    print('Training in child process ends: Speed: {} image/s, Data Size: {} images, End Time: {}'
          .format(throughput, train_loader._size, datetime.now().time()))
    return total_time, throughput


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
