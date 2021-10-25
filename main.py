import argparse
import os
import logging
import time
import math

from datetime import datetime
from functools import partial
from multiprocessing import Pool

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

    parser = argparse.ArgumentParser(description='PyTorch ImageNet DALI Data Loading')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                             'to have subdirectories named "train" and "val"; alternatively,\n' +
                             'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--process', default=4, type=int, metavar='N',
                        help='number of data loading processes (default: 4) in each node')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size per process (default: 256)')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency (default: 100)')
    args = parser.parse_args()
    return args


@pipeline_def
# create a CPU only dali pipelieg for reading data from the data source
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

def dali(batch_size, train_dir, print_freq, logger, num_shards, shard_id):
    logger.info('Launching training script in child process: train_dir[{}], batch_size[{}], print_freq[{}], '
                'num_shards[{}], current_shard_id[{}], starting at[{}]'
                .format(train_dir, batch_size, print_freq, num_shards, shard_id, datetime.now().time()))
    start = time.time()
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
    total_size, total_read_time, throughput = train(train_loader, batch_size, print_freq, logger, shard_id)
    total_time = time.time() - start
    logger.info('Training in child process ends: data: {} images, total time: {} sec, total read time: {} sec, '
                'data loading speed: {:3f} img/sec'
                .format(total_size, total_time, total_read_time, throughput))
    train_loader.reset()
    return total_size, total_read_time, throughput, total_time


def train(train_loader, batch_size, print_freq, logger, shard_id):
    train_loader_len = int(math.ceil(train_loader._size / batch_size))
    start = time.time()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze(-1).long()
        if i % print_freq == 0:
            if shard_id == 0:
                logger.info('[[}/{}] Time: {}'.format(
                    i, train_loader_len, time.time() - end))
        # use the sleep to mock actual training logic
        time.sleep(0.5)
        end = time.time()

    total_time = time.time() - start
    throughput = train_loader._size / total_time
    return train_loader._size, total_time, throughput


def main():
    global args
    args = parse()

    if not len(args.data):
        raise Exception("error: No data set provided")

    # set cluster logging
    # TODO(lu) does the log shows twice the info needed?
    logger = logging.getLogger('dev')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

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

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # arena doesn't support the pytorch multi-processing
    # rank is the node rank instead of the process rank
    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

    args.world_size = 1
    if args.distributed:
        logger.info('Launching distributed test with gloo backend')
        # distributed information will be passed in through environment variable WORLD_SIZE and RANK
        torch.distributed.init_process_group(backend='gloo',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logger.info('Launched distributed test in {} nodes'.format(args.world_size))

    # mock the actual machine learning process
    # each process in each node read a portion of the whole dataset
    num_shards = args.world_size * args.process
    shard_id = range(rank * args.process, (rank + 1) * args.process)

    logger.info('Launching training script: train_dir[{}], world_size[{}], master_addr[{}], master_port[{}], '
                'batch_size[{}], rank[{}], processes[{}], num_shards[{}], current_shard_id[{}]'
                .format(train_dir, args.world_size, master_addr, master_port, args.batch_size,
                        rank, args.process, num_shards, shard_id))

    pool = Pool(processes=args.process)
    dali_func = partial(dali, args.batch_size, train_dir, args.print_freq, logger, num_shards)

    for epoch in range(0, args.epochs):
        results = pool.map(dali_func, shard_id)
        total_size = 0.0
        total_read_time = 0.0
        image_per_second = 0.0
        total_time = 0.0
        for result in results:
            total_size += result[0]
            total_read_time += result[1]
            image_per_second += result[2]
            total_time += result[3]
        # TODO(lu) add a socket to receive the img/sec from all nodes in the cluster
        logger.info("Epoch {} training end: process average data: {} images, process average total time: {} sec, "
                    "process average read time: {} sec, "
                    "process data loading speed: {:3f} img/sec, process number: {}, "
                    "node data loading speed: {:3f} img/sec"
                    .format(epoch, total_size / args.process, total_time / args.process, total_read_time / args.process, 
                            image_per_second / args.process, args.process, image_per_second))
        # clear buffer cache requires special docker privileges
        # as a workaround, we clear the buffer cache manually
        # TODO(lu) enable setting docker privileges in arena to support clear buffer cache in script
        logger.info("Starts sleeping to give time for clearing system buffer cache manually")
        time.sleep(300)


if __name__ == '__main__':
    main()
