from __future__ import print_function

from configs import Config
from InGAN import InGAN
import os
from util import Visualizer, read_data
from traceback import print_exc


# Load configuration
conf = Config().parse()

# Prepare data
input_images = read_data(conf)

# Create complete model
gan = InGAN(conf)

# If required, fine-tune from some checkpoint
if conf.resume is not None:
    gan.resume(os.path.join(conf.resume))

# Define visualizer to monitor learning process
visualizer = Visualizer(gan, conf, input_images)

# Main training loop
for i in range(conf.max_iters + 1):

    # Train a single iteration on the current data instance
    try:
        gan.train_one_iter(i, input_images)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While training.' % i)
        print_exc()

    # Take care of all testing, saving and presenting of current results and status
    try:
        visualizer.test_and_display(i)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While testing or visualizing.' % i)
        print_exc()

    # Save snapshot when needed
    try:
        if i > 0 and not i % conf.save_snapshot_freq:
            gan.save(os.path.join(conf.output_dir_path, 'checkpoint_%07d.pth.tar' % i))
            del gan
            gan = InGAN(conf)
            gan.resume(os.path.join(conf.output_dir_path, 'checkpoint_%07d.pth.tar' % i))
            visualizer.gan = gan
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print('Something went wrong in iteration %d, While saving snapshot.' % i)
        print_exc()
