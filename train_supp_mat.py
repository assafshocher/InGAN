import os
import threading
import queue as Queue
import subprocess

base_dir = './side/'
abl_args = {'geo_side': []}


def experiment_was_not_already_exec(exp_name):
    for nm in os.listdir('results'):
        if nm.startswith(exp_name):
            return False
    return True


class Worker(threading.Thread):
    def __init__(self, inQ, gpu_id):
        super(Worker, self).__init__()
        self.inQ = inQ
        self.daemon = True
        self.env = os.environ.copy()  # copy of environment
        self.env['CUDA_VISIBLE_DEVICES'] = '{:d}'.format(gpu_id)
        self.start()

    def run(self):
        while True:
            try:
                exp_name, item = self.inQ.get()
            except Queue.Empty:
                break
            # verify that this experiment was not executed already
            if experiment_was_not_already_exec(exp_name):
                subprocess.call(item, env=self.env)
            self.inQ.task_done()


def main():
    q = Queue.Queue()
    workers = [Worker(q, gpu_id) for gpu_id in [0, 1]]
    for imgname in os.listdir(base_dir):
        full_img_name = os.path.join(base_dir, imgname)
        short_name = os.path.splitext(imgname)[0]
        cmd = ['python', 'train.py', '--input_image_path', full_img_name, '--gpu_id', '0']
        for aname, aa in abl_args.items():
            exp_name = '{}_{}'.format(short_name, aname)
            full_cmd = cmd + aa + ['--name', exp_name]
            q.put((exp_name, full_cmd))
    q.join()


if __name__ == '__main__':
    main()
