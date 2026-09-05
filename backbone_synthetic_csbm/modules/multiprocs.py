import atexit
import os
import sys
import time

import numpy as np
import torch.multiprocessing as mp

from misc.utils import seed_everything


class ParentProcess:
    def __init__(self, args, Server, Client):
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0]
        # Fedrated's original logical orchestration is preserved, but CUDA may
        # not be initialized safely in a forked subprocess on modern PyTorch.
        # ``spawn`` changes only process construction, not client scheduling,
        # checkpoint persistence, broadcasts, or aggregation semantics.
        self.context = mp.get_context('spawn')
        self.manager = self.context.Manager()
        self.sd = self.manager.dict()
        self.sd['is_done'] = False
        self.processes = []
        self.queues = {}
        self.children_joined = False
        self.create_workers(Client)
        seed_everything(self.args.seed, cuda=True)
        self.server = Server(args, self.sd, self.gpu_server)
        atexit.register(self.done)

    def create_workers(self, Client):
        for worker_id in range(self.args.n_workers):
            gpu_id = self.gpus[worker_id + 1] if worker_id < len(self.gpus) - 1 else self.gpus[(worker_id - (len(self.gpus) - 1)) % len(self.gpus)]
            self.queues[worker_id] = self.context.Queue()
            process = self.context.Process(target=WorkerProcess, args=(self.args, worker_id, gpu_id, self.queues[worker_id], self.sd, Client))
            process.start()
            self.processes.append(process)

    def start(self):
        os.makedirs(self.args.checkpt_path, exist_ok=True)
        os.makedirs(self.args.log_path, exist_ok=True)
        self.n_connected = round(self.args.n_clients * self.args.frac)
        for curr_rnd in range(self.args.n_rnds):
            self.updated = set()
            np.random.seed(self.args.seed + curr_rnd)
            selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
            st = time.time()
            self.server.on_round_begin(curr_rnd)
            while selected:
                active = []
                for worker_id, queue in self.queues.items():
                    c_id = selected.pop(0)
                    active.append(c_id)
                    queue.put((c_id, curr_rnd))
                    if not selected:
                        break
                self.wait(active)
            self.server.on_round_complete(self.updated)
            print(f'[main] round {curr_rnd + 1} done ({time.time() - st:.2f} s)')
        self.sd['is_done'] = True
        for queue in self.queues.values():
            queue.put(None)
        self.done()
        self.report_result()
        self.cleanup_runtime_checkpoints()

    def wait(self, selected):
        while True:
            pending = False
            for c_id in selected:
                if c_id not in self.sd:
                    pending = True
                else:
                    self.updated.add(c_id)
            if not pending:
                return
            failed = [
                (index, process.exitcode)
                for index, process in enumerate(self.processes)
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed:
                details = ', '.join(
                    f'worker {index} exit={exitcode}'
                    for index, exitcode in failed
                )
                raise RuntimeError(
                    f'Fedrated worker exited before producing its update: {details}'
                )
            time.sleep(0.1)

    def report_result(self):
        result = self.server.finalize()
        values = ', '.join(
            f'{value:.4f}' for value in result['client_test_metrics']
        )
        print(f"[result] paired client {result['metric']}: [{values}]")
        print(
            f"[result] {result['metric']}={result['mean']:.6f} "
            f"± {result['std']:.6f}; "
            f"F1={result['f1_mean']:.6f} ± {result['f1_std']:.6f}"
        )
        print(f'[result] log_dir={self.args.log_path}')

    def cleanup_runtime_checkpoints(self):
        try:
            filenames = os.listdir(self.args.checkpt_path)
        except OSError:
            return
        removed = 0
        for name in filenames:
            if not name.endswith('_current_state.pt'):
                continue
            path = os.path.join(self.args.checkpt_path, name)
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        if removed:
            print(f'[main] removed {removed} runtime current checkpoint(s).')

    def done(self):
        if self.children_joined:
            return
        for process in self.processes:
            process.join()
        self.children_joined = True


class WorkerProcess:
    def __init__(self, args, worker_id, gpu_id, queue, sd, Client):
        self.queue = queue
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        seed_everything(self.args.seed + 10000 + worker_id, cuda=True)
        self.client = Client(self.args, self.worker_id, self.gpu_id, self.sd)
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            message = self.queue.get()
            if message is not None:
                client_id, curr_rnd = message
                seed_everything(self.args.seed + curr_rnd * 1000 + client_id, cuda=True)
                self.client.switch_state(client_id)
                self.client.on_receive_message(curr_rnd)
                self.client.on_round_begin()
                self.client.save_state()
            time.sleep(1.0)
        sys.exit()
