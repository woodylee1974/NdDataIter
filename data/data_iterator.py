import numpy as np
import multiprocessing as mp
from multiprocessing.queues import Empty


class RandomLoader:
    def __init__(self, data_src, rng):
        self.data_src = data_src
        self.shuffle = list(range(len(self.data_src)))
        rng.shuffle(self.shuffle)

    def __getitem__(self, item):
        return self.shuffle[item]

    def __len__(self):
        return len(self.shuffle)


class SequentialLoader:
    def __init__(self, data_src):
        self.data_src = data_src
        self.sequential_list = list(range(len(self.data_src)))

    def __getitem__(self, item):
        return self.sequential_list[item]

    def __len__(self):
        return len(self.sequential_list)


class BatchLoader:
    def __init__(self, loader, batch_size):
        self.data_loader = loader
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_loader) // self.batch_size
        for i in range(n):
            yield self.data_loader[i * self.batch_size: (i+1) * self.batch_size]

    def __len__(self):
        return len(self.data_loader) // self.batch_size


class SimpleDataIterator:
    def __init__(self, owner):
        self.owner = owner

    def build_batch_data(self, batch_job):
        var_list = []
        for idx in batch_job:
            var_list.append(self.owner.data_src[idx])
        return self.owner.splice(var_list)

    def __next__(self):
        batch_job = next(self.owner.loader, None)
        if batch_job is None:
            if self.owner.config["auto_reset"]:
                self.owner.reset_loader()
                batch_job = next(self.owner.loader, None)
        if batch_job is None:
            raise StopIteration
        batch = self.build_batch_data(batch_job)
        return batch

    def __iter__(self):
        return self


class MultiProcessDataIterator:
    def __init__(self, owner, preprocess=None):
        self.owner = owner
        self.batch_data_buffer = {}
        self.data_queue = mp.Queue()
        self.worker_list = []
        self.job_queue_list = []
        self.finish_event = mp.Event()
        self.timeout = self.owner.config['timeout']
        self.in_turn_worker = 0
        self.in_progress_job = 0
        self.current_job_id = 0
        self.job_id = 0
        self.stopped = False

        for i in range(self.owner.worker_num):
            job_queue = mp.Queue()
            job_queue.cancel_join_thread()
            worker = mp.Process(
                target=self.worker_loop,
                args=(self.owner.data_src, job_queue, self.data_queue,
                      self.owner.splice, self.finish_event, self.timeout, preprocess)
            )
            worker.daemon = True
            worker.start()
            self.worker_list.append(worker)
            self.job_queue_list.append(job_queue)

        for _ in self.worker_list:
            self.assign_new_job()

    @staticmethod
    def worker_loop(data_src, job_queue, data_queue, splice, finish_event, timeout, preprocess):
        data_queue.cancel_join_thread()
        while True:
            try:
                job_id, job = job_queue.get(timeout=timeout)
            except Empty:
                continue
            if job is None:
                return
            elif finish_event.is_set():
                continue

            batch_data = splice([data_src[idx] for idx in job])
            if callable(preprocess):
                batch_data = preprocess(batch_data)
            data_queue.put((job_id, batch_data))
            del batch_data

    def __next__(self):
        if self.current_job_id in self.batch_data_buffer:
            batch_data = self.batch_data_buffer[self.current_job_id]
            self.assign_new_job()
            self.current_job_id += 1
            return batch_data

        if self.in_progress_job == 0:
            self.stop_all_workers()
            raise StopIteration

        while True:
            try:
                idx, batch_data = self.data_queue.get(timeout=self.timeout)
                self.in_progress_job -= 1
            except Exception as e:
                if not all(worker.is_alive() for worker in self.worker_list):
                    raise RuntimeError('Some worker unexpectedly exited.')
                if isinstance(e, Empty):
                    raise RuntimeError("Queue is empty until timeout.")
                raise

            if idx != self.current_job_id:
                self.batch_data_buffer[idx] = batch_data
                continue

            self.assign_new_job()
            self.current_job_id += 1
            return batch_data

    def assign_new_job(self):
        batch_job = next(self.owner.loader, None)
        if batch_job is None:
            if self.owner.config["auto_reset"]:
                self.owner.reset_loader()
                batch_job = next(self.owner.loader, None)
        if batch_job is None:
            return
        self.job_queue_list[self.in_turn_worker].put((self.job_id, batch_job))
        self.in_turn_worker += 1
        if self.in_turn_worker >= self.owner.worker_num:
            self.in_turn_worker = 0
        self.job_id += 1
        self.in_progress_job += 1

    def stop_all_workers(self):
        if not self.stopped:
            self.stopped = True
            self.finish_event.set()
            for job_queue in self.job_queue_list:
                job_queue.put((self.job_id,  None))
                job_queue.close()

            for worker in self.worker_list:
                worker.join()

    def __del__(self):
        if self.owner.worker_num > 0:
            self.stop_all_workers()


class DataIterator:
    def __init__(self, data_src, preprocess=None, **config):
        default_config = {
            "shuffle": False,
            "batch_size": 1,
            "workers": 0,
            "seed": None,
            "loader": None,
            "auto_reset": False,
            "timeout": 10
        }
        self.config = default_config
        self.config.update(config)
        self.data_src = data_src
        self.preprocess = preprocess
        self.rng = np.random.RandomState(self.config["seed"])
        self.reset_loader()
        self.worker_num = min(mp.cpu_count(), self.config["workers"])

    def reset_loader(self):
        if self.config["loader"] is None:
            loader = RandomLoader(self.data_src, self.rng) if self.config["shuffle"] else \
                SequentialLoader(self.data_src)
            self.loader = iter(BatchLoader(loader, self.config["batch_size"]))
        else:
            self.loader = iter(self.config["loader"])

    def __iter__(self):
        if self.worker_num == 0:
            return SimpleDataIterator(self)
        else:
            return MultiProcessDataIterator(self, self.preprocess)

    @staticmethod
    def splice(variable_list):
        cols = [[] for _ in range(len(variable_list[0]))]
        for row in variable_list:
            for idx, col in enumerate(list(row)):
                cols[idx].append(col)
        return tuple(np.stack(col, axis=0) for col in cols)



if __name__ == '__main__':
    from data_source import NdArrayDataSource
    a = np.arange(1000)
    b = np.zeros((1000, 3))
    c = np.arange(1000) * 100

    for i in range(1000):
        b[i, 0] = i

    ds = NdArrayDataSource([a, b, c])
    for x, y, z in DataIterator(ds, batch_size=3, workers=100, shuffle=True):
        print(y)


