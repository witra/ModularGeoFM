import time
import threading
import torch
import pandas as pd
from lightning.pytorch.callbacks import Callback
from pynvml import *


class GPUMonitor(Callback):
    """
    Fully robust Lightning callback for GPU monitoring using nvidia-ml-py.
    Handles crashes, interrupts, EarlyStopping, and ensures safe shutdown.
    """

    def __init__(self, interval: float = 1.0, output_csv: str = "gpu_monitor_log.csv"):
        super().__init__()
        self.interval = interval
        self.output_csv = output_csv
        self.running = False
        self.thread = None
        self.data = []
        self.state = {"phase": "init", "epoch": -1, "step": -1}
        self.monitor_enabled = False
        self.trainer = None

    # -------------------------------------------------------------
    # Internal background loop
    # -------------------------------------------------------------
    def _monitor_loop(self):
        try:
            nvmlInit()
            n_gpus = torch.cuda.device_count()

            if n_gpus == 0:
                print("[GPUMonitor] No CUDA devices found — monitoring disabled.")
                return

            while self.running:
                timestamp = time.time()
                metrics = {}

                for i in range(n_gpus):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    util_gpu = util.gpu
                    util_gpu_mem = util.memory
                    mem_info = nvmlDeviceGetMemoryInfo(handle)#.used / (1024**2)
                    mem_used = mem_info.used / (1024**2)
                    mem_free = mem_info.free / (1024**2)
                    mem_total = mem_info.total / (1024**2)

                    self.data.append({
                        "time": timestamp,
                        "gpu-id": i,
                        "gpu_util_percent": util_gpu,
                        "mem_util_percent": util_gpu_mem,
                        "mem_util_MB": mem_used,
                        "mem_free_MB": mem_free,
                        "mem_total_MB": mem_total,
                        "phase": self.state["phase"],
                        "epoch": self.state["epoch"],
                        "step": self.state["step"],
                    })

                    metrics[f"gpu_{i}_util"] = util_gpu
                    metrics[f"gpu_{i}_mem"] = mem_used

                # Real-time logging to any Lightning logger (W&B, TensorBoard, CSVLogger, etc.)
                if self.trainer and self.trainer.logger:
                    self.trainer.logger.log_metrics(metrics, step=self.trainer.global_step)

                time.sleep(self.interval)

        except Exception as e:
            print(f"[GPUMonitor] Monitoring error: {e}")

        finally:
            # Always try to shut down NVML cleanly
            try:
                nvmlShutdown()
            except:
                pass

    # -------------------------------------------------------------
    # Clean shutdown (safe in all cases)
    # -------------------------------------------------------------
    def _stop_thread_and_save(self, crashed=False):
        if not self.monitor_enabled:
            return

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.data:
            fname = (self.output_csv if not crashed else self.output_csv.replace(".csv", "_crashed.csv"))
            df = pd.DataFrame(self.data)
            df.to_csv(fname, index=False)
            print(f"[GPUMonitor] GPU log saved to {fname}")
        else:
            print("[GPUMonitor] No GPU metrics to save.")

    # -------------------------------------------------------------
    # Lightning lifecycle hooks
    # -------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        self.trainer = trainer

        # Disable if not using CUDA
        if pl_module.device.type != "cuda":
            print("[GPUMonitor] Training is on CPU — GPU monitor disabled.")
            return

        print("[GPUMonitor] Starting GPU monitor thread using nvidia-ml-py.")

        self.monitor_enabled = True
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def on_fit_end(self, trainer, pl_module):
        print("[GPUMonitor] Training finished — stopping monitor.")
        self._stop_thread_and_save(crashed=False)

    def on_exception(self, trainer, pl_module, exception):
        print("[GPUMonitor] Training crashed — stopping monitor.")
        self._stop_thread_and_save(crashed=True)

    # -------------------------------------------------------------
    # Phase sync hooks
    # -------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.state.update({
            "phase": "train",
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.state.update({
            "phase": "train_idle",
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        })

    def on_validation_start(self, trainer, pl_module):
        self.state.update({
            "phase": "validation",
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        })

    def on_validation_end(self, trainer, pl_module):
        self.state.update({
            "phase": "validation_idle",
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        })
