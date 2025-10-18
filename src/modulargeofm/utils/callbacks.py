# import time
# import threading
# import torch
# import pandas as pd
# from pytorch_lightning import Callback

# class LightningGPUMonitor(Callback):
#     """
#     Lightning callback that monitors GPU utilization and memory in the background,
#     synchronized with training/validation events.
#     CPU-safe and aware of the actual training device.
#     """
#     def __init__(self, interval: float = 1.0, output_csv: str = "gpu_monitor_log.csv"):
#         """
#         Args:
#             interval: Polling interval in seconds.
#             output_csv: Path to save the log CSV.
#         """
#         super().__init__()
#         self.interval = interval
#         self.output_csv = output_csv
#         self.running = False
#         self.thread = None
#         self.data = []
#         self.state = {"phase": "init", "epoch": -1, "step": -1}
#         self.monitor_enabled = False

#     # -------------------------------------------------------------
#     # Internal monitoring loop
#     # -------------------------------------------------------------
#     def _monitor_loop(self):
#         try:
#             import pynvml
#             pynvml.nvmlInit()
#             n_gpus = torch.cuda.device_count()
#             if n_gpus == 0:
#                 print("[GPUMonitor] No CUDA devices found — monitoring disabled.")
#                 return

#             while self.running:
#                 timestamp = time.time()
#                 for i in range(n_gpus):
#                     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#                     util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
#                     mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
#                     self.data.append({
#                         "time": timestamp,
#                         "gpu": i,
#                         "util_percent": util,
#                         "mem_MB": mem,
#                         "phase": self.state["phase"],
#                         "epoch": self.state["epoch"],
#                         "step": self.state["step"]
#                     })
#                 time.sleep(self.interval)
#             pynvml.nvmlShutdown()
#         except Exception as e:
#             print(f"[GPUMonitor] Disabled due to error: {e}")

#     # -------------------------------------------------------------
#     # Lightning lifecycle hooks
#     # -------------------------------------------------------------
#     def on_fit_start(self, trainer, pl_module):
#         # Check if the training device is CUDA
#         device_type = pl_module.device.type
#         if device_type != "cuda":
#             print(f"[GPUMonitor] Training is on {device_type} — GPU monitoring disabled.")
#             self.monitor_enabled = False
#             return

#         # Enable monitoring
#         self.monitor_enabled = True
#         self.running = True
#         self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
#         self.thread.start()
#         print(f"[GPUMonitor] Started background GPU monitoring every {self.interval}s.")

#     def on_fit_end(self, trainer, pl_module):
#         if not self.monitor_enabled:
#             return

#         self.running = False
#         if self.thread is not None:
#             self.thread.join()

#         if self.data:
#             df = pd.DataFrame(self.data)
#             df.to_csv(self.output_csv, index=False)
#             print(f"[GPUMonitor] GPU utilization log saved to {self.output_csv}")
#         else:
#             print("[GPUMonitor] No GPU data collected (CPU-only run or error).")

#     # -------------------------------------------------------------
#     # Event hooks for synchronization
#     # -------------------------------------------------------------
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         self.state.update({
#             "phase": "train",
#             "epoch": trainer.current_epoch,
#             "step": trainer.global_step,
#         })

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.state.update({
#             "phase": "train_idle",
#             "epoch": trainer.current_epoch,
#             "step": trainer.global_step,
#         })

#     def on_validation_start(self, trainer, pl_module):
#         self.state.update({
#             "phase": "validation",
#             "epoch": trainer.current_epoch,
#             "step": trainer.global_step,
#         })

#     def on_validation_end(self, trainer, pl_module):
#         self.state.update({
#             "phase": "validation_idle",
#             "epoch": trainer.current_epoch,
#             "step": trainer.global_step,
#         })
