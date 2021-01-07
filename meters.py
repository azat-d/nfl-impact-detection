import datetime

import torch

from sklearn.metrics import average_precision_score

import slowfast.utils.logging as logging
from slowfast.utils.meters import ScalarMeter
import slowfast.utils.misc as misc
from fvcore.common.timer import Timer

logger = logging.get_logger(__name__)


class TrainValMeter(object):
    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_boxes = []
        self.all_labels = []
        self.overall_iters = overall_iters
        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_boxes = []
        self.all_labels = []

    def update_stats(self, preds, boxes, labels, loss=None, lr=None):
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            if boxes is not None:
                self.all_boxes.append(boxes)
            self.all_labels.append(labels)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        all_preds = torch.cat(self.all_preds, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)

        if len(self.all_boxes) > 0:
            all_boxes = torch.cat(self.all_boxes, dim=0)
            all_areas = (all_boxes[:, 3] - all_boxes[:, 1]) * (all_boxes[:, 4] - all_boxes[:, 2])
            mask = torch.ge(all_areas, self.cfg.NFL.MIN_BOX_AREA)
            all_preds = all_preds[mask].numpy().flatten()
            all_labels = all_labels[mask].numpy().flatten()
        else:
            all_preds = all_preds.numpy().flatten()
            all_labels = all_labels.numpy().flatten()

        self.score = average_precision_score(all_labels, all_preds)
        if log:
            stats = {"mode": self.mode, "score": self.score}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode == "val":
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "score": self.score,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats)
