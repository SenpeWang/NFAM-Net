# -*- coding: utf-8 -*-
"""Lightweight TensorBoard loggers.

TBLogger: base class wrapping SummaryWriter, extensible via inheritance.
DFRStats: nematic residual statistics accumulated per stage and flushed
at the end of each epoch.
"""


class TBLogger:
    """Lightweight TensorBoard logger base class.

    Wraps SummaryWriter and provides scalar writing plus step management.
    Inherit to extend with custom logging logic.
    """

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def scalar(self, tag, value):
        self.writer.add_scalar(tag, float(value), self.step)

    def set_step(self, step):
        self.step = step

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class DFRStats(TBLogger):
    """Nematic residual statistics logger (per stage).

    Only accumulates during forward; call flush_epoch() at epoch end to
    write to TensorBoard. Logged per stage:
        Q/{name}/S             — Q-tensor order parameter
        Q/{name}/gate          — gate mean
        Q/{name}/scale         — global scale
    """

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self._buf = {}

    def log_q_update(self, name, gate_mean, scale, s_after):
        """Accumulate a single Q-tensor update (not written immediately)."""
        buf = self._buf.setdefault(name, {'gate': [], 'scale': [], 'S': []})
        buf['gate'].append(gate_mean)
        buf['scale'].append(scale)
        buf['S'].append(s_after)

    def flush_epoch(self):
        """Write accumulated per-stage means to TB, then clear buffers."""
        for name, buf in self._buf.items():
            if buf['gate']:
                n = len(buf['gate'])
                self.scalar(f'Q/{name}/S',
                            sum(buf['S']) / n)
                self.scalar(f'Q/{name}/gate',
                            sum(buf['gate']) / n)
                self.scalar(f'Q/{name}/scale',
                            sum(buf['scale']) / n)
        self._buf.clear()
        self.flush()
