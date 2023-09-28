import csv
from collections import OrderedDict
from typing import Dict, Mapping, Union

import distributed as dist


Scalar = Union[int, float]


class CSVLogger:
    """
    CSV file logger. CSV headers are step, and names passed to `log`.

    Args:
        path (str): path to write logs to
    """

    def __init__(
        self,
        path: str,
        # dump_every_n_steps: int = 100,
    ) -> None:
        self.path = path
        self._log_buffer: OrderedDict[int, Dict[str, float]] = OrderedDict()
        self._main_process = dist.is_main_process()

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Log scalar data to file.

        Args:
            name (string): a unique name to group scalars
            data (float/int): scalar data to log
            step (int): step value to record
        """
        if self._main_process:
            self._log_buffer.setdefault(step, {})[name] = float(data)
            self._log_buffer[step]["step"] = step

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Add multiple scalar values.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int): step value to record
        """
        for k, v in payload.items():
            self.log(k, v, step)

    def flush(self) -> None:
        data = self._log_buffer
        if not data:
            return

        if self._main_process:
            with open(self.path, "a") as f:
                data_list = list(data.values())
                w = csv.DictWriter(f, data_list[0].keys())
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(data_list)

        self._log_buffer.clear()

    def close(self) -> None:
        self.flush()
