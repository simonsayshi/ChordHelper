import subprocess
import csv
import io
import logging

logger = logging.getLogger(__name__)


class GPUMonitor:
    def __init__(self):
        """
        Checks if nvidia-smi is available.
        """
        self.available = False
        try:
            subprocess.check_output(["nvidia-smi", "-L"])
            self.available = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("nvidia-smi not found. GPU monitoring disabled.")

    def get_stats(self):
        """
        Returns a list of dicts (one per GPU) with current stats.
        """
        if not self.available:
            return []

        try:
            # Query specific metrics in CSV format
            # nounits removes 'MiB', '%' etc so we get pure numbers
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.check_output(cmd).decode("utf-8")

            stats = []
            reader = csv.reader(io.StringIO(result))
            for row in reader:
                # row: [index, util, temp, mem_used, mem_total, power]
                try:
                    stats.append(
                        {
                            "id": int(row[0]),
                            "util_pct": float(row[1]),
                            "temp_c": float(row[2]),
                            "mem_used_gb": float(row[3]) / 1024,
                            "mem_total_gb": float(row[4])/ 1024,
                            "power_w": float(row[5]),
                        }
                    )
                except ValueError:
                    logger.warning(f"Failed to parse nvidia-smi output row: {row}")
                    continue
            return stats

        except Exception as e:
            # Don't crash training if monitoring fails
            return []
