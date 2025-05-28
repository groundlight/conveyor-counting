import time
import logging

MAX_LOGGING_PERIOD_SEC = 1.0

logger = logging.getLogger(__name__)

class PerfTimer:
    def __init__(self, name: str, debug_only: bool = True):
        self._debug_only = debug_only
        if self._logger_active():
            self._name = name
            self._last_logged_time = 0.0
            
        if debug_only:
            self.log = logger.debug
        else:
            self.log = logger.info
            
    def start(self) -> None:
        if self._logger_active():
            self._start_time = time.perf_counter()
        
        
    def stop(self) -> None:
        if not self._logger_active():
            return
        
        stop_time = time.perf_counter()
        if stop_time - self._last_logged_time > MAX_LOGGING_PERIOD_SEC:
            self._last_logged_time = stop_time
            elapsed_time = stop_time - self._start_time
            self.log(f'{self._name}: {elapsed_time:.2f} second(s)')
            
    def _logger_active(self) -> bool:
        if self._debug_only and not logger.isEnabledFor(logging.DEBUG):
            return False
        else:
            return True
        
class LoopManager:
    def __init__(self, loop_name: str, loop_time: float) -> None:
        self._loop_name = loop_name
        self._loop_time = loop_time
        self._target_fps = 1.0 / loop_time if loop_time > 0 else float('inf')
        
    def start(self) -> None:
        self._start_time = time.perf_counter()
        
    def wait(self) -> bool:
        stop_time = time.perf_counter()
        elapsed_time = stop_time - self._start_time
        actual_fps = 1.0 / elapsed_time if elapsed_time > 0 else float('inf')
        
        remaining_time_to_wait = self._loop_time - elapsed_time

        if remaining_time_to_wait > 0.0:
            time.sleep(remaining_time_to_wait)
        else:
            time.sleep(.01) # avoid hogging the CPU
            fps_diff = -(actual_fps - self._target_fps)
            logger.warning(
                f'{self._loop_name} fell behind. '
                f'Target: {self._loop_time:.4f}s ({self._target_fps:.1f} FPS) | '
                f'Actual: {elapsed_time:.4f}s ({actual_fps:.1f} FPS) | '
                f'Diff: {-remaining_time_to_wait:.4f}s ({fps_diff:.1f} FPS)'
                )
            