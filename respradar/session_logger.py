"""CSV session logger for the dashboard."""
import csv
import os
import time


class SessionLogger:
    """Logs BPM samples and motion events to a timestamped CSV file.

    Schema:
        t_sec, event, fft_bpm, zc_bpm, ac_bpm,
        consensus_bpm, consensus_confidence, motion_state, notes

    Event types:
        SESSION_START  — first row of a recording, notes = filename
        BPM_SAMPLE     — periodic snapshot (default every 1 s)
        MOTION_START   — rising edge of motion detector
        MOTION_END     — falling edge of motion detector
        SESSION_END    — last row before file close

    `t_sec` is monotonic-clock seconds since the session started — immune to
    wall-clock changes (NTP, manual edits) during recording.
    """

    HEADER = ["t_sec", "event", "fft_bpm", "zc_bpm", "ac_bpm",
              "consensus_bpm", "consensus_confidence", "motion_state", "notes"]

    def __init__(self):
        self.file = None
        self.writer = None
        self.start_mono = None
        self.path = None
        self.events_logged = 0

    @property
    def is_recording(self):
        return self.file is not None

    def start(self, directory="."):
        """Open a new CSV file, write the header, log SESSION_START."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"radar_session_{ts}.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.start_mono = time.monotonic()
        self.events_logged = 0
        wall = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write("SESSION_START",
                    notes=f"{os.path.basename(self.path)} | wall_start={wall}")
        return self.path

    def stop(self):
        """Log SESSION_END and close the file."""
        if self.file is not None:
            self._write("SESSION_END")
            self.file.close()
            self.file = None
            self.writer = None
            self.start_mono = None

    def t_elapsed(self):
        if self.start_mono is None:
            return 0.0
        return time.monotonic() - self.start_mono

    def _write(self, event_type, fft=None, zc=None, ac=None,
               consensus=None, confidence=None, motion=None, notes=""):
        if self.writer is None:
            return
        row = [
            f"{self.t_elapsed():.3f}",
            event_type,
            f"{fft:.2f}" if fft is not None else "",
            f"{zc:.2f}" if zc is not None else "",
            f"{ac:.2f}" if ac is not None else "",
            f"{consensus:.2f}" if consensus is not None else "",
            confidence if confidence is not None else "",
            f"{int(motion)}" if motion is not None else "",
            notes,
        ]
        self.writer.writerow(row)
        self.file.flush()           # survive a crash — every row hits disk
        self.events_logged += 1

    def log_bpm_sample(self, fft, zc, ac, consensus, confidence, motion):
        """Periodic BPM snapshot — call every 1 s during recording."""
        self._write("BPM_SAMPLE",
                    fft=fft, zc=zc, ac=ac,
                    consensus=consensus, confidence=confidence,
                    motion=motion)

    def log_motion_event(self, started, fft, zc, ac, consensus, confidence):
        """Rising/falling edge of the motion detector — call on transition only."""
        ev = "MOTION_START" if started else "MOTION_END"
        self._write(ev,
                    fft=fft, zc=zc, ac=ac,
                    consensus=consensus, confidence=confidence,
                    motion=(1 if started else 0))


