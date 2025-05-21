#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:37:03 2025

@author: owenanderson

Improved Single-file P300 experiment script using LSL, with enhanced error handling,
logging, precise stimulus timing, and configurable parameters.
  1) Listens to EEG from OpenBCI GUI (LSL).
  2) Presents stimuli & sends LSL markers (PsychoPy) with precise timing.
  3) Merges EEG & markers in real time into a CSV.
  4) Converts data to BIDS after experiment ends.

Enhancements implemented:
  • Parameterized merge threshold and polling sleep interval.
  • Replaced print() with logging for robust, timestamped messages.
  • Wrapped critical sections with try/except for better error handling.
  • Implemented flush_remaining() to write leftover data on termination.
  • Used win.callOnFlip() for more accurate stimulus marker timing.
  • **Clock correction:** Computes an offset from the EEG inlet and subtracts it from marker timestamps.
  
Requirements:
  pip install pylsl psychopy mne mne-bids pandas
"""

import threading
import time
import random
import csv
import os
import sys
from collections import deque
from datetime import datetime
import numpy as np
import logging

# Configure logging: timestamped, INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Configuration / Parameters ---
EEG_STREAM_NAME    = "OpenBCI_EEG"       # Must match your OpenBCI GUI stream name
MARKER_STREAM_NAME = "LSL_MarkerStream"  # Marker stream name

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# P300 Experiment Timing Parameters
NUM_TRIALS           = 300
TARGET_PROBABILITY   = 0.25
STIMULUS_DURATION    = 0.2  # seconds
INTER_TRIAL_INTERVAL = 0.7  # seconds

# File/directories
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)
BIDS_ROOT = "bids_dataset"
os.makedirs(BIDS_ROOT, exist_ok=True)

# BIDS metadata parameters
SUBJECT_ID = "sub-01"
SESSION_ID = "ses-01"
TASK_NAME  = "p300"
RUN_ID     = "run-01"

# --- New/Configurable constants ---
MERGE_THRESHOLD = 0.002  # seconds threshold for aligning EEG and marker timestamps
POLL_SLEEP = 0.001       # sleep time (in seconds) between polls in collector loop

###############################################################################
#                         DATA COLLECTOR (Improved with Clock Correction)
###############################################################################
class LSLDataCollector(threading.Thread):
    """
    Background thread that:
      - Subscribes to EEG and Marker LSL streams.
      - Computes a clock offset from the EEG inlet.
      - Buffers and merges data by timestamp (adjusting marker timestamps).
      - Writes merged rows to a CSV file.
      - Flushes any remaining data upon termination.
    """
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event
        self.eeg_buffer = deque()
        self.marker_buffer = deque()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = os.path.join(RAW_DATA_DIR, f"EEG_LSL_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0  # will hold the computed clock offset

    def resolve_streams(self):
        try:
            logging.info("Resolving EEG LSL stream...")
            from pylsl import resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{EEG_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            # Compute the clock offset between the EEG device and the local clock.
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"Computed clock offset: {self.clock_offset:.6f} seconds")

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error(f"No Marker stream found with name '{MARKER_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)

            logging.info("LSL streams resolved. Starting data collection...")
        except Exception as e:
            logging.exception("Exception during stream resolution:")
            sys.exit(1)

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                    row = [ts_eeg] + eeg_data + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(EEG_CHANNELS)) + [marker]
                writer.writerow(row)

    def run(self):
        from pylsl import resolve_byprop, StreamInlet  # local import for thread safety
        self.resolve_streams()
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self.stop_event.is_set():
                    try:
                        sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                        if sample_eeg is not None and ts_eeg is not None:
                            self.eeg_buffer.append((ts_eeg, sample_eeg))
                    except Exception as e:
                        logging.exception("Error pulling EEG sample:")
                    try:
                        sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                        if sample_marker is not None and ts_marker is not None:
                            # Adjust the marker timestamp by subtracting the clock offset.
                            adjusted_ts_marker = ts_marker - self.clock_offset
                            marker_val = sample_marker[0]
                            self.marker_buffer.append((adjusted_ts_marker, marker_val))
                    except Exception as e:
                        logging.exception("Error pulling Marker sample:")

                    # Merge data from both buffers
                    while self.eeg_buffer and self.marker_buffer:
                        ts_eeg, eeg_data = self.eeg_buffer[0]
                        ts_marker, marker = self.marker_buffer[0]
                        if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                            row = [ts_eeg] + eeg_data + [marker]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                            self.marker_buffer.popleft()
                        elif ts_marker < ts_eeg:
                            row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                            writer.writerow(row)
                            self.marker_buffer.popleft()
                        else:
                            row = [ts_eeg] + eeg_data + [""]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                    time.sleep(POLL_SLEEP)
                # After stop_event is set, flush remaining data
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. Data saved to {self.output_csv}")

###############################################################################
#                        EXPERIMENT LOGIC (Enhanced Stimulus Timing)
###############################################################################
def run_p300_experiment():
    from pylsl import StreamInfo, StreamOutlet
    # Create LSL Marker stream
    try:
        marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
        marker_outlet = StreamOutlet(marker_info)
        logging.info("Marker stream created.")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        sys.exit(1)

    # PsychoPy setup
    try:
        from psychopy import visual, core
        win = visual.Window(
            size=(1920, 1080),
            color="black",
            units="norm",
            fullscr=False
        )
        logging.info("PsychoPy window created.")
    except Exception as e:
        logging.exception("Error setting up PsychoPy window:")
        sys.exit(1)

    # Stimuli creation
    stim_white_flash = visual.Rect(
        win=win,
        width=2,
        height=2,
        fillColor="white",
        lineColor="white",
        pos=(0, 0)
    )
    stim_nontarget = visual.TextStim(win, text="", color="white", height=50)

    # Generate trial list
    trials = []
    for _ in range(NUM_TRIALS):
        if random.random() < TARGET_PROBABILITY:
            trials.append(("target", "1"))
        else:
            trials.append(("non-target", "0"))

    # Loop over trials with precise marker timing
    for idx, (stim_type, marker_val) in enumerate(trials, start=1):
        stimulus = stim_white_flash if stim_type == "target" else stim_nontarget
        stimulus.draw()
        # Schedule marker push immediately after the window flip for precision
        win.callOnFlip(lambda: marker_outlet.push_sample([marker_val]))
        win.flip()
        logging.info(f"Trial {idx}/{NUM_TRIALS}: {stim_type}, marker={marker_val}")
        core.wait(STIMULUS_DURATION)
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
    win.close()
    logging.info("P300 experiment finished.")

###############################################################################
#                             BIDS CONVERSION
###############################################################################
def convert_to_bids(csv_path):
    logging.info(f"Converting {csv_path} to BIDS...")
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.exception("Failed to read CSV file for BIDS conversion:")
        return

    # Verify that expected EEG channel columns exist
    missing_cols = [ch for ch in EEG_CHANNELS if ch not in df.columns]
    if missing_cols:
        logging.error(f"CSV is missing expected EEG channels: {missing_cols}")
        return

    # Extract EEG data
    data_eeg = []
    for ch in EEG_CHANNELS:
        col_data = pd.to_numeric(df[ch], errors="coerce").fillna(0).values
        data_eeg.append(col_data)
    data_eeg = np.array(data_eeg)  # shape: (n_channels, n_samples)

    # Process markers
    raw_markers = df["marker"].astype(str).replace({"nan": "", "None": ""}).values
    markers = []
    for i, m in enumerate(raw_markers):
        if m != "":
            markers.append((i, m))

    from mne import create_info
    from mne.io import RawArray
    info = create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLE_RATE, ch_types="eeg")
    raw = RawArray(data_eeg, info)

    # Create annotations from markers
    annotations_onset = []
    annotations_desc  = []
    for (idx, marker_val) in markers:
        onset_sec = idx / SAMPLE_RATE  # Alternatively, use LSL timestamps if available
        annotations_onset.append(onset_sec)
        annotations_desc.append(f"Marker/{marker_val}")

    import mne
    raw.set_annotations(mne.Annotations(
        onset=annotations_onset,
        duration=[0.0]*len(annotations_onset),
        description=annotations_desc
    ))

    from mne_bids import write_raw_bids, BIDSPath
    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=SUBJECT_ID.replace("sub-", ""),
        session=SESSION_ID.replace("ses-", ""),
        task=TASK_NAME.replace("task-", ""),
        run=RUN_ID.replace("run-", ""),
        datatype="eeg"
    )
    try:
        write_raw_bids(
            raw,
            bids_path,
            overwrite=True,
            allow_preload=True,
            format='BrainVision'
        )
        logging.info(f"BIDS dataset written at: {bids_path.directory}")
    except Exception as e:
        logging.exception("Error during BIDS conversion:")

###############################################################################
#                                   MAIN
###############################################################################
def main():
    logging.info("Starting P300 experiment script with enhanced precision and robustness.")
    # 1) Start data collector thread
    stop_event = threading.Event()
    collector = LSLDataCollector(stop_event)
    collector.start()

    # 2) Run the experiment (stimulus presentation)
    run_p300_experiment()

    # 3) Stop the collector thread gracefully
    logging.info("Experiment complete. Stopping data collector thread...")
    stop_event.set()
    collector.join()
    csv_path = collector.output_csv

    # 4) Convert CSV to BIDS format
    convert_to_bids(csv_path)

    logging.info("All processes complete. Exiting script.")

if __name__ == "__main__":
    main()
