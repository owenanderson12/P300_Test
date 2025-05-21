#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:31:21 2025

@author: owenanderson

Preprocess and visualize averaged P300 responses from a BIDS dataset
that uses BrainVision .vhdr/.vmrk files with markers "Marker/1.0" and "Marker/0.0".

Steps:
  1) Read BIDS data using MNE-BIDS.
  2) Band-pass filter.
  3) Inspect the continuous time-series for channel 8 interactively.
  4) Extract events from annotations.
  5) Epoch the data around those events.
  6) Average (compute ERPs) and visualize the waveforms and topomaps.
"""

import os
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Adjust these to match your experiment ===
BIDS_ROOT      = "bids_dataset"   # Folder containing your BIDS dataset
SUBJECT_ID     = "01"
SESSION_ID     = "01"
TASK_NAME      = "p300"
RUN_ID         = "01"
EEG_REFERENCE  = "average"
HP_FILTER      = 1.0
LP_FILTER      = 30.0
EPOCH_TMIN     = -0.2
EPOCH_TMAX     = 0.8
BASELINE       = (None, 0.0)

def main():
    # 1) Define a BIDS path
    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=SUBJECT_ID,
        session=SESSION_ID,
        task=TASK_NAME,
        run=RUN_ID,
        datatype="eeg"
    )

    print(f"[INFO] Reading raw data from BIDS path: {bids_path}")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    print("[INFO] Manually loading data into memory...")
    raw.load_data()

    print("[DEBUG] Found annotation descriptions:")
    for desc in raw.annotations.description:
        print(f"'{desc}'")

    print(f"[INFO] Setting EEG reference to: {EEG_REFERENCE}")
    raw.set_eeg_reference(ref_channels=EEG_REFERENCE)

        # --- Custom montage for your channels ---
    # Define custom 3D coordinates (in meters) for your 8 channels.
    # Here we set the z-coordinate to 0.0. Adjust these values as needed.
    # --- Custom montage for OpenBCI 8-channel configuration ---
    # Mapping:
    # CH1: Fp1, CH2: Fp2, CH3: C3, CH4: C4, CH5: T7, CH6: T8, CH7: P7, CH8: P8
    ch_pos = {
        'CH1': [-0.08, -0.10, 0.0],   # O1: left occipital
        'CH2': [-0.13,  0.00, 0.0],   # T7: left temporal
        'CH3': [-0.08,  0.02, 0.0],   # C3: left central
        'CH4': [-0.03,  0.09, 0.0],   # Fp1: left frontal pole
        'CH5': [ 0.03,  0.09, 0.0],   # Fp2: right frontal pole
        'CH6': [ 0.08,  0.02, 0.0],   # C4: right central
        'CH7': [ 0.13,  0.00, 0.0],   # T8: right temporal
        'CH8': [ 0.08, -0.10, 0.0]    # O2: right occipital
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage)

    print("[INFO] Custom montage set for channels CH1-CH8.")

    print(f"[INFO] Filtering data: {HP_FILTER} - {LP_FILTER} Hz")
    raw.filter(l_freq=HP_FILTER, h_freq=LP_FILTER, fir_design='firwin')

    # --- Interactive Inspection of Continuous Data for Channel 8 ---
    print("[INFO] Launching interactive continuous plot for Channel 8. Close the window to continue.")
    raw.copy().pick_channels(["CH8"]).plot(duration=30, scalings='auto')

    print("[INFO] Extracting events from annotations...")
    events, event_id = mne.events_from_annotations(
        raw,
        event_id={
            'Marker/1.0': 1,  # target
            'Marker/0.0': 2   # non-target
        }
    )
    print("Event ID mapping:", event_id)

    print(f"[INFO] Creating epochs (tmin={EPOCH_TMIN}, tmax={EPOCH_TMAX}) with baseline={BASELINE}")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        picks="eeg",
        preload=True
    )

    print("[INFO] Averaging epochs by condition...")
    evoked_target = epochs[1].average()
    evoked_nontarget = epochs[2].average()

    diff_evoked = mne.combine_evoked([evoked_target, evoked_nontarget], weights=[1, -1])
    diff_evoked.comment = "Target - NonTarget"

    # Common plotting parameters
    plot_params = dict(
        spatial_colors=False,
        units=dict(eeg='µV'),
        scalings=dict(eeg=1e6),
        ylim=dict(eeg=[-100000000, 100000000]),
        show=False  # We'll add legend & colors before showing
    )

    # A helper function to color lines + add a legend
    def color_and_legend(evoked, fig):
        ax = fig.axes[0]  # main axis
        lines = ax.lines
        n_chans = len(evoked.ch_names)
        cmap = cm.get_cmap('Dark2', n_chans)
        for i, line in enumerate(lines):
            line.set_color(cmap(i))
            line.set_label(evoked.ch_names[i])
        ax.legend(loc='upper right', ncol=2)

    # --- Plot TARGET ---
    print("[INFO] Plotting Target (Marker/1.0)")
    fig_target = evoked_target.plot(**plot_params, window_title="Target (Marker/1.0)")
    color_and_legend(evoked_target, fig_target)
    fig_target.show()

    # --- Plot NON-TARGET ---
    print("[INFO] Plotting Non-Target (Marker/0.0)")
    fig_nontarget = evoked_nontarget.plot(**plot_params, window_title="Non-Target (Marker/0.0)")
    color_and_legend(evoked_nontarget, fig_nontarget)
    fig_nontarget.show()

    # --- Plot DIFFERENCE ---
    print("[INFO] Plotting Difference (Target - NonTarget)")
    fig_diff = diff_evoked.plot(**plot_params, window_title="Difference (Target - NonTarget)")
    color_and_legend(diff_evoked, fig_diff)
    fig_diff.show()

    # (Optional) Plot topomaps around 300 ms
    times_of_interest = [0.3]
    evoked_target.plot_topomap(times=times_of_interest)
    evoked_nontarget.plot_topomap(times=times_of_interest)
    diff_evoked.plot_topomap(times=times_of_interest)

    print("[INFO] Preprocessing and plotting complete.")
    print("Close the plots to finish.")
    plt.show()

if __name__ == "__main__":
    main()
