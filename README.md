# P300 Test

A project used to test the validity of the OpenBCI Ultracortex Mark IV Headset and gain familiarity. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

## Repository Structure

```text
.
├── bids_dataset/           # BIDS-formatted sample dataset for P300 experiment ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
├── data/
│   └── raw/                # Raw EEG data files ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
├── P300_Test.py            # Main script to run P300 recording session ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
└── preprocess_and_visualize.py  # Script to preprocess EEG data and visualize P300 ERPs ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
```

## Requirements

* Python 3.8+ ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* pyOpenBCI or other OpenBCI Python SDK ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* MNE (for EEG preprocessing and ERP analysis) ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* NumPy (numerical operations) ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* SciPy (signal processing utilities) ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* Matplotlib (plotting ERPs and diagnostics) ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* pandas (data handling, optional) ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

## Installation

Install dependencies via pip:

```bash
pip install mne numpy scipy matplotlib pandas pyOpenBCI
```

## Usage

### 1. Running a P300 Test Session

Launch the main script to record P300 responses from a participant:

```bash
python P300_Test.py --subject SUBJ_ID --output-dir data/raw
```

* `--subject`: Participant identifier (e.g., `sub-01`). ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* `--output-dir`: Directory to save raw EEG files. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

### 2. Preprocessing and Visualization

Process raw EEG and generate ERP plots:

```bash
python preprocess_and_visualize.py --input-dir data/raw --bids-dir bids_dataset --output-dir results
```

* `--input-dir`: Path to raw data. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
* `--bids-dir`: BIDS-formatted dataset root for metadata. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main/bids_dataset))
* `--output-dir`: Directory for cleaned data and plots. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

## Data Format

Raw EEG recordings are stored in `data/raw/` (OpenBCI format) and organized into BIDS under `bids_dataset/`, ensuring compatibility with standard neuroimaging tools. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

## License

This project is released under the MIT License. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))

## Author

Owen Anderson – Brain–computer interface researcher and project maintainer. ([github.com](https://github.com/owenanderson12/P300_Test/tree/main))
