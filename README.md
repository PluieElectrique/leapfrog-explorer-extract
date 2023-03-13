# LeapFrog Explorer Extract

Extract files from the "data.arc" of LeapFrog Explorer games.

## Setup

You will need Python 3.8+ (tested on 3.10). Install dependencies with:

```
pip install -r requirements.txt
```

([tqdm](https://tqdm.github.io/) can be skipped if you don't want a progress bar.)

## Usage

```
python extract.py path/to/data.arc path/to/output_dir
```

See all options with `python extract.py -h`.

## Supported features

- Extracting files from data.arc
- Reading uncompressed OOT images (pixel formats 0, 1, 2)
