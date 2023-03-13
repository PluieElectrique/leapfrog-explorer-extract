import logging
import lzma

import numpy as np
from PIL import Image


class Parser:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, n=-1):
        """Read n bytes, or all remaining bytes if n = -1."""
        if n == -1:
            n = len(self.data) - self.i
        elif n < 0:
            raise Exception("n must be -1 or >= 0")
        elif self.i + n > len(self.data):
            remaining = len(self.data) - self.i
            raise Exception(
                f"Tried to read {n} byte(s) but only {remaining} byte(s) left"
            )

        d = self.data[self.i : self.i + n]
        self.i += n
        return d

    def eof(self):
        return self.i == len(self.data)

    def read_u32(self):
        return int.from_bytes(self.read(4), "little")

    def read_i32(self):
        return int.from_bytes(self.read(4), "little", signed=True)

    def read_string(self):
        """Read a null-terminated string."""
        string = b""
        while (c := self.read(1)) != b"\x00":
            string += c
        return string.decode()

    def __len__(self):
        return len(self.data)


def decompress(data, decomp_size):
    """Decompress LZMA1 stream used in data.arc."""
    if len(data) == decomp_size:
        # Not compressed
        return data

    # We can't use lzma.decompress() since it expects an "end-of-stream marker" not used in data.arc
    decomp = lzma.LZMADecompressor(
        # Leaving all settings (e.g. lp, pb, lc) at defaults seems to work.
        format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA1}]
    )
    out = decomp.decompress(data)
    assert len(decomp.unused_data) == 0

    if len(out) > decomp_size:
        # So far, the decompressed data only ever differs from the expected size by 1.
        # And, that extra byte is always null. So, this doesn't seem to be a problem?
        extra = len(out) - decomp_size
        logging.debug(f"Found {extra} extra bytes, truncating data: {out[-extra:]}")
        # We truncate because later code (e.g. OOT reading) expects the size to match decomp_size
        out = out[:-extra]

    return out


def read_oot(data):
    data = Parser(data)

    unknown1     = data.read_u32()
    width        = data.read_u32()
    height       = data.read_u32()
    width_pad    = data.read_u32()  # Width rounded to next power of 2
    height_pad   = data.read_u32()  # Height rounded to next power of 2
    uncomp_size  = data.read_u32()  # Unsure since I have no compressed files to test
    unknown2     = [data.read_u32() for _ in range(3)]
    pixel_format = data.read_i32()
    unknown3     = [data.read_u32() for _ in range(6)]

    compressed = pixel_format < 0
    if compressed:
        pixel_format = -pixel_format

    logging.debug(f"Width:  {width :>3} (padded to {width_pad :>3})")
    logging.debug(f"Height: {height:>3} (padded to {height_pad:>3})")
    logging.debug(f"Compressed: {compressed}")
    logging.debug(f"Uncompressed size (?): {uncomp_size}")
    logging.debug(f"Pixel format: {pixel_format}")
    logging.debug(f"Unknown data 1: {unknown1}")
    logging.debug(f"Unknown data 2: {unknown2}")
    logging.debug(f"Unknown data 3: {unknown3}")

    if pixel_format not in [0, 1, 2]:
        raise Exception(f"Unsupported pixel format: {pixel_format}")
    elif compressed:
        raise Exception(
            "OOT compression not supported yet "
            f"(compressed size {len(data)}, uncompressed size (?) {uncomp_size})"
        )

    dtype = np.dtype(np.uint16).newbyteorder("<")
    i = np.frombuffer(data.read(), dtype=dtype)
    i = i.reshape(height_pad, width_pad)

    if pixel_format == 0:
        # rgba5551
        r = ((i >> 11) & 0x1F) << 3
        g = ((i >>  6) & 0x1F) << 3
        b = ((i >>  1) & 0x1F) << 3
        a = ((i >>  0) & 0x1)   * 0xFF
        i = np.stack([r, g, b, a], axis=-1)
    elif pixel_format == 1:
        # rgba4444
        r = ((i >> 12) & 0xF) << 4
        g = ((i >>  8) & 0xF) << 4
        b = ((i >>  4) & 0xF) << 4
        a = ((i >>  0) & 0xF) << 4
        i = np.stack([r, g, b, a], axis=-1)
    elif pixel_format == 2:
        # rgb565
        r = ((i >> 11) & 0x1F) << 3
        g = ((i >>  5) & 0x3F) << 2
        b = ((i >>  0) & 0x1F) << 3
        i = np.stack([r, g, b], axis=-1)

    mode = "RGBA" if pixel_format in [0, 1] else "RGB"
    i = Image.fromarray(i.astype(np.uint8), mode)
    # There doesn't seem to be anything hidden in the padding regions
    i = i.crop((0, 0, width, height))

    return i


if __name__ == "__main__":
    import argparse
    import os
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable=None, *args):
            return iterable

    parser = argparse.ArgumentParser(
        description='Extract files from the "data.arc" of LeapFrog Explorer games'
    )
    parser.add_argument("data_arc", default="data.arc", help="Path to data.arc file")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument(
        "--oot_format", default="png", metavar="FORMAT",
        help="Convert OOT images to this format (default: %(default)s)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s: %(message)s")

    with open(args.data_arc, "rb") as f:
        arc = Parser(f.read())

    header_decomp_size = arc.read_u32()
    header_comp_size   = arc.read_u32()

    header = Parser(decompress(arc.read(header_comp_size), header_decomp_size))

    files = []
    while not header.eof():
        file_path = header.read_string()
        offset_comp = header.read_u32()
        decomp_size = header.read_u32()
        files.append((file_path, offset_comp, decomp_size))

    for i, (file_path, offset_comp, decomp_size) in enumerate(tqdm(files)):
        logging.debug(f"Extracting {file_path}")
        out_path = os.path.join(args.out_dir, file_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if i < len(files) - 1:
            # Subtract offsets to get the compressed size
            comp_size = files[i + 1][1] - offset_comp
        else:
            # For the last compressed file, read until the end of data.arc
            comp_size = -1

        file = decompress(arc.read(comp_size), decomp_size)

        if file_path.endswith(".oot"):
            try:
                i = read_oot(file)
            except Exception as exc:
                raise Exception(f"Failed to read OOT image: {file_path}") from exc

            i.save(out_path + f".{args.oot_format}")
        else:
            with open(out_path, "wb") as f:
                f.write(file)
