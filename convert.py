"""
This script converts the PyTorch weights of OWL-ViT to the gguf file.
Required files can be downloaded using files.sh

The file is structured as follows:
    - Hyperparameters
    - Vocabulary
    - Text model
    - Vision model
    - Detection head [cls, bbox]

For each tensor, the bytes are packed as follows:
    - Number of dimensions    (int)
    - Name length             (int)
    - Dimensions              (int[n_dims])
    - Name                    (char[name_length])
    - Data                    (float[n_dims])
"""

"""

HEADER -> KV_DATA => TI_DATA, WEIGHTS

"""

import os
import shutil
import struct
import tempfile
from enum import Enum, auto
from io import BufferedWriter
from typing import IO, Any, Sequence, Mapping
from string import ascii_letters, digits
import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys


# Constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION = 2  # GGML_QNT_VERSION from ggml.h


# FTYPE = 0
# MODEL_DIR = '/mnt/sda/sudhanva/Q-stuff/vit_sud/model'


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", type=str, required=True)
parser.add_argument("-o", "--out-dir", type=str, required=False)
parser.add_argument("-f", "--use-f16", action="store_false")

if len(sys.argv) < 4:
    print("Usage: convert.py -m path-to-model-repo -o output-dir -f [use-f32]\n")
    sys.exit(1)


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def parse_params(model_dir):
    """Parse and write both vision and text hparams"""
    with open(f"{model_dir}/config.json", "r") as f:
        config = json.load(f)

        h_params = {"projection_dim": config["projection_dim"]}

        t_params = {
            "vocab_size": config["text_config"]["vocab_size"],
            "num_hidden_layers": config["text_config"]["num_hidden_layers"],
            "hidden_size": config["text_config"]["hidden_size"],
        }

        v_params = {
            "image_size": config["vision_config"]["image_size"],
            "hidden_size": config["vision_config"]["hidden_size"],
            "num_channels": config["vision_config"]["num_channels"],
            "num_hidden_layers": config["vision_config"]["num_hidden_layers"],
            "patch_size": config["vision_config"]["patch_size"],
        }

        return h_params, v_params, t_params


def parse_tokenizer(file, model_dir):
    tokenizer_loc = f"{model_dir}/vocab.json"
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    with open(tokenizer_loc, "r", encoding="utf8") as f:
        _tokens_raw = json.load(f)

        if "<|endoftext|>" in _tokens_raw:
            # ensures exact same model as tokenizer_type == tiktoken
            # details: https://github.com/ggerganov/whisper.cpp/pull/725
            del _tokens_raw["<|endoftext|>"]
        tokens = {
            bytes([byte_decoder[c] for c in token]): int(idx)
            for token, idx in _tokens_raw.items()
        }

        # write tokenizer
        file.write(struct.pack("i", len(tokens)))

        for key in tokens:
            file.write(struct.pack("i", len(key)))
            file.write(key)


def parse_weights(file, model_dir, ftype):
    model = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")

    for k, v in model.items():
        print(f"Processing layer: {k} with shape: {v.shape} and type: {v.dtype}")
        process_and_write_layer(file, k, v, ftype)


def process_and_write_layer(file, name, tensor, ftype):
    """
    https://github.com/ggerganov/ggml/blob/2aae01fd9b8f9399f343cf18f46f38996ef52e2c/examples/whisper/convert-pt-to-ggml.py#L296C28-L296C35

    Perform squeeze and convert to numpy

    """
    # look into pytorch inference implementation
    # data = tensor.numpy() ^
    data = tensor.squeeze().numpy()
    n_dims = len(data.shape)

    """
        Converting small tensors to fp32
        https://github.com/ggerganov/ggml/blob/2aae01fd9b8f9399f343cf18f46f38996ef52e2c/examples/whisper/convert-pt-to-ggml.py#L306
        """

    # Check why certain layers were type-casted
    if use_f16:
        if n_dims < 2:
            print("  Converting to float32")
            data = data.astype(np.float32)

    else:
        data = data.astype(np.float32)

    # Look for information on the reshaping operation
    # Broadcasting operation was performed to add more dimensions for patch embeddings
    # https://github.com/ggerganov/ggml/blob/master/examples/sam/convert-pth-to-ggml.py#L128
    # https://github.com/ggerganov/ggml/blob/master/examples/sam/main.cpp#L734-L735
    # Later it was used as a 4D tensor ^
    # GGML Repeat is broadcast??
    # https://github.com/ggerganov/ggml/blob/9d562d712513c77a4de44ad0428be62bc3f2a9cf/include/ggml/ggml.h#L999

    # header
    layer_name = name.encode("utf-8")
    file.write(struct.pack("iii", n_dims, len(layer_name), ftype))
    for i in range(n_dims):
        file.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    file.write(layer_name)

    # data -> https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#
    data.tofile(file)


def write_gguf(fname_out, model_dir, use_f16):

    h_params, v_params, t_params = parse_params(model_dir)

    # Write to file
    fout = open(fname_out, "wb")

    # Refer -> https://docs.python.org/3/library/struct.html
    fout.write(struct.pack("<I", GGUF_MAGIC))  # Magic: ggml in hex
    fout.write(struct.pack("I", GGUF_VERSION))

    for h_param in h_params.values():
        fout.write(struct.pack("i", h_param))

    for v_param in v_params.values():
        fout.write(struct.pack("i", v_param))

    for t_param in t_params.values():
        fout.write(struct.pack("i", t_param))

    fout.write(struct.pack("i", use_f16))

    parse_tokenizer(fout, model_dir)
    parse_weights(fout, model_dir, use_f16)

    fout.close()

    print("Model converted and saved to '{}'".format(fname_out))


if __name__ == "__main__":

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise ValueError(f"Could not find directory {model_dir}")

    if args.out_dir is None:
        out_dir = model_dir
    else:
        out_dir = Path(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # use 16-bit or 32-bit floats
    # Read about pytorch storage mechanisms & mixed precision
    if args.use_f16 is not None:
        use_f16 = args.use_f16

    print(use_f16)

    # Output file name
    fname_out = f"./OWL-ViT.gguf"
    if not use_f16:
        fname_out = f"./OWL-ViT-f32.gguf"

    write_gguf(fname_out, model_dir, use_f16)
