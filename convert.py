'''
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
'''

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

#
# constants
#

GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION     = 2  # GGML_QNT_VERSION from ggml.h

FTYPE = 0
MODEL_DIR = '/mnt/sda/sudhanva/Q-stuff/vit_sud/model' 


# ------------------------


def get_hparams():
    '''Parse and write both vision and text hparams'''
    with open(f'{MODEL_DIR}/config.json', 'r') as f:
        config = json.load(f)

        h_params = {
          "projection_dim": config['projection_dim']    
        }

        t_params = {
            "vocab_size" : config['text_config']['vocab_size'],
            "num_hidden_layers" : config['text_config']['num_hidden_layers'],
            "hidden_size" : config['text_config']['hidden_size']
        }

        v_params = {
            "image_size" : config['vision_config']['image_size'],
            "hidden_size" : config['vision_config']['hidden_size'],
            "num_channels": config['vision_config']['num_channels'],
            "num_hidden_layers" : config['vision_config']['num_hidden_layers'],
            "patch_size" : config['vision_config']['patch_size']
        }

        return h_params, v_params, t_params
    
def write_gguf():

    # Output file name
    fname_out = f"./OWL-ViT-{['f32', 'f16'][FTYPE]}.gguf"

    # Write to file
    with open(fname_out, "wb") as fout:
        fout.write(struct.pack("i", GGUF_MAGIC))  # Magic: ggml in hex
        h_params, v_params, t_params = get_hparams()
        for param in h_params.values():
            fout.write(struct.pack("i", param))

        for param in v_params.values():
              
        

       

if __name__ == "__main__":
    # Add model arg parsing

    write_gguf()





# -------------


def main():

    hparams = extract_hparams(MODEL_DIR)
    write_gguf(MODEL_DIR, hparams)

        #     if k == "image_encoder.blocks.0.norm1.weight":
        #         n_enc_state = v.shape[0]

def write_gguf(model_dir, hparams):
        # Output file name
        fname_out = f"./OWL-ViT-{['f32', 'f16'][FTYPE]}.gguf"

        # Write to file
        with open(fname_out, "wb") as fout:
                fout.write(struct.pack("i", GGUF_MAGIC))  # Magic: ggml in hex
                for param in hparams.values():
                        fout.write(struct.pack("i", param))
                fout.write(struct.pack("i", FTYPE))

                # Write id2label dictionary to the file
                write_id2label(fout, model_dir)

                # Process and write model weights
                write_weights(fout, model_dir, FTYPE)
        print("Done. Output file: " + fname_out)

def write_id2label(file, model_dir):
        with open(f'{model_dir}/config.json', 'r') as f:
                j = json.load(f)

        file.write(struct.pack("i", len(j['id2label'])))
        for k, v in j['id2label'].items():
                file.write(struct.pack("i", int(k)))
                enc_ = v.encode("utf-8")
                file.write(struct.pack("i", len(enc_)))
                file.write(enc_)
        
def write_weights(file, model_dir, ftype):
        # Load pytorch model
        model = torch.load(f'{model_dir}/pytorch_model.bin', map_location="cpu")

        for k, v in model.items():
                print(f"Processing variable: {k} with shape: {v.shape} and type: {v.dtype}")
                process_and_write_variable(file, k, v, ftype)

def process_and_write_variable(file, name, tensor, ftype):
        data = tensor.numpy()
        curr_ftype = (1 if ftype == 1 and tensor.ndim != 1 and name not in ["pos_embed", "cls_token"] else 0)
        data = data.astype(np.float32) if curr_ftype == 0 else data.astype(np.float16)

        if name == 'vit.embeddings.patch_embeddings.projection.bias':
                tensor = tensor.reshape([1,tensor.shape[0], 1, 1])

                # print(k, v.reshape([1, v.shape[0], 1, 1]).shape)

        str_name = name.encode("utf-8")
        file.write(struct.pack("iii", len(data.shape), len(str_name), curr_ftype))
        for dim_size in reversed(data.shape):
                file.write(struct.pack("i", dim_size))
        file.write(str_name)
        data.tofile(file)


