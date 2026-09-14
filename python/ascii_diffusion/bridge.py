"""
Exports a trained DiffusionModel to JSON that js/load-weights.js can load
directly into the tf.js model built by js/model.js's buildModel(). See
model.py's module docstring for how the two were kept in exact numeric sync.

Shape conventions (confirmed against a live tf.js instance — see
bridge_test/):
  - Linear:  torch weight [out,in]      -> TF dense kernel [in,out]   (transpose)
  - Conv2d:  torch weight [out,in,kh,kw] -> TF conv kernel [kh,kw,in,out] (permute 2,3,1,0)
  - biases:                same shape [out] on both sides, no transform
  - LayerNorm gamma/beta:  same shape [channels] on both sides, no transform

Numbers are packed as base64-encoded float16 (little-endian), not JSON
arrays of numbers. For a model with several million parameters, a JSON
array of float32 text runs to ~13 bytes/value (e.g. "-0.123456789,") -
185MB for a 14M-parameter model - which sails past GitHub's 100MB hard
per-file push limit. Git LFS is the usual answer to that, but GitHub Pages
serves LFS-tracked files as their small pointer stub, not the real content
(a permanent, documented Pages limitation) - so LFS "fixes" the push and
then breaks the live site. Base64 float16 instead gets the same model down
to ~2.4MB/million-params (~2.4MB here vs 185MB) - small enough to just be a
normal tracked file, no LFS, no Pages workaround needed. Precision cost is
negligible for this purpose (median relative error ~0.02%, verified against
numpy's float16 reference before this was written - see bridge_test/).

The `*_np` helpers below take plain numpy arrays (call `.detach().cpu().numpy()`
first) specifically so the shape transforms can be unit-tested without a
working torch install — see bridge_test/verify_export_shapes.py.
"""
import base64
import json
import os

import numpy as np


def dense_kernel_np(torch_weight):
    """torch Linear.weight [out,in] -> TF dense kernel [in,out]."""
    return torch_weight.T


def conv_kernel_np(torch_weight):
    """torch Conv2d.weight [out,in,kh,kw] -> TF conv2d kernel [kh,kw,in,out]."""
    return torch_weight.transpose(2, 3, 1, 0)


def _pack(arr):
    """numpy array (any shape/dtype) -> {shape, dtype, data_b64} with values
    packed as little-endian float16. This is the ONLY numeric encoding used
    in exported weight files - see module docstring for why."""
    arr16 = np.ascontiguousarray(arr, dtype="<f2")
    return {
        "shape": list(arr16.shape),
        "dtype": "float16",
        "data_b64": base64.b64encode(arr16.tobytes()).decode("ascii"),
    }


def _dense_entry(linear):
    w = linear.weight.detach().cpu().numpy()
    b = linear.bias.detach().cpu().numpy()
    return {"kernel": _pack(dense_kernel_np(w)), "bias": _pack(b)}


def _conv_entry(conv_same2d):
    w = conv_same2d.conv.weight.detach().cpu().numpy()
    b = conv_same2d.conv.bias.detach().cpu().numpy()
    return {"kernel": _pack(conv_kernel_np(w)), "bias": _pack(b)}


def _norm_entry(norm):
    return {
        "gamma": _pack(norm.gamma.detach().cpu().numpy()),
        "beta": _pack(norm.beta.detach().cpu().numpy()),
    }


def _film_entry(film):
    return _dense_entry(film.proj)


def _block_entry(block):
    return {
        "conv1": _conv_entry(block.conv1), "norm1": _norm_entry(block.norm1), "film1": _film_entry(block.film1),
        "conv2": _conv_entry(block.conv2), "norm2": _norm_entry(block.norm2), "film2": _film_entry(block.film2),
    }


def _build_meta(model, extra_meta=None):
    meta = {
        "height": model.height,
        "width": model.width,
        "vocabSize": model.vocab_size,
        "hidden": model.hidden,
        "timeDim": model.time_dim,
        "bottleneckBlocks": len(model.bottleneck),
        "bottleneckKernel": model.bottleneck[0].conv1.kernel,
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


def _build_weights_dict(model):
    return {
        "tDense1": _dense_entry(model.tDense1),
        "tDense2": _dense_entry(model.tDense2),
        "stemConv": _conv_entry(model.stemConv), "stemNorm": _norm_entry(model.stemNorm), "stemFilm": _film_entry(model.stemFilm),
        "downConv": _conv_entry(model.downConv), "downNorm": _norm_entry(model.downNorm), "downFilm": _film_entry(model.downFilm),
        "bottleneck": [_block_entry(b) for b in model.bottleneck],
        "refine1Conv": _conv_entry(model.refine1Conv), "refine1Norm": _norm_entry(model.refine1Norm), "refine1Film": _film_entry(model.refine1Film),
        "refine2Conv": _conv_entry(model.refine2Conv), "refine2Norm": _norm_entry(model.refine2Norm), "refine2Film": _film_entry(model.refine2Film),
        "convOut": _conv_entry(model.convOut),
    }


def export_model(model, path, extra_meta=None):
    """Writes a single JSON file with {meta: {...}, weights: {...}}. Fine for
    smaller models; see export_model_chunked() for anything that risks
    approaching GitHub's 100MB per-file push limit as models get bigger."""
    bundle = {"meta": _build_meta(model, extra_meta), "weights": _build_weights_dict(model)}
    with open(path, "w") as f:
        json.dump(bundle, f)
    return path


def export_model_chunked(model, out_dir, chunk_size_bytes=20_000_000, extra_meta=None):
    """
    Writes out_dir/manifest.json (small - just meta + the chunk list) plus
    out_dir/part-000.txt, part-001.txt, ... - plain-text fragments of the
    serialized weights payload, split purely by character count with no
    regard for layer boundaries (the payload is pure ASCII - base64 plus
    JSON punctuation - so a character-count split is exactly a byte-count
    split, no encoding edge cases). js/app.js's loadPretrainedChunked()
    fetches every part, concatenates them back in order, and JSON.parses
    the reassembled string - the result is byte-for-byte the same
    {weights: {...}} object export_model() would have produced in one file,
    just never sitting in any single file bigger than chunk_size_bytes.
    This is what actually keeps a model's exported size from ever
    threatening the 100MB/file GitHub hard limit again, no matter how much
    bigger the model gets - chunk count grows, not chunk size.
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = json.dumps({"weights": _build_weights_dict(model)})

    chunk_files = []
    for i, start in enumerate(range(0, len(payload), chunk_size_bytes)):
        chunk = payload[start:start + chunk_size_bytes]
        fname = f"part-{i:03d}.txt"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(chunk)
        chunk_files.append(fname)

    manifest = {
        "meta": _build_meta(model, extra_meta),
        "numChunks": len(chunk_files),
        "chunkFiles": chunk_files,
        "totalLength": len(payload),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path, chunk_files
