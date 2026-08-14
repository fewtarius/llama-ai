#!/usr/bin/env python3
"""Read GGUF metadata (no tensor data) and extract key fields needed for KV
cache memory budgeting."""
import struct
import sys
import json

# GGUF magic + version
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# Type codes from gguf.py
GGML_TYPE_SIZE = {
    0: 4, 1: 4, 2: 4, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
    12: 2, 13: 2, 14: 4, 15: 4, 16: 8, 17: 8, 18: 4, 19: 8, 20: 8, 21: 8, 22: 12, 23: 16,
}

# GGUF v3 value type codes (NOT ggml quantization types).
# Reference: gguf.py GGUFValueType enum.
KV_TYPE_NAMES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

# GGUF type element sizes for array element decoding (only uint32/int32/float32/uint64/int64 supported in arrays)
ARRAY_ELT_SIZES = {0: 1, 1: 1, 4: 4, 5: 4, 11: 8, 12: 8, 6: 4}


def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def read_kv(f, kv_type):
    # GGUF v3 value type codes.
    if kv_type == 0:  # uint8
        return struct.unpack("<B", f.read(1))[0]
    if kv_type == 1:  # int8
        return struct.unpack("<b", f.read(1))[0]
    if kv_type == 4:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    if kv_type == 5:  # int32
        return struct.unpack("<i", f.read(4))[0]
    if kv_type == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    if kv_type == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif kv_type == 6:
        return struct.unpack("<f", f.read(4))[0]
    elif kv_type == 7:  # bool (1 byte)
        return struct.unpack("<B", f.read(1))[0] != 0
    elif kv_type == 8:  # string (length-prefixed)
        return read_string(f)
    elif kv_type == 9:  # array
        array_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        array_elt_size = ARRAY_ELT_SIZES.get(array_type)
        if array_elt_size is None:
            # String arrays: each element is itself a length-prefixed string.
            if array_type == 8:
                for _ in range(array_len):
                    read_string(f)
            elif array_type == 9:
                # Nested arrays - bail out by reading nothing further; we
                # only have these in tokenizer config which we don't care about.
                return f"<nested array len={array_len}>"
            else:
                # Unknown element type - assume 8 bytes per element
                f.read(array_len * 8)
            return f"<array type={array_type}>"
        f.read(array_len * array_elt_size)
        return f"[array type={array_type} len={array_len}]"
    return None


def main():
    path = sys.argv[1]
    # Fields we care about. Stop after we find all of them OR hit a parse error.
    fields_of_interest = {
        "block_count", "embedding_length", "context_length",
        "head_count", "head_count_kv",
        "key_length", "value_length",
        "head_count_kv", "attention.head_count", "attention.head_count_kv",
        "attention.key_length", "attention.value_length",
        "expert_count", "expert_used_count",
        "nextn_predict_layers", "ssm.state_size", "full_attention_interval",
        "expert_feed_forward_length", "ssm.inner_size", "ssm.group_count",
        "ssm.time_step_rank", "ssm.conv_kernel", "rope.freq_base",
    }
    out = {}
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for i in range(n_kv):
            try:
                key = read_string(f)
                kv_type = struct.unpack("<I", f.read(4))[0]
                val = read_kv(f, kv_type)
            except (struct.error, MemoryError, UnicodeDecodeError):
                # Stop parsing - we likely hit tensor data or the tokenizer
                # section which has long string arrays we can't easily skip.
                break
            short_key = key.split(".", 1)[-1] if "." in key else key
            if short_key in fields_of_interest or key in fields_of_interest:
                out[key] = val
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
