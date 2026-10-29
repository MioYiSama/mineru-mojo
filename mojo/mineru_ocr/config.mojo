"""Architecture constants of MinerU2.5-Pro-2605-1.2B (HF `config.json`).

The native engine is specialized to this checkpoint's `Qwen2VLForConditionalGeneration`
dimensions; kernels take them as compile-time parameters.
"""

# Text decoder (`text_config`).
comptime CONTEXT_LENGTH = 8192
comptime TEXT_LAYERS = 24
comptime TEXT_HIDDEN = 896
comptime TEXT_INTERMEDIATE = 4864
comptime VOCAB_SIZE = 151936
comptime TEXT_HEADS = 14
comptime TEXT_KV_HEADS = 2
comptime TEXT_HEAD_DIM = 64
comptime TEXT_ROPE_THETA = Float32(1.0e6)
# `rope_parameters.mrope_section` = [8, 12, 12] frequency pairs for the
# temporal, height and width position axes.
comptime MROPE_TEMPORAL_PAIRS = 8
comptime MROPE_HEIGHT_PAIRS = 12
comptime IM_END_TOKEN = 151645
comptime END_OF_TEXT_TOKEN = 151643

# Vision tower (`vision_config`).
comptime VISION_LAYERS = 32
comptime VISION_HIDDEN = 1280
comptime VISION_HEADS = 16
comptime VISION_HEAD_DIM = VISION_HIDDEN // VISION_HEADS
comptime VISION_ROPE_THETA = Float32(1.0e4)
comptime PATCH_DIM = 3 * 2 * 14 * 14
"""Flattened patch length: channels x temporal patch x patch x patch."""
comptime MERGE_SIZE = 2
comptime MERGE_UNIT = MERGE_SIZE * MERGE_SIZE
"""Patches merged into one text-decoder token."""
comptime MERGED_DIM = VISION_HIDDEN * MERGE_UNIT
