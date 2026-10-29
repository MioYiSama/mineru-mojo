"""Qwen2-VL (``Qwen2VLForConditionalGeneration``) as a MAX custom architecture.

``max serve --custom-architectures <this directory>`` reads ``ARCHITECTURES``.
"""

from .arch import qwen2vl_arch

ARCHITECTURES = [qwen2vl_arch]
