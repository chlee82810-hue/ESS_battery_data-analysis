"""Resolve a Korean font on macOS or Linux; override with BATTERY_FONT_PATH."""
import os
from pathlib import Path

def korean_font():
    options=[os.environ.get('BATTERY_FONT_PATH',''),'/System/Library/Fonts/Supplemental/AppleGothic.ttf',
             '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for item in options:
        if item and Path(item).is_file():return item
    raise FileNotFoundError('Install a Korean TTF font and set BATTERY_FONT_PATH to its absolute path.')
