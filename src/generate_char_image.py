# https://qiita.com/lazykyama/items/65bcce351f3d1cf07d8e

import hashlib
import argparse
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm

fill_color = {"black": "#fff", "white": "#000"}
ignore_hash = {"black": {"not_visible": "1b9566a98c65d8ca35271032fa35fed1",
                         "tofu": "239642996bad937da5526c0d58625453"},
               "white": {"not_visible": "4085d51d7796fabc9e22bffd35172ecd",
                         "tofu": "87d19ce02c822de512c56a7736bd8451"}}


def generate_char_img(char, font, size, color):
    img = Image.new('L', size, color)
    draw = ImageDraw.Draw(img)
    font_size = int(size[0] * 0.6)
    font_type = ImageFont.truetype(font, font_size)

    char_displaysize = font_type.getsize(char)
    offset = tuple((si - sc) // 2 for si, sc in zip(size, char_displaysize))
    if not all(o >= 0 for o in offset):
        return

    draw.text((offset[0], offset[1] // 2), char, font=font_type, fill=fill_color[color])

    image_hash = hashlib.md5(img.tobytes()).hexdigest()

    if image_hash == ignore_hash[color]["not_visible"]:
        # not visible
        return
    if image_hash == ignore_hash[color]["tofu"]:
        # tofu
        return

    output_path = Path(f"out")
    font_name = Path(font).stem
    path = output_path / color / font_name
    path.mkdir(parents=True, exist_ok=True)

    img.save(path / f"char_{ord(char)}.png", 'png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='character image generator')
    parser.add_argument('--font', type=str)
    parser.add_argument('--color', type=str)
    args = parser.parse_args()

    size = (60, 60)
    char_list = [chr(i) for i in range(1114111)]
    for c in tqdm(char_list):
        generate_char_img(c, args.font, size, args.color)
