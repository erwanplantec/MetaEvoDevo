import PIL
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

font_path = plt.matplotlib.get_data_path()+'/fonts/ttf/'

def gen_points(s, font_size=40, normalize_size=False):
    font = PIL.ImageFont.truetype(font_path+'DejaVuSans.ttf', font_size)
    (left, top, right, bottom) = font.getbbox(s)
    w, h = right-left, bottom-top
    im = PIL.Image.new('L', (w, h))
    draw  = PIL.ImageDraw.Draw(im)
    draw.text((-left, -top), s, fill=255, font=font)
    im = np.uint8(im)
    y, x = np.float32(im.nonzero())
    pos = np.column_stack([x, y])
    if len(pos) > 0:
        pos -= (w/2, h/2)
        pos[:,1] *= -1
    if normalize_size:
        pos /= font_size
    return pos


def load_image(url, max_size=40):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji, size=40):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    return load_image(url, size)