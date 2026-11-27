# -*- coding: utf-8 -*-
from manim import *
import os, json

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_width = 9
config.frame_height = 16
config.background_color = "#10121a"

SUN = "#ffd166"
SKY = "#4cc9f0"
MINT = "#80ed99"
PINK = "#ff7eb6"
DIM = "#7b8294"
CARD = "#1d2130"
FONT = "DejaVu Sans"

def T(s, size=34, color=WHITE, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)

def fit(m, max_w=7.8):
    if m.width > max_w:
        m.scale_to_fit_width(max_w)
    return m

def card(w, h, color=SKY):
    return RoundedRectangle(width=w, height=h, corner_radius=0.22,
                            fill_color=CARD, fill_opacity=0.92,
                            stroke_color=color, stroke_width=2.5)

def make_header(tag, title):
    t1 = fit(T(tag, 22, SUN), 7.6)
    t2 = fit(T(title, 32, WHITE), 7.8)
    return VGroup(t1, t2).arrange(DOWN, buff=0.14).move_to([0, 6.2, 0])

def setup_vo(scene):
    vo_durs = json.load(open(os.environ["VO_DUR_JSON"])) if os.environ.get("VO_DUR_JSON") else {}
    tl_path = os.environ.get("VO_TIMELINE_JSON")
    timeline = []
    def end_vo(sid, t0, fb):
        dur = vo_durs.get(sid)
        if dur is not None:
            rem = dur - (scene.time - t0) + 0.45
            if rem > 0.05:
                scene.wait(rem)
        else:
            scene.wait(fb)
    def save_tl():
        if tl_path:
            with open(tl_path, "w") as f:
                json.dump({"timeline": timeline, "total": float(scene.time)}, f)
    return timeline, end_vo, save_tl
