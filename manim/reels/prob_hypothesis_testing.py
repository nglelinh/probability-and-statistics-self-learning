# -*- coding: utf-8 -*-
from common import *

class HypothesisTesting(Scene):
    def construct(self):
        timeline, end_vo, save_tl = setup_vo(self)
        t0 = self.time; timeline.append(("s1", t0))
        hdr = make_header("KIỂM ĐỊNH GIẢ THUYẾT", "Thí nghiệm Quý bà Nếm trà & P-value vs. Bayes")
        self.play(FadeIn(hdr), run_time=0.6)

        b1 = card(7.6, 2.7, SUN).move_to([0, 1.9, 0])
        t1 = VGroup(
            fit(T("R. A. Fisher: 8 tách trà (4 sữa trước, 4 trà trước)", 22, SUN), 7.0),
            fit(T("Nếu đoán mò (H₀): P(đúng cả 4) = 1 / C(8,4) = 1/70 ≈ 0,014", 21, WHITE), 7.0),
            fit(T("p-value = 0,014 < 0,05 ⇒ Bác bỏ H₀!", 23, MINT), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b1)

        b2 = card(7.6, 2.7, PINK).move_to([0, -1.7, 0])
        t2 = VGroup(
            fit(T("Cảnh báo p-value ≠ P(H₀ đúng | Dữ liệu)!", 24, PINK), 7.0),
            fit(T("• p-value là P(Dữ liệu | H₀ đúng)", 22, WHITE), 7.0),
            fit(T("• Muốn biết P(H | D) trực tiếp ⇒ Dùng Suy diễn Bayes!", 22, SKY), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b2)

        self.play(FadeIn(b1), FadeIn(t1), run_time=1.0)
        self.play(FadeIn(b2), FadeIn(t2), run_time=1.0)
        end_vo("s1", t0, 11.0)
        save_tl()
