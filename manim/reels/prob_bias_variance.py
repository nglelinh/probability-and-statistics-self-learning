# -*- coding: utf-8 -*-
from common import *

class BiasVarianceTradeoff(Scene):
    def construct(self):
        timeline, end_vo, save_tl = setup_vo(self)
        t0 = self.time; timeline.append(("s1", t0))
        hdr = make_header("ĐÁNH ĐỔI BIAS – VARIANCE", "Phân rã Sai số Kỳ vọng & Điều chuẩn")
        self.play(FadeIn(hdr), run_time=0.6)

        b1 = card(7.6, 2.6, SUN).move_to([0, 2.0, 0])
        t1 = VGroup(
            fit(T("E[(y − f̂(x))²] = Bias² + Variance + Nhiễu σ²", 23, SUN), 7.0),
            fit(T("Mô hình quá đơn giản: Bias cao • Quá phức tạp: Variance cao", 21, WHITE), 7.0),
        ).arrange(DOWN, buff=0.22).move_to(b1)

        b2 = card(7.6, 2.9, MINT).move_to([0, -1.6, 0])
        t2 = VGroup(
            fit(T("Điều chuẩn (Regularization) kiểm soát Variance:", 23, MINT), 7.0),
            fit(T("• Ridge (L₂): co mọi trọng số về gần 0", 22, SKY), 7.0),
            fit(T("• Lasso (L₁): ép trọng số dư thừa về đúng 0!", 22, PINK), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b2)

        self.play(FadeIn(b1), FadeIn(t1), run_time=1.0)
        self.play(FadeIn(b2), FadeIn(t2), run_time=1.0)
        end_vo("s1", t0, 11.0)
        save_tl()
