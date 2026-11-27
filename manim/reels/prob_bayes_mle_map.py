# -*- coding: utf-8 -*-
from common import *

class BayesMLEMAP(Scene):
    def construct(self):
        timeline, end_vo, save_tl = setup_vo(self)
        t0 = self.time; timeline.append(("s1", t0))
        hdr = make_header("FREQUENTIST VS. BAYESIAN", "Từ MLE tới MAP và Điều chuẩn Học máy")
        self.play(FadeIn(hdr), run_time=0.6)

        b1 = card(7.6, 2.6, SKY).move_to([0, 2.0, 0])
        t1 = VGroup(
            fit(T("MLE (Tần suất):  max_w  ∑ log P(xᵢ | w)", 24, SKY), 7.0),
            fit(T("Chỉ tin vào dữ liệu quan sát → Dễ overfit khi ít mẫu!", 22, PINK), 7.0),
        ).arrange(DOWN, buff=0.22).move_to(b1)

        b2 = card(7.6, 3.0, MINT).move_to([0, -1.6, 0])
        t2 = VGroup(
            fit(T("MAP (Bayesian):  max_w [ ∑ log P(xᵢ | w) + log P(w) ]", 22, MINT), 7.0),
            fit(T("• Prior Gaussian N(0, τ²I)  ⇒  Điều chuẩn Ridge (L₂)", 22, WHITE), 7.0),
            fit(T("• Prior Laplace  ⇒  Điều chuẩn Lasso (L₁)!", 22, SUN), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b2)

        self.play(FadeIn(b1), FadeIn(t1), run_time=1.0)
        self.play(FadeIn(b2), FadeIn(t2), run_time=1.0)
        end_vo("s1", t0, 11.0)
        save_tl()
