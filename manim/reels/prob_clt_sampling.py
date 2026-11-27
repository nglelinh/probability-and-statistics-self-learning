# -*- coding: utf-8 -*-
from common import *

class CLTSampling(Scene):
    def construct(self):
        timeline, end_vo, save_tl = setup_vo(self)
        t0 = self.time; timeline.append(("s1", t0))
        hdr = make_header("ĐỊNH LÝ GIỚI HẠN TRUNG TÂM (CLT)", "Phân phối bất kỳ → Trung bình mẫu thành Hình Chuông")
        self.play(FadeIn(hdr), run_time=0.6)

        b1 = card(7.6, 2.7, SKY).move_to([0, 1.9, 0])
        t1 = VGroup(
            fit(T("Lấy n biến ngẫu nhiên độc lập cùng phân phối (μ, σ²)", 22, WHITE), 7.0),
            fit(T("Dù tổng thể ban đầu lệch hay phân phối đều,", 22, SKY), 7.0),
            fit(T("khi n đủ lớn (n ≥ 30): X̄ ~ N(μ, σ²/n)!", 25, SUN), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b1)

        b2 = card(7.6, 2.7, MINT).move_to([0, -1.7, 0])
        t2 = VGroup(
            fit(T("Quy tắc Căn bậc hai: SE = σ / √n", 26, MINT), 7.0),
            fit(T("Muốn giảm sai số chuẩn đi 2 lần,", 23, WHITE), 7.0),
            fit(T("bạn phải tăng cỡ mẫu n lên 4 lần!", 24, PINK), 7.0),
        ).arrange(DOWN, buff=0.2).move_to(b2)

        self.play(FadeIn(b1), FadeIn(t1), run_time=1.0)
        self.play(FadeIn(b2), FadeIn(t2), run_time=1.0)
        end_vo("s1", t0, 11.0)
        save_tl()
