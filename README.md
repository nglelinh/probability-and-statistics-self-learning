# Probability and Statistics Self-Learning

Khóa học **Xác suất & Thống kê / Thống kê tính toán** (Probability & Statistics / Computational Statistics) của **Nguyen Le Linh** (`nglelinh`). Trang được xây trên Jekyll, hỗ trợ hai ngôn ngữ, và xuất bản qua GitHub Pages.

**Trang trực tuyến:** [https://nglelinh.github.io/probability-and-statistics-self-learning/](https://nglelinh.github.io/probability-and-statistics-self-learning/)

**Mã nguồn:** [github.com/nglelinh/probability-and-statistics-self-learning](https://github.com/nglelinh/probability-and-statistics-self-learning)

---

## Trạng thái đa ngôn ngữ

| Ngôn ngữ | Vai trò | Phạm vi hiện tại |
|----------|---------|------------------|
| **Tiếng Việt (`contents/vi`)** | Bản khóa học đầy đủ | Chương 01–08 (~28 bài) |
| **English (`contents/en`)** | Bản đối chiếu đang mở | Chương 01 đã có đủ 7 bài; Chương 02–08 chưa dịch |

Chuyển ngôn ngữ trên một bài Chapter 01 tiếng Anh sẽ tìm bài tiếng Việt **cùng `chapter` và `order`**. Các bài EN dùng cùng `chapter: "01"` và cùng `order` với bản VI tương ứng.

---

## Nội dung khóa học

Khóa học đi từ tư duy thống kê tính toán (mô phỏng, resampling, lựa chọn mô hình) sang xác suất, ước lượng, kiểm định, hồi quy và ANOVA. Chương 01 mở đầu bằng mô phỏng Monte Carlo và Luật số lớn, rồi ôn giải tích cần thiết, supervised learning, cross-validation và tiêu chí chọn mô hình.

**Sách chính:** Sheldon Ross, *A First Course in Probability*, Pearson, 2012.

**Tham khảo:** *OpenIntro Statistics*; Allen B. Downey, *Think Stats*; Mario F. Triola, *Elementary Statistics*.

---

## Chạy local / build

Cần Ruby và Bundler.

```bash
bundle install
bundle exec jekyll serve
```

Mở [http://127.0.0.1:4000/probability-and-statistics-self-learning/](http://127.0.0.1:4000/probability-and-statistics-self-learning/).

Chỉ build (không serve):

```bash
bundle exec jekyll build
```

Output nằm trong `_site/` (thư mục này không commit).

---

## Cấu trúc nội dung

```
contents/
├── en/
│   ├── chapter00/          # Ghi chú English track (WIP)
│   └── chapter01/          # Bản tiếng Anh Chương 01
│       └── _posts/
│           ├── 01_Introduction_to_Statistical_Problems.md
│           ├── 02_Math_Reference_Continuity.md
│           ├── 03_Math_Reference_Derivatives.md
│           ├── 04_Supervised_Learning.md
│           ├── 05_ML_Workflow.md
│           ├── 06_Cross_Validation.md
│           └── 07_Model_Selection.md
└── vi/
    ├── chapter01/ … chapter08/   # Bản tiếng Việt đầy đủ
```

Front matter bài giảng:

```markdown
---
layout: post
title: Lesson title
chapter: "01"
order: 1
owner: nglelinh
lang: en
categories:
- chapter01
lesson_type: required
---
```

Công thức dùng MathJax với `$$...$$`.

---

## Deploy

GitHub Actions (`.github/workflows/jekyll.yml`) build Jekyll 4.x (cần custom plugin đa ngôn ngữ) và deploy GitHub Pages khi push `main`. Trong **Settings → Pages**, Source phải là **GitHub Actions**.

Xem thêm [DEPLOYMENT.md](./DEPLOYMENT.md) và [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Giấy phép và credit

Nội dung khóa học dùng cho mục đích giáo dục. Giao diện dựa trên theme [Lanyon](https://github.com/poole/lanyon) (Mark Otto), Jekyll, và MathJax.

**Giảng viên:** Nguyen Le Linh — nglelinh@gmail.com
