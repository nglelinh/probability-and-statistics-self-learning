# Setup Guide - Hướng dẫn Cài đặt

Hướng dẫn chi tiết để thiết lập template này cho khóa học của bạn.

## 🚀 Bước 1: Tạo Repository từ Template

1. Nhấn nút **"Use this template"** trên GitHub
2. Đặt tên repository mới (ví dụ: `machine-learning-course`)
3. Chọn Public hoặc Private
4. Nhấn **"Create repository from template"**

## ⚙️ Bước 2: Cấu hình Cơ bản

### 2.1. Cập nhật `_config.yml`

Mở file `_config.yml` và thay đổi các thông tin sau:

```yaml
# Setup
title:               "Your Course Title"          # Tên khóa học của bạn
description:         'Your Course Description'    # Mô tả khóa học
url:                 https://your-username.github.io
baseurl:             '/your-repo-name'            # Tên repository của bạn
imgurl:              https://your-username.github.io/your-repo-name/img

# Language-specific configurations
t:
  en:
    title: "Your Course Title"
    description: "Your Course Description"
  vi:
    title: "Tiêu đề Khóa học của bạn"
    description: "Mô tả khóa học của bạn"

# About/contact
author:
  name:              Your Name                    # Tên của bạn
  email:             your.email@example.com       # Email của bạn
```

### 2.2. Cập nhật GitHub Link

Mở file `_layouts/default.html` và tìm dòng 24:

```html
<a class="github-logo__wrapper" target="_blank" href="https://github.com/your-username/your-repo-name" title="Github">
```

Thay `your-username/your-repo-name` bằng username và tên repository thực tế của bạn.

### 2.3. Cập nhật Thông tin Tác giả

Chỉnh sửa file `AUTHORS.md` với thông tin của bạn:

```markdown
## Course Instructor

**[Your Name]** - [Your Institution]
Brief bio and role in the course.

Email: your.email@example.com
Website: [Your Website](https://your-website.com)
```

## 📝 Bước 3: Tùy chỉnh Nội dung

### 3.1. Trang chủ

Chỉnh sửa các file trong `home/_posts/`:

- `21-01-20-introduction.md` - Giới thiệu khóa học
- `21-01-20-contents.md` - Nội dung và mục tiêu
- `21-02-03-makers.md` - Thông tin giảng viên

### 3.2. Nội dung Bài giảng

Tạo nội dung trong `contents/en/` (tiếng Anh) và `contents/vi/` (tiếng Việt):

**Cấu trúc thư mục:**

```
contents/
├── en/
│   └── chapter01/
│       └── _posts/
│           └── 21-01-07-01_00_Introduction.md
└── vi/
    └── chapter01/
        └── _posts/
            └── 21-01-07-01_00_Gioi_thieu.md
```

**Format file bài giảng:**

```markdown
---
layout: post
title: Lesson Title
chapter: '01'
order: 1
owner: Your Name
lang: en  # hoặc vi
categories:
- chapter01
---

# Lesson Content

Your content here...

Use LaTeX for formulas: $$f(x) = x^2$$
```

## 🌐 Bước 4: Cấu hình GitHub Pages

1. Vào **Settings > Pages** trong repository của bạn
2. Chọn **Source**: "GitHub Actions"
3. Workflow sẽ tự động chạy khi bạn push code

## 🧪 Bước 5: Test Local

```bash
# Cài đặt dependencies
bundle install

# Chạy Jekyll server
bundle exec jekyll serve

# Mở browser tại
http://127.0.0.1:4000/your-repo-name/
```

## 📋 Checklist

Trước khi deploy, kiểm tra:

- [ ] Đã cập nhật `_config.yml` với thông tin đúng
- [ ] Đã thay GitHub link trong `_layouts/default.html`
- [ ] Đã cập nhật `AUTHORS.md`
- [ ] Đã chỉnh sửa nội dung trang chủ
- [ ] Đã test local thành công
- [ ] GitHub Pages đã được cấu hình
- [ ] Tất cả links hoạt động đúng
- [ ] Images hiển thị đúng
- [ ] Math formulas render đúng
- [ ] Language switching hoạt động
- [ ] Search hoạt động

## 🎨 Tùy chỉnh Nâng cao

### Logo và Favicon

Thay thế các file trong `public/`:
- `logo.png` - Logo của khóa học
- `convex-logo-144x144.png` - Favicon

### CSS Styling

Chỉnh sửa các file trong `public/css/`:
- `lanyon.css` - Layout chính
- `poole.css` - Base styles
- `syntax.css` - Code highlighting

### JavaScript

Chỉnh sửa các file trong `public/js/`:
- `script.js` - Chức năng chung
- `multilang.js` - Xử lý đa ngôn ngữ
- `search.js` - Tìm kiếm

## 🆘 Troubleshooting

### Site không build được

1. Kiểm tra syntax trong `_config.yml`
2. Đảm bảo tất cả front matter trong markdown files đúng format
3. Xem logs trong GitHub Actions

### Images không hiển thị

1. Kiểm tra đường dẫn: `{{ site.imgurl }}/chapter_img/your-image.png`
2. Đảm bảo image file tồn tại trong `img/chapter_img/`
3. Kiểm tra `imgurl` trong `_config.yml`

### Math formulas không render

1. Kiểm tra MathJax script trong `_includes/head.html`
2. Sử dụng `$$...$$` cho inline math
3. Sử dụng block format cho công thức phức tạp

### Search không hoạt động

1. Kiểm tra `search-index.json` và `search-index-vi.json` được generate
2. Xem console log trong browser
3. Đảm bảo Lunr.js được load

## 📚 Tài liệu Tham khảo

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)
- [MathJax Documentation](https://docs.mathjax.org/)

## 💬 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra [Issues](../../issues) có vấn đề tương tự không
2. Tạo issue mới với thông tin chi tiết
3. Liên hệ qua email trong `_config.yml`

---

**Good luck with your course! 🎓**

