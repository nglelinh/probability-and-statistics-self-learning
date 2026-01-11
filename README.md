# Jekyll Course Template - Multilingual Support

Template để tạo trang web khóa học với hỗ trợ đa ngôn ngữ (Tiếng Anh và Tiếng Việt), được xây dựng trên Jekyll và GitHub Pages.

## 🎯 Tính năng chính

- ✅ Hỗ trợ đa ngôn ngữ (English/Vietnamese)
- ✅ Cấu trúc nội dung theo chương (chapters)
- ✅ Tìm kiếm nội dung
- ✅ Responsive design
- ✅ MathJax support cho công thức toán học
- ✅ Tự động deploy lên GitHub Pages
- ✅ Custom Jekyll plugins
- ✅ Exam/Quiz templates

## 🚀 Cách sử dụng template này

> 📖 **Xem hướng dẫn chi tiết trong [SETUP.md](./SETUP.md)**

### Bước 1: Tạo repository mới từ template

1. Nhấn nút "Use this template" trên GitHub
2. Đặt tên cho repository mới của bạn (ví dụ: `machine-learning-course`)
3. Chọn Public hoặc Private
4. Nhấn "Create repository from template"

### Bước 2: Cấu hình cơ bản

Chỉnh sửa file `_config.yml`:

```yaml
# Setup
title:               "Tên Khóa Học Của Bạn"
description:         'Mô tả khóa học'
url:                 https://your-username.github.io
baseurl:             '/your-repo-name'
imgurl:              https://your-username.github.io/your-repo-name/img

# About/contact
author:
  name:              Tên Giảng Viên
  email:             email@example.com
```

### Bước 3: Cấu hình GitHub Pages

1. Vào **Settings > Pages**
2. Chọn **Source**: "GitHub Actions"
3. Workflow sẽ tự động chạy khi bạn push code

### Bước 4: Tùy chỉnh nội dung

#### Trang chủ

Chỉnh sửa các file trong `home/_posts/`:
- `21-01-20-introduction.md` - Giới thiệu khóa học
- `21-01-20-contents.md` - Nội dung khóa học
- `21-02-03-makers.md` - Thông tin giảng viên

#### Nội dung các chương

Tạo nội dung trong `contents/en/` và `contents/vi/`:

```
contents/
├── en/
│   ├── chapter00/
│   │   └── _posts/
│   │       └── 21-01-01-00_Introduction.md
│   ├── chapter01/
│   │   └── _posts/
│   │       └── 21-01-07-01_00_Introduction.md
│   └── ...
└── vi/
    ├── chapter00/
    │   └── _posts/
    │       └── 21-01-01-00_Gioi_thieu.md
    └── ...
```

**Format file bài giảng:**

```markdown
---
layout: post
title: Tiêu đề bài giảng
chapter: '00'
order: 1
owner: Tên tác giả
lang: en  # hoặc vi
categories:
- chapter00
---

Nội dung bài giảng ở đây...

Sử dụng LaTeX cho công thức: $$f(x) = x^2$$
```

## 📁 Cấu trúc thư mục

```
.
├── _config.yml              # Cấu hình Jekyll
├── _includes/               # Các component tái sử dụng
│   ├── head.html
│   └── sidebar.html
├── _layouts/                # Layouts cho pages
│   ├── default.html
│   ├── page.html
│   └── post.html
├── _plugins/                # Custom Jekyll plugins
│   ├── multilang.rb         # Hỗ trợ đa ngôn ngữ
│   ├── multilang_post_url.rb
│   ├── redirect_generator.rb
│   └── search_generator.rb
├── contents/                # Nội dung khóa học
│   ├── en/                  # Nội dung tiếng Anh
│   │   ├── chapter00/
│   │   ├── chapter01/
│   │   └── ...
│   └── vi/                  # Nội dung tiếng Việt
│       ├── chapter00/
│       ├── chapter01/
│       └── ...
├── home/                    # Trang chủ
│   └── _posts/
├── img/                     # Hình ảnh
│   └── chapter_img/
├── public/                  # CSS, JS, assets
│   ├── css/
│   ├── js/
│   └── logo.png
├── Gemfile                  # Ruby dependencies
├── index.html               # Trang chủ
└── README.md                # Hướng dẫn dự án
```

## 🛠️ Development

### Cài đặt môi trường

```bash
# Cài đặt Ruby dependencies
bundle install

# Chạy Jekyll local server
bundle exec jekyll serve

# Truy cập tại
http://127.0.0.1:4000/your-baseurl/
```

### Thêm chương mới

1. Tạo thư mục mới trong `contents/en/chapterXX/` và `contents/vi/chapterXX/`
2. Tạo thư mục `_posts/` bên trong
3. Thêm file markdown với format: `YYYY-MM-DD-title.md`
4. Đảm bảo front matter có đầy đủ thông tin

### Thêm hình ảnh

1. Đặt hình ảnh vào `img/chapter_img/`
2. Tham chiếu trong markdown:

```markdown
![Alt text]({{ site.imgurl }}/chapter_img/your-image.png)
```

## 🎨 Tùy chỉnh giao diện

### CSS

Chỉnh sửa các file trong `public/css/`:
- `lanyon.css` - Layout chính
- `poole.css` - Base styles
- `syntax.css` - Code highlighting

### JavaScript

Chỉnh sửa các file trong `public/js/`:
- `script.js` - Chức năng chung
- `multilang.js` - Xử lý đa ngôn ngữ
- `search.js` - Tìm kiếm

## 📝 Viết nội dung với LaTeX

Template hỗ trợ MathJax để hiển thị công thức toán học:

```markdown
Inline math: $$f(x) = x^2$$

Display math:
$$
\min_{x \in \mathbb{R}^n} f(x)
$$
```

## 🔍 Tìm kiếm

Tìm kiếm được tạo tự động từ plugin `search_generator.rb`:
- `search-index.json` - Index tiếng Anh
- `search-index-vi.json` - Index tiếng Việt

## 🌐 Đa ngôn ngữ

### Sử dụng translation tags

Trong template:

```liquid
{% t home %}           <!-- Hiển thị "Home" hoặc "Trang chủ" -->
{% language_switch %}  <!-- Nút chuyển ngôn ngữ -->
```

### Cấu hình translations

Trong `_config.yml`:

```yaml
t:
  en:
    title: "Course Title"
    home: "Home"
    chapters: "Chapters"
  vi:
    title: "Tiêu đề Khóa học"
    home: "Trang chủ"
    chapters: "Các chương"
```

## 📚 Tạo đề thi

Để tạo đề thi hoặc bài tập:

1. Tạo file HTML mới trong thư mục gốc
2. Sử dụng cấu trúc HTML cơ bản với MathJax
3. File sẽ tự động được build và deploy

## 🤝 Đóng góp

Để đóng góp vào khóa học:

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/new-chapter`
3. Commit changes: `git commit -am 'Add new chapter'`
4. Push to branch: `git push origin feature/new-chapter`
5. Tạo Pull Request

## 📄 License

Template này sử dụng theme Lanyon và được phát triển cho mục đích giáo dục.

## 🙏 Credits

- **Theme**: [Lanyon](https://github.com/poole/lanyon) by Mark Otto
- **Jekyll**: Static site generator
- **MathJax**: Mathematical formula rendering

## 📞 Hỗ trợ

Nếu có vấn đề, vui lòng:
1. Kiểm tra [Issues](../../issues)
2. Tạo issue mới nếu chưa có
3. Liên hệ qua email trong `_config.yml`

---

**Happy Teaching! 🎓**
