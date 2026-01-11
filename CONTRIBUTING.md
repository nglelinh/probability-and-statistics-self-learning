# Hướng dẫn Đóng góp / Contributing Guide

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án này! / Thank you for your interest in contributing to this project!

## 🌟 Cách đóng góp / How to Contribute

### 1. Báo cáo lỗi / Report Bugs

Nếu bạn tìm thấy lỗi, vui lòng tạo issue với các thông tin sau:
If you find a bug, please create an issue with the following information:

- Mô tả rõ ràng về lỗi / Clear description of the bug
- Các bước để tái hiện / Steps to reproduce
- Kết quả mong đợi / Expected behavior
- Screenshots (nếu có) / Screenshots (if applicable)

### 2. Đề xuất tính năng mới / Suggest New Features

Để đề xuất tính năng mới:
To suggest a new feature:

- Tạo issue với label "enhancement"
- Mô tả rõ tính năng và lý do cần thiết
- Đưa ra ví dụ cụ thể nếu có

### 3. Cải thiện nội dung / Improve Content

#### Sửa lỗi nội dung / Fix Content Errors

1. Fork repository
2. Tạo branch mới: `git checkout -b fix/chapter-XX-typo`
3. Sửa lỗi trong file markdown
4. Commit: `git commit -m "Fix typo in Chapter XX"`
5. Push và tạo Pull Request

#### Thêm nội dung mới / Add New Content

1. Fork repository
2. Tạo branch: `git checkout -b feature/chapter-XX-new-topic`
3. Tạo file mới trong `contents/en/chapterXX/_posts/` và `contents/vi/chapterXX/_posts/`
4. Tuân theo format:

```markdown
---
layout: post
title: Your Title
chapter: 'XX'
order: YY
owner: Your Name
lang: en  # or vi
categories:
- chapterXX
---

Your content here...
```

5. Commit và tạo Pull Request

### 4. Cải thiện code / Improve Code

#### CSS/JavaScript

1. Fork repository
2. Tạo branch: `git checkout -b style/improve-navigation`
3. Chỉnh sửa trong `public/css/` hoặc `public/js/`
4. Test local: `bundle exec jekyll serve`
5. Commit và tạo Pull Request

#### Jekyll Plugins

1. Tạo branch: `git checkout -b feature/new-plugin`
2. Thêm plugin vào `_plugins/`
3. Cập nhật documentation
4. Test kỹ lưỡng
5. Tạo Pull Request

## 📝 Quy tắc viết code / Code Style Guidelines

### Markdown

- Sử dụng tiêu đề phân cấp đúng (H1, H2, H3...)
- Thêm dòng trống giữa các đoạn
- Sử dụng code blocks với syntax highlighting
- Đánh số công thức toán học rõ ràng

### LaTeX

- Sử dụng `$$...$$` cho inline math
- Sử dụng block math cho công thức phức tạp:

```markdown
$$
\begin{align}
f(x) &= x^2 \\
g(x) &= 2x
\end{align}
$$
```

### Code Examples

- Comment rõ ràng
- Sử dụng tên biến có ý nghĩa
- Thêm docstring cho functions
- Đảm bảo code chạy được

```python
def example_function(data, param1, param2=0.01):
    """
    Example function with clear documentation
    
    Args:
        data: Input data
        param1: First parameter
        param2: Second parameter (default: 0.01)
    
    Returns:
        result: Processed result
    """
    result = process_data(data, param1, param2)
    return result
```

## 🔍 Review Process

### Pull Request sẽ được review dựa trên:

1. **Chất lượng nội dung / Content Quality**
   - Chính xác về mặt toán học / Mathematical accuracy
   - Giải thích rõ ràng / Clear explanations
   - Ví dụ phù hợp / Appropriate examples

2. **Code Quality**
   - Tuân theo style guidelines
   - Không có lỗi / No errors
   - Test đầy đủ / Well tested

3. **Documentation**
   - Comment đầy đủ / Well commented
   - README được cập nhật / Updated README
   - Changelog được cập nhật / Updated changelog

4. **Đa ngôn ngữ / Multilingual**
   - Nội dung có trong cả EN và VI / Content in both EN and VI
   - Translation chính xác / Accurate translation

## 🎯 Ưu tiên / Priorities

### High Priority
- Sửa lỗi toán học / Fix mathematical errors
- Sửa lỗi code / Fix code bugs
- Cải thiện accessibility
- Thêm tests

### Medium Priority
- Thêm ví dụ mới / Add new examples
- Cải thiện giải thích / Improve explanations
- Thêm visualizations
- Cải thiện UI/UX

### Low Priority
- Refactoring code
- Tối ưu performance / Performance optimization
- Thêm tính năng mới / Add new features

## 🧪 Testing

### Local Testing

```bash
# Install dependencies
bundle install

# Run Jekyll server
bundle exec jekyll serve

# Open browser
open http://127.0.0.1:4000/your-baseurl/
```

### Kiểm tra / Check:

- [ ] Tất cả pages load được / All pages load correctly
- [ ] Không có broken links
- [ ] Images hiển thị đúng / Images display correctly
- [ ] Math formulas render đúng / Math formulas render correctly
- [ ] Responsive trên mobile / Responsive on mobile
- [ ] Chuyển ngôn ngữ hoạt động / Language switching works
- [ ] Search hoạt động / Search works

## 📚 Tài liệu tham khảo / References

Khi thêm nội dung, vui lòng tham khảo:
When adding content, please refer to:

- Textbooks and academic sources relevant to your course
- Peer-reviewed papers and publications
- Reputable online courses and lectures
- Official documentation and standards

## 💬 Liên hệ / Contact

Nếu có câu hỏi, vui lòng:
If you have questions:

1. Tạo issue với label "question"
2. Tham gia discussions
3. Email: (xem trong _config.yml)

## 🙏 Cảm ơn / Thank You

Mọi đóng góp đều được trân trọng! / Every contribution is appreciated!

Danh sách contributors sẽ được cập nhật trong README.md
Contributors will be listed in README.md

---

**Happy Contributing! 🎉**

