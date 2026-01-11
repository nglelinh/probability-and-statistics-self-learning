---
layout: post
title: "Initial Settings"
chapter: home
order: 1
owner: kyeongminwoo
---

The contents of this repository are hosted as a [Github Blog](<https://convex-optimization-for-all.github.io/>) using Jekyll.
Therefore, to edit existing content or create new content, you must follow Jekyll's directory structure and content writing conventions.
Additionally, you need to verify that changes are properly reflected in your local environment (your current computer) through a web browser.

We have compiled environment setup instructions for those who are not familiar with GitHub or Jekyll.
If you have difficulties following the guide, please leave an [issue](https://github.com/convex-optimization-for-all/convex-optimization-for-all.github.io/issues) in the repository or contact us via the email below for assistance.

(Kyeongmin Woo, wgm0601@gmail.com)

## 1. Git Installation

All work management for this blog is performed through Git and GitHub. Please visit the website below to install Git.

[https://git-scm.com/downloads](https://git-scm.com/downloads)


## 2. Downloading the Repository

To modify the blog, enter the following command in the terminal to download the blog's source code.

```bash
git clone https://github.com/convex-optimization-for-all/convex-optimization-for-all.github.io.git
```

## 3. Local Hosting

Before applying changed or modified content to the blog, you must verify that the work has been performed as intended through local hosting.
If you merge work content that does not follow Jekyll's required conventions into the repository, the blog hosted on the actual web may not function properly.
For local hosting setup, you can choose between two methods: using a virtual environment (Docker) (Option 1) or directly installing the Jekyll environment locally (Option 2).
After completing the local hosting setup and running the local server, you can check your blog content through the `127.0.0.1:4000` address in your web browser.

### 3-1. (Option 1) Docker Installation

### A. Docker Installation

Using Docker enables local hosting without direct environment installation on your local machine.
Please visit the website below to install Docker.

[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

### B. Local Hosting

Enter the following command in the terminal.

```bash
$ docker-compose up
```

### 3-2. (Option 2) Jekyll Environment Installation

### A. Jekyll and Ruby Package Installation

- [Installing Ruby](<https://jekyllrb.com/docs/installation/>): Jekyll is built with Ruby. Therefore, you need to install Ruby to use Jekyll.
- [Installing Jekyll](<https://jekyllrb.com/docs/>): Once Ruby is installed, enter the cloned repository and install Jekyll.
- [Installing Bundle Gem](<https://jekyllrb.com/docs/>): You need to additionally install Ruby packages required for hosting. Run the following command in the repository's project directory.

```bash
$ bundle install
```

### B. Local Hosting


Enter the following command in the terminal.

```
$ jekyll serve
```

If hosting doesn't work, you can also try the following command.

```
$ bundle exec jekyll serve
```

If both commands don't work, the Jekyll environment has not been properly installed.
