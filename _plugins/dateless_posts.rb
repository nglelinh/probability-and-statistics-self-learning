# frozen_string_literal: true

# Lesson files in this course are stored as 01_Title.md inside _posts/,
# without Jekyll's required YYYY-MM-DD- prefix. Core PostReader skips
# those files, so both Vietnamese and English lessons would never render.
# Ingest them as posts so chapter pages, search, and language switching work.

require "set"

module Jekyll
  class DatelessPostsReader
    def initialize(site)
      @site = site
    end

    def read!
      existing = @site.posts.docs.map(&:path).to_set
      dated = Document::DATE_FILENAME_MATCHER
      dateless = Document::DATELESS_FILENAME_MATCHER

      Dir.glob(@site.in_source_dir("**/_posts/*.{md,markdown,html}")).each do |path|
        next if existing.include?(path)

        basename = File.basename(path)
        next if dated.match?(basename)
        next unless dateless.match?(basename)

        doc = Document.new(path, :site => @site, :collection => @site.posts)
        doc.read
        doc.data["date"] ||= Time.utc(2021, 1, 1)
        next unless @site.publisher.publish?(doc)

        @site.posts.docs << doc
        existing << path
      end

      @site.posts.docs.sort!
      @site.instance_variable_set(:@post_attr_hash, {})
    end
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  Jekyll::DatelessPostsReader.new(site).read!
end
