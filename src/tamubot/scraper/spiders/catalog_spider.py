import os
from urllib.parse import urlparse

import scrapy

from tamubot.scraper.items import PdfManifestItem, TamuPageItem
from tamubot.scraper.utils.cleaner import clean_html_content

# Graduate catalog paths per department
DEPT_PATHS = {
    "CSCE": [
        "/graduate/colleges-schools-interdisciplinary/engineering/computer-science/",
        "/graduate/course-descriptions/csce/",
    ],
    "ISEN": [
        "/graduate/colleges-schools-interdisciplinary/engineering/industrial-systems/",
        "/graduate/course-descriptions/isen/",
    ],
    "STAT": [
        "/graduate/colleges-schools-interdisciplinary/science/statistics/",
        "/graduate/colleges-schools-interdisciplinary/arts-and-sciences/statistics/",
        "/graduate/course-descriptions/stat/",
    ],
    "ECEN": [
        "/graduate/colleges-schools-interdisciplinary/engineering/electrical-computer/",
        "/graduate/course-descriptions/ecen/",
    ],
}


class CatalogSpider(scrapy.Spider):
    name = "catalog"
    allowed_domains = ["catalog.tamu.edu"]

    def __init__(self, departments=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_urls: set[str] = set()

        # If departments given (comma-separated), restrict to those paths
        if departments:
            self.dept_list = [d.strip().upper() for d in departments.split(",")]
        else:
            self.dept_list = []

        # Build start_urls and allowed path prefixes
        if self.dept_list:
            self.start_urls = []
            self._allowed_prefixes: list[str] = []
            for dept in self.dept_list:
                paths = DEPT_PATHS.get(dept, [])
                if not paths:
                    self.logger.warning(f"No catalog paths configured for {dept}")
                    continue
                for p in paths:
                    self.start_urls.append(f"https://catalog.tamu.edu{p}")
                    self._allowed_prefixes.append(p)
        else:
            # Full-site crawl (original behaviour)
            self.start_urls = ["https://catalog.tamu.edu/"]
            self._allowed_prefixes = []

        # Progress log for resumability
        self.log_path = os.path.join("tamu_data", "scraper", "logs", "progress_log.txt")
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.visited_urls.add(line.strip())

        self.logger.info(f"Loaded {len(self.visited_urls)} visited URLs from history.")
        self.logger.info(f"Departments: {self.dept_list or 'ALL'}")
        self.logger.info(f"Start URLs: {self.start_urls}")

    # -- helpers ----------------------------------------------------------

    def _url_in_scope(self, url: str) -> bool:
        """Return True if url is within allowed crawl scope."""
        parsed = urlparse(url)
        if parsed.netloc != "catalog.tamu.edu":
            return False
        if not self._allowed_prefixes:
            return True  # full-site mode
        return any(parsed.path.startswith(pfx) for pfx in self._allowed_prefixes)

    @staticmethod
    def dept_for_url(url: str) -> str | None:
        """Return the department code that owns *url*, or None."""
        parsed = urlparse(url)
        for dept, paths in DEPT_PATHS.items():
            if any(parsed.path.startswith(p) for p in paths):
                return dept
        return None

    # -- scrapy callbacks -------------------------------------------------

    def parse(self, response):
        if response.url not in self.visited_urls:
            cleaned_text = clean_html_content(response.body)

            page_item = TamuPageItem()
            page_item["url"] = response.url
            page_item["title"] = response.css("title::text").get(default="").strip()
            page_item["content"] = cleaned_text
            yield page_item

            self.visited_urls.add(response.url)

        # Follow internal links within scope
        for link in response.css("a"):
            href = link.attrib.get("href")
            if not href:
                continue

            absolute_url = response.urljoin(href)
            clean_url = absolute_url.split("#")[0]
            parsed_url = urlparse(absolute_url)

            # Check for PDF
            if parsed_url.path.lower().endswith(".pdf"):
                pdf_item = PdfManifestItem()
                pdf_item["url"] = absolute_url
                pdf_item["program_name"] = link.css("::text").get(default="").strip()
                pdf_item["department"] = response.css("title::text").get(default="").strip()
                yield pdf_item
                continue

            if self._url_in_scope(clean_url) and clean_url not in self.visited_urls:
                yield scrapy.Request(clean_url, callback=self.parse)
