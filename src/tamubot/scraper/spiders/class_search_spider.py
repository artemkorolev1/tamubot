import json

import scrapy

from tamubot.scraper.items import ClassSectionItem


class ClassSearchSpider(scrapy.Spider):
    name = "class_search"
    allowed_domains = ["howdyportal.tamu.edu"]
    start_urls = ["https://howdyportal.tamu.edu/uPortal/p/public-class-search-ui.ctf1/max/render.uP"]

    # ── Config ──────────────────────────────────────────────────────────
    # DEPARTMENTS = None means "no department filter" (scrape every subject).
    # Pass `-a all_departments=1` to enable that mode for a one-off broad crawl.
    DEPARTMENTS = {"CSCE", "ISEN", "STAT", "ECEN"}
    GRADUATE_ONLY = True  # course number >= 600
    TARGET_TERMS = {
        "202611",
        "202621",
        "202631",  # Spring / Summer / Fall 2026
    }

    _TERM_SEMESTER = {"11": "Spring", "21": "Summer", "31": "Fall"}

    custom_settings = {
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
    }

    def __init__(self, all_departments=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `-a all_departments=1` (or true/yes) drops the department filter entirely.
        if str(all_departments).lower() in {"1", "true", "yes"}:
            self.DEPARTMENTS = None
            self.logger.info("all_departments mode: no department filter applied")

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def term_code_to_folder(term_code: str) -> str:
        """'202611' → 'Spring_2026'"""
        year = term_code[:4]
        sem_code = term_code[4:]
        sem = ClassSearchSpider._TERM_SEMESTER.get(sem_code, sem_code)
        return f"{sem}_{year}"

    # ── Callbacks ───────────────────────────────────────────────────────

    def parse(self, response):
        yield scrapy.Request(
            url="https://howdyportal.tamu.edu/api/all-terms", callback=self.parse_terms, dont_filter=True
        )

    def parse_terms(self, response):
        terms = json.loads(response.text)
        terms.sort(key=lambda x: x.get("STVTERM_CODE", ""), reverse=True)

        for term in terms:
            term_code = term.get("STVTERM_CODE")
            if term_code not in self.TARGET_TERMS:
                continue
            self.logger.info(f"Queueing term: {term_code} ({term.get('STVTERM_DESC')})")
            yield scrapy.Request(
                url="https://howdyportal.tamu.edu/api/course-sections",
                method="POST",
                body=json.dumps({"termCode": term_code}),
                headers={"Content-Type": "application/json"},
                callback=self.parse_sections,
                meta={"term_code": term_code},
                dont_filter=True,
            )

    def parse_sections(self, response):
        term_code = response.meta["term_code"]
        sections = json.loads(response.text)
        self.logger.info(f"Processing {len(sections)} sections for term {term_code}")

        for sec in sections:
            campus = sec.get("SWV_CLASS_SEARCH_SITE", "")
            if campus != "College Station":
                continue

            subject = sec.get("SWV_CLASS_SEARCH_SUBJECT", "")
            course = sec.get("SWV_CLASS_SEARCH_COURSE", "")

            if self.DEPARTMENTS is not None and subject not in self.DEPARTMENTS:
                continue
            if self.GRADUATE_ONLY and int(course) < 600:
                continue

            item = ClassSectionItem()
            item["term_code"] = term_code
            item["crn"] = sec.get("SWV_CLASS_SEARCH_CRN")
            item["title"] = sec.get("SWV_CLASS_SEARCH_TITLE")
            item["subject"] = subject
            item["course"] = course
            item["section"] = sec.get("SWV_CLASS_SEARCH_SECTION")
            item["instructor"] = sec.get("SWV_CLASS_SEARCH_INSTRCTR_JSON")
            item["raw_data"] = sec

            if sec.get("SWV_CLASS_SEARCH_HAS_SYL_IND") == "Y":
                syllabus_url = (
                    f"https://howdyportal.tamu.edu/api/course-syllabus-pdf?termCode={term_code}&crn={item['crn']}"
                )
                item["file_urls"] = [syllabus_url]

            yield item
