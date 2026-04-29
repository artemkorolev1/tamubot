"""Static mapping of degree programs to relevant course ID prefixes.

Used by ``rag_input_builder`` to optionally pre-populate ``course_ids``
when the student's program is known but the query lacks explicit course references.
"""

from __future__ import annotations

PROGRAM_COURSES: dict[str, list[str]] = {
    "Computer Science BS": [
        "CSCE 121",
        "CSCE 221",
        "CSCE 222",
        "CSCE 312",
        "CSCE 313",
        "CSCE 314",
        "CSCE 315",
        "CSCE 410",
        "CSCE 411",
        "CSCE 420",
        "CSCE 431",
        "CSCE 462",
        "CSCE 481",
    ],
    "Computer Engineering BS": [
        "CSCE 121",
        "CSCE 221",
        "CSCE 222",
        "CSCE 312",
        "CSCE 313",
        "ECEN 214",
        "ECEN 248",
        "ECEN 325",
        "ECEN 350",
        "ECEN 454",
    ],
    "Industrial & Systems Engineering BS": [
        "ISEN 210",
        "ISEN 302",
        "ISEN 303",
        "ISEN 305",
        "ISEN 315",
        "ISEN 340",
        "ISEN 350",
        "ISEN 370",
        "ISEN 420",
        "ISEN 430",
    ],
    "Electrical Engineering BS": [
        "ECEN 214",
        "ECEN 248",
        "ECEN 303",
        "ECEN 314",
        "ECEN 325",
        "ECEN 350",
        "ECEN 403",
        "ECEN 454",
        "ECEN 474",
    ],
}
