"""Launch a single v6b_pilot_job backfill via the running webserver's GraphQL.
Routes through the same code server the daemon is connected to so the queued
runs actually execute."""

import json
import sys

import requests

STEMS = [
    "202611_CSCE_608_600_46648",
    "202611_CSCE_611_600_50668",
    "202611_CSCE_612_600_42640",
    "202611_CSCE_614_600_42743",
    "202611_CSCE_625_600_19180_HP",
    "202611_CSCE_629_600_54978",
    "202611_CSCE_633_600_12367_HP",
    "202611_CSCE_635_600_46646",
    "202611_CSCE_641_600_58437",
    "202611_CSCE_648_600_60319",
]

QUERY = """
mutation Launch($bp: LaunchBackfillParams!) {
  launchPartitionBackfill(backfillParams: $bp) {
    __typename
    ... on LaunchBackfillSuccess { backfillId launchedRunIds }
    ... on PythonError { message stack }
    ... on PartitionSetNotFoundError { message }
    ... on InvalidStepError { invalidStepKey }
  }
}
"""

variables = {
    "bp": {
        "selector": {
            "repositorySelector": {
                "repositoryLocationName": "definitions.py",
                "repositoryName": "__repository__",
            },
            "partitionSetName": "v6b_pilot_job_partition_set",
        },
        "partitionNames": STEMS,
    }
}

r = requests.post(
    "http://localhost:3000/graphql",
    json={"query": QUERY, "variables": variables},
    timeout=30,
)
print(json.dumps(r.json(), indent=2))
sys.exit(0 if r.ok and "errors" not in r.json() else 1)
