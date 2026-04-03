import json
import os
import xml.etree.ElementTree as ET

# Coverage
with open("coverage.json") as f:
    cov = json.load(f)
pct = cov["totals"]["percent_covered"]
covered = cov["totals"]["covered_lines"]
total = cov["totals"]["num_statements"]
cov_icon = "✅" if pct >= 80 else "❌"

# Test results
tree = ET.parse("results.xml")
root = tree.getroot()
suite = root.find("testsuite") if root.tag == "testsuites" else root
tests = int(suite.attrib.get("tests", 0))
failures = int(suite.attrib.get("failures", 0))
errors = int(suite.attrib.get("errors", 0))
passed = tests - failures - errors
res_icon = "✅" if failures == 0 and errors == 0 else "❌"

summary = (
    f"## {res_icon} Tests: {passed}/{tests} passed"
    f" &nbsp;|&nbsp; {cov_icon} Coverage: {pct:.1f}%"
    f" ({covered}/{total} lines)\n"
)

with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
    f.write(summary)
