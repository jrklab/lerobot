# Robot scripts and run logs live under runs/ (gitignored); pytest must never import them:
# importing a robot script runs it, and two named *_test.py moved the robot during test runs
# on Oct 27. The tests are in tests/.
collect_ignore_glob = ["runs/*", "demos/*"]
