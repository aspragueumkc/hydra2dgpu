# Python Cache Discipline

- After any structural change to a Python module (signature changes, new return values, new classes, changed imports), **always purge `__pycache__`** before the user restarts QGIS:
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  ```
- QGIS holds modules in memory for the session, and stale `.pyc` files cause invisible failures (wrong arity, missing attributes, silent fallback paths).
- When in doubt, purge before asking the user to restart.
