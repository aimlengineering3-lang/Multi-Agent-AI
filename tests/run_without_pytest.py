"""Tiny fallback runner: `python -m tests.run_without_pytest` (pytest is preferred)."""
import importlib, inspect, pkgutil, sys, traceback
import tests

failed = passed = 0
for m in pkgutil.iter_modules(tests.__path__):
    if not m.name.startswith("test_"):
        continue
    mod = importlib.import_module(f"tests.{m.name}")
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and fn.__module__ == mod.__name__:
            try:
                fn(); passed += 1
            except Exception:
                failed += 1; print(f"FAIL {m.name}.{name}"); traceback.print_exc()
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
