"""CLI shim keeping persisted fixture classes in their stable import module."""

from tests.performance.query_workload import main


if __name__ == "__main__":
    main()
