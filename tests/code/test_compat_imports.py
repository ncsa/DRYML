from __future__ import annotations


def test_old_and_new_imports_work():
    import dryml.code as code
    from dryml.code.ast_tools import AccessCollector, collect_accesses_from_source
    from dryml.code.callable_info import CallableInfo, analyze_callable
    from dryml.code.source import SourceInfo, get_source_info

    assert callable(code.analyze)
    assert code.CodeTargetSpec is not None
    assert CallableInfo is not None
    assert callable(analyze_callable)
    assert SourceInfo is not None
    assert callable(get_source_info)
    assert AccessCollector is not None
    assert callable(collect_accesses_from_source)
