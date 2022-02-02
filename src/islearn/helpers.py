from typing import Any, Callable, TypeVar, Optional

T = TypeVar("T")


def e_assert_present(expression: T, message: Optional[str] = None) -> T:
    return e_assert(expression, lambda e: e is not None, message)


def e_assert(expression: T, assertion: Callable[[T], bool], message: Optional[str] = None) -> T:
    assert assertion(expression), message or ""
    return expression
