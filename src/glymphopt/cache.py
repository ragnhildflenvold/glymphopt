from typing import Any, Callable, Generic, TypeVar, Optional
import numpy as np

T = TypeVar("T")


class CacheObject(Generic[T]):
    def __init__(self):
        self.lastx = {}
        self.val: Optional[T] = None


def cache_fetch[T](
    cache: CacheObject[T],
    func: Callable[..., T],
    cache_kwargs: dict[str, Any],
    **funkwargs: Any,
) -> T:
    is_old_kwargs = all(
        [
            key in cache.lastx and np.array_equal(val, cache.lastx[key])
            for key, val in cache_kwargs.items()
        ]
    )
    if (not is_old_kwargs) or (cache.val is None):
        cache.val = func(**funkwargs)
        for key, val in cache_kwargs.items():
            cache.lastx[key] = val
    return cache.val
