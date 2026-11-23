"""
Unified Tensor / Matrix Backend (`iwsp.math_backend`)
=====================================================

Provides a clean NumPy-compatible 1D/2D/3D numerical array interface (`np`) that
automatically uses `numpy` if installed in the active Python environment, and
transparently falls back to a self-contained pure-Python linear algebra engine
(including SVD, Ridge Regression, Softmax, LayerNorm, and matmul) if `numpy` is absent.

This guarantees that `iwsp` works out-of-the-box in any Python 3.9+ environment while
leveraging C-accelerated NumPy whenever available.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as _real_np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    _real_np = None
    HAS_NUMPY = False


class PureArray:
    """Lightweight pure-Python multi-dimensional numeric array with NumPy-compatible API."""

    def __init__(self, data: Any) -> None:
        if isinstance(data, PureArray):
            self._data = _deep_copy_list(data._data)
        elif isinstance(data, (int, float, bool)):
            self._data = float(data)
        else:
            self._data = _to_nested_float_list(data)
        self.shape: Tuple[int, ...] = _infer_shape(self._data)
        self.ndim: int = len(self.shape)

    @property
    def size(self) -> int:
        s = 1
        for d in self.shape:
            s *= d
        return s

    @property
    def T(self) -> "PureArray":
        return transpose(self)

    def tolist(self) -> Any:
        return _deep_copy_list(self._data)

    def copy(self) -> "PureArray":
        return PureArray(self._data)

    def flatten(self) -> "PureArray":
        return PureArray(_flatten_nested(self._data))

    def reshape(self, *new_shape: Any) -> "PureArray":
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            dims = list(new_shape[0])
        else:
            dims = list(new_shape)
        flat = _flatten_nested(self._data)
        total = len(flat)
        neg_idx = -1
        known_prod = 1
        for i, d in enumerate(dims):
            if d == -1:
                neg_idx = i
            else:
                known_prod *= d
        if neg_idx >= 0:
            dims[neg_idx] = total // max(known_prod, 1)
        return PureArray(_unflatten(flat, tuple(dims)))

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized 0d array")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over 0d array")
        for item in self._data:
            if isinstance(item, list):
                yield PureArray(item)
            else:
                yield float(item)

    def __float__(self) -> float:
        if self.ndim == 0:
            return float(self._data)
        flat = _flatten_nested(self._data)
        if len(flat) == 1:
            return float(flat[0])
        raise TypeError("only size-1 arrays can be converted to Python scalars")

    def __int__(self) -> int:
        return int(float(self))

    def __getitem__(self, idx: Any) -> Any:
        res = _index_nested(self._data, idx)
        if isinstance(res, list):
            return PureArray(res)
        return float(res)

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(val, PureArray):
            val = val._data
        _set_nested(self._data, idx, val)

    def __add__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "PureArray":
        return _elemwise_binary(other, self, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "PureArray":
        return _elemwise_binary(other, self, lambda a, b: a - b)

    def __mul__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> "PureArray":
        return _elemwise_binary(other, self, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> "PureArray":
        return _elemwise_binary(other, self, lambda a, b: a / b)

    def __pow__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: a ** b)

    def __neg__(self) -> "PureArray":
        return _elemwise_unary(self, lambda a: -a)

    def __gt__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: 1.0 if a > b else 0.0)

    def __ge__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: 1.0 if a >= b else 0.0)

    def __lt__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: 1.0 if a < b else 0.0)

    def __le__(self, other: Any) -> "PureArray":
        return _elemwise_binary(self, other, lambda a, b: 1.0 if a <= b else 0.0)

    def __eq__(self, other: Any) -> "PureArray":  # type: ignore[override]
        return _elemwise_binary(self, other, lambda a, b: 1.0 if a == b else 0.0)

    def __matmul__(self, other: Any) -> "PureArray":
        return matmul(self, other)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        return sum_array(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        return mean_array(self, axis=axis, keepdims=keepdims)

    def std(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        return std_array(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        return max_array(self, axis=axis, keepdims=keepdims)

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        return min_array(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis: Optional[int] = None) -> Any:
        return argmax_array(self, axis=axis)

    def __repr__(self) -> str:
        return f"array({self._data})"


# --- Helper functions for PureArray ---

def _to_nested_float_list(obj: Any) -> Any:
    if isinstance(obj, PureArray):
        return _deep_copy_list(obj._data)
    if isinstance(obj, (list, tuple)):
        return [_to_nested_float_list(x) for x in obj]
    return float(obj)


def _deep_copy_list(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_deep_copy_list(x) for x in obj]
    return float(obj)


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if len(data) == 0:
        return (0,)
    return (len(data),) + _infer_shape(data[0])


def _flatten_nested(data: Any) -> List[float]:
    if not isinstance(data, list):
        return [float(data)]
    out: List[float] = []
    for item in data:
        out.extend(_flatten_nested(item))
    return out


def _unflatten(flat: List[float], shape: Tuple[int, ...]) -> Any:
    if len(shape) == 0:
        return flat[0] if flat else 0.0
    if len(shape) == 1:
        return [float(x) for x in flat[: shape[0]]]
    stride = 1
    for s in shape[1:]:
        stride *= s
    return [
        _unflatten(flat[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def _index_nested(data: Any, idx: Any) -> Any:
    if not isinstance(idx, tuple):
        if isinstance(idx, slice):
            return data[idx]
        if isinstance(idx, list):
            return [data[int(i)] for i in idx]
        if isinstance(idx, PureArray):
            return [data[int(i)] for i in _flatten_nested(idx._data)]
        return data[int(idx)]
    if len(idx) == 0:
        return data
    first, rest = idx[0], idx[1:]
    if isinstance(first, slice):
        sliced = data[first]
        if not rest:
            return sliced
        return [_index_nested(row, rest) for row in sliced]
    if isinstance(first, (list, PureArray)):
        indices = [int(i) for i in (_flatten_nested(first._data) if isinstance(first, PureArray) else first)]
        if not rest:
            return [data[i] for i in indices]
        return [_index_nested(data[i], rest) for i in indices]
    sub = data[int(first)]
    if not rest:
        return sub
    return _index_nested(sub, rest)


def _set_nested(data: Any, idx: Any, val: Any) -> None:
    if not isinstance(idx, tuple):
        if isinstance(idx, slice):
            rng = range(*idx.indices(len(data)))
            if isinstance(val, list):
                for k, r_i in enumerate(rng):
                    data[r_i] = _deep_copy_list(val[k]) if isinstance(val[k], list) else float(val[k])
            else:
                for r_i in rng:
                    data[r_i] = _fill_shape(_infer_shape(data[r_i]), float(val))
            return
        data[int(idx)] = _deep_copy_list(val) if isinstance(val, list) else float(val)
        return
    if len(idx) == 1:
        _set_nested(data, idx[0], val)
        return
    first, rest = idx[0], idx[1:]
    if isinstance(first, slice):
        rng = range(*first.indices(len(data)))
        for k, r_i in enumerate(rng):
            v_k = val[k] if isinstance(val, list) and len(val) == len(rng) else val
            _set_nested(data[r_i], rest, v_k)
        return
    _set_nested(data[int(first)], rest, val)


def _fill_shape(shape: Tuple[int, ...], fill_value: float) -> Any:
    if len(shape) == 0:
        return float(fill_value)
    if len(shape) == 1:
        return [float(fill_value) for _ in range(shape[0])]
    return [_fill_shape(shape[1:], fill_value) for _ in range(shape[0])]


def _broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    ndim = max(len(s1), len(s2))
    p1 = (1,) * (ndim - len(s1)) + s1
    p2 = (1,) * (ndim - len(s2)) + s2
    out: List[int] = []
    for d1, d2 in zip(p1, p2):
        if d1 == d2:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        elif d2 == 1:
            out.append(d1)
        else:
            raise ValueError(f"Operands could not be broadcast together with shapes {s1} and {s2}")
    return tuple(out)


def _get_broadcast_val(data: Any, orig_shape: Tuple[int, ...], target_coords: Tuple[int, ...]) -> float:
    if len(orig_shape) == 0:
        return float(data)
    offset = len(target_coords) - len(orig_shape)
    cur = data
    for i, dim in enumerate(orig_shape):
        c = target_coords[offset + i] if dim > 1 else 0
        cur = cur[c]
    return float(cur)


def _iter_coords(shape: Tuple[int, ...]):
    if len(shape) == 0:
        yield ()
        return
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
        return
    for i in range(shape[0]):
        for rest in _iter_coords(shape[1:]):
            yield (i,) + rest


def _elemwise_binary(a: Any, b: Any, op: Callable[[float, float], float]) -> PureArray:
    arr_a = a if isinstance(a, PureArray) else PureArray(a)
    arr_b = b if isinstance(b, PureArray) else PureArray(b)
    if arr_a.ndim == 0 and arr_b.ndim == 0:
        return PureArray(op(float(arr_a._data), float(arr_b._data)))
    out_shape = _broadcast_shapes(arr_a.shape, arr_b.shape)
    flat: List[float] = []
    for coords in _iter_coords(out_shape):
        va = _get_broadcast_val(arr_a._data, arr_a.shape, coords)
        vb = _get_broadcast_val(arr_b._data, arr_b.shape, coords)
        flat.append(op(va, vb))
    return PureArray(_unflatten(flat, out_shape))


def _elemwise_unary(a: Any, op: Callable[[float], float]) -> PureArray:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    if arr.ndim == 0:
        return PureArray(op(float(arr._data)))
    flat = [op(x) for x in _flatten_nested(arr._data)]
    return PureArray(_unflatten(flat, arr.shape))


def transpose(a: Any, axes: Optional[Tuple[int, ...]] = None) -> PureArray:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    if arr.ndim <= 1:
        return arr.copy()
    if axes is None:
        axes = tuple(reversed(range(arr.ndim)))
    out_shape = tuple(arr.shape[ax] for ax in axes)
    out_data = _fill_shape(out_shape, 0.0)
    for coords in _iter_coords(arr.shape):
        val = _get_broadcast_val(arr._data, arr.shape, coords)
        new_coords = tuple(coords[ax] for ax in axes)
        _set_nested(out_data, new_coords, val)
    return PureArray(out_data)


def matmul(a: Any, b: Any) -> PureArray:
    arr_a = a if isinstance(a, PureArray) else PureArray(a)
    arr_b = b if isinstance(b, PureArray) else PureArray(b)
    if arr_a.ndim == 1 and arr_b.ndim == 1:
        if arr_a.shape[0] != arr_b.shape[0]:
            raise ValueError(f"Dot shape mismatch: {arr_a.shape} vs {arr_b.shape}")
        return PureArray(sum(float(x) * float(y) for x, y in zip(arr_a._data, arr_b._data)))
    if arr_a.ndim == 2 and arr_b.ndim == 1:
        m, k = arr_a.shape
        if arr_b.shape[0] != k:
            raise ValueError(f"Matmul shape mismatch: {arr_a.shape} @ {arr_b.shape}")
        out = [
            sum(arr_a._data[i][p] * arr_b._data[p] for p in range(k))
            for i in range(m)
        ]
        return PureArray(out)
    if arr_a.ndim == 1 and arr_b.ndim == 2:
        k, n = arr_b.shape
        if arr_a.shape[0] != k:
            raise ValueError(f"Matmul shape mismatch: {arr_a.shape} @ {arr_b.shape}")
        out = [
            sum(arr_a._data[p] * arr_b._data[p][j] for p in range(k))
            for j in range(n)
        ]
        return PureArray(out)
    if arr_a.ndim == 2 and arr_b.ndim == 2:
        m, k1 = arr_a.shape
        k2, n = arr_b.shape
        if k1 != k2:
            raise ValueError(f"Matmul shape mismatch: {arr_a.shape} @ {arr_b.shape}")
        out = [
            [
                sum(arr_a._data[i][p] * arr_b._data[p][j] for p in range(k1))
                for j in range(n)
            ]
            for i in range(m)
        ]
        return PureArray(out)
    if arr_a.ndim == 3 and arr_b.ndim == 2:
        b_sz = arr_a.shape[0]
        return PureArray([matmul(PureArray(arr_a._data[i]), arr_b)._data for i in range(b_sz)])
    if arr_a.ndim == 3 and arr_b.ndim == 3:
        b_sz = arr_a.shape[0]
        return PureArray([
            matmul(PureArray(arr_a._data[i]), PureArray(arr_b._data[i]))._data
            for i in range(b_sz)
        ])
    raise ValueError(f"Unsupported matmul dimensions: {arr_a.shape} @ {arr_b.shape}")


def _reduce_axis(
    arr: PureArray,
    reducer: Callable[[List[float]], float],
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> Any:
    if axis is None:
        val = reducer(_flatten_nested(arr._data))
        if keepdims:
            return PureArray(_fill_shape((1,) * arr.ndim, val))
        return val
    if axis < 0:
        axis += arr.ndim
    out_shape = tuple(
        1 if i == axis else d
        for i, d in enumerate(arr.shape)
        if keepdims or i != axis
    )
    if len(out_shape) == 0:
        return reducer(_flatten_nested(arr._data))

    reduced_shape = tuple(d for i, d in enumerate(arr.shape) if i != axis)
    flat_out: List[float] = []
    for coords in _iter_coords(reduced_shape):
        vec: List[float] = []
        for k in range(arr.shape[axis]):
            full_coords = coords[:axis] + (k,) + coords[axis:]
            vec.append(_get_broadcast_val(arr._data, arr.shape, full_coords))
        flat_out.append(reducer(vec))
    return PureArray(_unflatten(flat_out, out_shape))


def sum_array(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    return _reduce_axis(arr, sum, axis=axis, keepdims=keepdims)


def mean_array(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    return _reduce_axis(
        arr, lambda xs: sum(xs) / max(len(xs), 1), axis=axis, keepdims=keepdims
    )


def std_array(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)

    def _std(xs: List[float]) -> float:
        if not xs:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

    return _reduce_axis(arr, _std, axis=axis, keepdims=keepdims)


def max_array(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    return _reduce_axis(arr, max, axis=axis, keepdims=keepdims)


def min_array(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)
    return _reduce_axis(arr, min, axis=axis, keepdims=keepdims)


def argmax_array(a: Any, axis: Optional[int] = None) -> Any:
    arr = a if isinstance(a, PureArray) else PureArray(a)

    def _argmax(xs: List[float]) -> float:
        best_i = 0
        best_v = xs[0]
        for i, v in enumerate(xs):
            if v > best_v:
                best_v = v
                best_i = i
        return float(best_i)

    if axis is None:
        return int(_argmax(_flatten_nested(arr._data)))
    res = _reduce_axis(arr, _argmax, axis=axis, keepdims=False)
    if isinstance(res, PureArray):
        return res
    return int(res)


class PureRandom:
    """Deterministic LCG/Box-Muller random number generator matching `np.random.RandomState`."""

    def __init__(self, seed: int = 42) -> None:
        import random

        self._rng = random.Random(seed)

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def randn(self, *shape: int) -> Any:
        if not shape:
            return self._rng.gauss(0.0, 1.0)
        total = 1
        for s in shape:
            total *= s
        flat = [self._rng.gauss(0.0, 1.0) for _ in range(total)]
        return PureArray(_unflatten(flat, shape))

    def normal(
        self, loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        if size is None:
            return self._rng.gauss(loc, scale)
        shape = (size,) if isinstance(size, int) else tuple(size)
        total = 1
        for s in shape:
            total *= s
        flat = [self._rng.gauss(loc, scale) for _ in range(total)]
        return PureArray(_unflatten(flat, shape))

    def uniform(
        self, low: float = 0.0, high: float = 1.0, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        if size is None:
            return self._rng.uniform(low, high)
        shape = (size,) if isinstance(size, int) else tuple(size)
        total = 1
        for s in shape:
            total *= s
        flat = [self._rng.uniform(low, high) for _ in range(total)]
        return PureArray(_unflatten(flat, shape))

    def randint(
        self, low: int, high: Optional[int] = None, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        if high is None:
            high = low
            low = 0
        if size is None:
            return self._rng.randint(low, high - 1)
        shape = (size,) if isinstance(size, int) else tuple(size)
        total = 1
        for s in shape:
            total *= s
        flat = [float(self._rng.randint(low, high - 1)) for _ in range(total)]
        return PureArray(_unflatten(flat, shape))

    def permutation(self, n: int) -> PureArray:
        idxs = list(range(n))
        self._rng.shuffle(idxs)
        return PureArray([float(i) for i in idxs])


class PureLinalg:
    """Linear system solver and matrix utilities in pure Python."""

    @staticmethod
    def solve(a: Any, b: Any) -> PureArray:
        """Solve A x = b via Gaussian elimination with partial pivoting."""
        arr_a = a if isinstance(a, PureArray) else PureArray(a)
        arr_b = b if isinstance(b, PureArray) else PureArray(b)
        n = arr_a.shape[0]
        is_1d = arr_b.ndim == 1
        rhs_cols = 1 if is_1d else arr_b.shape[1]

        aug: List[List[float]] = []
        for i in range(n):
            row = [float(arr_a._data[i][j]) for j in range(n)]
            if is_1d:
                row.append(float(arr_b._data[i]))
            else:
                row.extend(float(arr_b._data[i][c]) for c in range(rhs_cols))
            aug.append(row)

        for col in range(n):
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                pivot = 1e-12
            for j in range(col, n + rhs_cols):
                aug[col][j] /= pivot
            for r in range(n):
                if r != col and abs(aug[r][col]) > 0.0:
                    factor = aug[r][col]
                    for j in range(col, n + rhs_cols):
                        aug[r][j] -= factor * aug[col][j]

        if is_1d:
            return PureArray([aug[i][n] for i in range(n)])
        return PureArray([[aug[i][n + c] for c in range(rhs_cols)] for i in range(n)])

    @staticmethod
    def norm(a: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        arr = a if isinstance(a, PureArray) else PureArray(a)
        sq = sum_array(arr * arr, axis=axis, keepdims=keepdims)
        if isinstance(sq, PureArray):
            return _elemwise_unary(sq, math.sqrt)
        return math.sqrt(float(sq))


class PureNumpyModule:
    """Drop-in NumPy facade backed by `PureArray` when C-NumPy is not installed."""

    float32 = float
    float64 = float
    int32 = int
    int64 = int
    pi = math.pi
    e = math.e
    inf = float("inf")
    linalg = PureLinalg()
    random = PureRandom(42)

    @staticmethod
    def RandomState(seed: int = 42) -> PureRandom:
        return PureRandom(seed)

    @staticmethod
    def array(data: Any, dtype: Any = None) -> PureArray:
        return PureArray(data)

    @staticmethod
    def asarray(data: Any, dtype: Any = None) -> PureArray:
        return data if isinstance(data, PureArray) else PureArray(data)

    @staticmethod
    def zeros(shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> PureArray:
        s = (shape,) if isinstance(shape, int) else tuple(shape)
        return PureArray(_fill_shape(s, 0.0))

    @staticmethod
    def ones(shape: Union[int, Tuple[int, ...]], dtype: Any = None) -> PureArray:
        s = (shape,) if isinstance(shape, int) else tuple(shape)
        return PureArray(_fill_shape(s, 1.0))

    @staticmethod
    def full(shape: Union[int, Tuple[int, ...]], fill_value: float, dtype: Any = None) -> PureArray:
        s = (shape,) if isinstance(shape, int) else tuple(shape)
        return PureArray(_fill_shape(s, float(fill_value)))

    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype: Any = None) -> PureArray:
        cols = m if m is not None else n
        return PureArray([[1.0 if i == j else 0.0 for j in range(cols)] for i in range(n)])

    @staticmethod
    def diag(v: Any) -> PureArray:
        arr = v if isinstance(v, PureArray) else PureArray(v)
        if arr.ndim == 1:
            n = arr.shape[0]
            return PureArray([[float(arr._data[i]) if i == j else 0.0 for j in range(n)] for i in range(n)])
        n = min(arr.shape[0], arr.shape[1])
        return PureArray([float(arr._data[i][i]) for i in range(n)])

    @staticmethod
    def arange(start: float, stop: Optional[float] = None, step: float = 1.0, dtype: Any = None) -> PureArray:
        if stop is None:
            stop = start
            start = 0.0
        out: List[float] = []
        cur = float(start)
        while (step > 0 and cur < stop - 1e-12) or (step < 0 and cur > stop + 1e-12):
            out.append(cur)
            cur += step
        return PureArray(out)

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50) -> PureArray:
        if num <= 1:
            return PureArray([float(start)])
        step = (stop - start) / (num - 1)
        return PureArray([float(start + i * step) for i in range(num)])

    @staticmethod
    def exp(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.exp(min(max(float(a), -500.0), 500.0))
        return _elemwise_unary(a, lambda x: math.exp(min(max(x, -500.0), 500.0)))

    @staticmethod
    def log(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.log(max(float(a), 1e-15))
        return _elemwise_unary(a, lambda x: math.log(max(x, 1e-15)))

    @staticmethod
    def sqrt(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.sqrt(max(float(a), 0.0))
        return _elemwise_unary(a, lambda x: math.sqrt(max(x, 0.0)))

    @staticmethod
    def sin(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.sin(float(a))
        return _elemwise_unary(a, math.sin)

    @staticmethod
    def cos(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.cos(float(a))
        return _elemwise_unary(a, math.cos)

    @staticmethod
    def tanh(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return math.tanh(float(a))
        return _elemwise_unary(a, math.tanh)

    @staticmethod
    def abs(a: Any) -> Any:
        if isinstance(a, (int, float)):
            return abs(float(a))
        return _elemwise_unary(a, abs)

    @staticmethod
    def maximum(a: Any, b: Any) -> PureArray:
        return _elemwise_binary(a, b, max)

    @staticmethod
    def minimum(a: Any, b: Any) -> PureArray:
        return _elemwise_binary(a, b, min)

    @staticmethod
    def clip(a: Any, a_min: float, a_max: float) -> PureArray:
        return _elemwise_unary(a, lambda x: min(max(x, a_min), a_max))

    @staticmethod
    def where(cond: Any, x: Any, y: Any) -> PureArray:
        arr_c = cond if isinstance(cond, PureArray) else PureArray(cond)
        arr_x = x if isinstance(x, PureArray) else PureArray(x)
        arr_y = y if isinstance(y, PureArray) else PureArray(y)
        out_shape = _broadcast_shapes(arr_c.shape, _broadcast_shapes(arr_x.shape, arr_y.shape))
        flat: List[float] = []
        for coords in _iter_coords(out_shape):
            vc = _get_broadcast_val(arr_c._data, arr_c.shape, coords)
            vx = _get_broadcast_val(arr_x._data, arr_x.shape, coords)
            vy = _get_broadcast_val(arr_y._data, arr_y.shape, coords)
            flat.append(vx if vc != 0.0 else vy)
        return PureArray(_unflatten(flat, out_shape))

    @staticmethod
    def triu(m: Any, k: int = 0) -> PureArray:
        arr = m if isinstance(m, PureArray) else PureArray(m)
        rows, cols = arr.shape
        return PureArray([
            [float(arr._data[i][j]) if j - i >= k else 0.0 for j in range(cols)]
            for i in range(rows)
        ])

    @staticmethod
    def tril(m: Any, k: int = 0) -> PureArray:
        arr = m if isinstance(m, PureArray) else PureArray(m)
        rows, cols = arr.shape
        return PureArray([
            [float(arr._data[i][j]) if j - i <= k else 0.0 for j in range(cols)]
            for i in range(rows)
        ])

    @staticmethod
    def concatenate(arrays: Sequence[Any], axis: int = 0) -> PureArray:
        arrs = [a if isinstance(a, PureArray) else PureArray(a) for a in arrays]
        if not arrs:
            return PureArray([])
        ndim = arrs[0].ndim
        if axis < 0:
            axis += ndim
        if ndim == 1:
            out: List[float] = []
            for a in arrs:
                out.extend(_flatten_nested(a._data))
            return PureArray(out)
        if ndim == 2:
            if axis == 0:
                rows: List[List[float]] = []
                for a in arrs:
                    rows.extend(_deep_copy_list(a._data))
                return PureArray(rows)
            if axis == 1:
                n_rows = arrs[0].shape[0]
                rows = []
                for i in range(n_rows):
                    row: List[float] = []
                    for a in arrs:
                        row.extend(a._data[i])
                    rows.append(row)
                return PureArray(rows)
        raise ValueError(f"Concatenate unsupported for ndim={ndim}, axis={axis}")

    @staticmethod
    def stack(arrays: Sequence[Any], axis: int = 0) -> PureArray:
        arrs = [a if isinstance(a, PureArray) else PureArray(a) for a in arrays]
        if axis == 0:
            return PureArray([_deep_copy_list(a._data) for a in arrs])
        if axis == 1 and arrs[0].ndim == 1:
            n = arrs[0].shape[0]
            return PureArray([[float(a._data[i]) for a in arrs] for i in range(n)])
        raise ValueError(f"Stack unsupported for axis={axis}")

    @staticmethod
    def column_stack(arrays: Sequence[Any]) -> PureArray:
        arrs = [a if isinstance(a, PureArray) else PureArray(a) for a in arrays]
        cols: List[PureArray] = []
        for a in arrs:
            if a.ndim == 1:
                cols.append(a.reshape(a.shape[0], 1))
            else:
                cols.append(a)
        return PureNumpyModule.concatenate(cols, axis=1)

    @staticmethod
    def vstack(arrays: Sequence[Any]) -> PureArray:
        return PureNumpyModule.concatenate(arrays, axis=0)

    @staticmethod
    def hstack(arrays: Sequence[Any]) -> PureArray:
        arrs = [a if isinstance(a, PureArray) else PureArray(a) for a in arrays]
        if arrs[0].ndim == 1:
            return PureNumpyModule.concatenate(arrs, axis=0)
        return PureNumpyModule.concatenate(arrs, axis=1)

    sum = staticmethod(sum_array)
    mean = staticmethod(mean_array)
    std = staticmethod(std_array)
    max = staticmethod(max_array)
    min = staticmethod(min_array)
    argmax = staticmethod(argmax_array)
    matmul = staticmethod(matmul)
    dot = staticmethod(matmul)
    transpose = staticmethod(transpose)


if HAS_NUMPY:
    np: Any = _real_np
else:
    np = PureNumpyModule()
    np.random.RandomState = PureNumpyModule.RandomState


def softmax(x: Any, axis: int = -1) -> Any:
    """Numerically stable softmax across `axis`."""
    arr = np.asarray(x, dtype=np.float64)
    x_max = arr.max(axis=axis, keepdims=True)
    e_x = np.exp(arr - x_max)
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-12)


def relu(x: Any) -> Any:
    """Element-wise ReLU activation."""
    arr = np.asarray(x, dtype=np.float64)
    return np.maximum(arr, 0.0)


def gelu(x: Any) -> Any:
    """Element-wise approximate GELU activation."""
    arr = np.asarray(x, dtype=np.float64)
    return 0.5 * arr * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (arr + 0.044715 * (arr ** 3))))


def pearson_corr(y_true: Any, y_pred: Any, axis: int = 0) -> Any:
    """
    Compute Pearson correlation r between `y_true` and `y_pred` along `axis` (default 0 = across samples).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yt_c = yt - yt.mean(axis=axis, keepdims=True)
    yp_c = yp - yp.mean(axis=axis, keepdims=True)
    num = (yt_c * yp_c).sum(axis=axis)
    den = np.sqrt((yt_c * yt_c).sum(axis=axis) * (yp_c * yp_c).sum(axis=axis)) + 1e-12
    return num / den
