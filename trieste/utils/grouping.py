from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Mapping, Iterator, Callable
from returns.primitives.hkt import Kind1, SupportsKind1, dekind

T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")
V = TypeVar("V")
G = TypeVar("G", bound="Grouping")


class Grouping(ABC, Generic[T_co]):
    @abstractmethod
    def map(self: G, f: Callable[[T_co], U]) -> Kind1[G, U]:
        ...

    @abstractmethod
    def zip_with(self: G, other: Kind1[G, U], f: Callable[[T_co, U], V]) -> Kind1[G, V]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        ...


class One(Grouping[T_co], SupportsKind1["One", T_co]):
    def __init__(self, o: T_co):
        self._o = o

    def get(self) -> T_co:
        return self._o

    def map(self, f: Callable[[T_co], U]) -> One[U]:
        return One(f(self._o))

    def zip_with(self, other: Kind1[One, U], f: Callable[[T_co, U], V]) -> One[V]:
        return One(f(self.get(), other.get()))

    def __iter__(self) -> Iterator[T_co]:
        return iter((self._o,))


class Many(Grouping[T_co], SupportsKind1["Many", T_co]):
    def __init__(self, m: Mapping[str, T_co]):
        self._m = m

    def __getitem__(self, key: str) -> T_co:
        return self._m[key]

    def map(self, f: Callable[[T_co], U]) -> Many[U]:
        return Many({k: f(v) for k, v in self._m.items()})

    def zip_with(self, other: Kind1[Many, U], f: Callable[[T_co, U], V]) -> Many[V]:
        return Many({k: f(self[k], dekind(other)[k]) for k in self._m})

    def __iter__(self) -> Iterator[T_co]:
        return iter(self._m.values())
