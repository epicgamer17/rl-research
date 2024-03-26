from collections.abc import Sized
from abc import ABC, abstractclassmethod

from typing import TypeVar, Generic

T = TypeVar("T")


class Storable(Sized, Generic[T]):
    """
    Classes that implement this interface can be used to store an object of user-defined type T with the __store__ method
    """

    @abstractclassmethod
    def __store__(self, t: T):
        pass


class Sampleable(ABC, Generic[T]):
    """
    Classes that implement this interface can be used to sample an object of user-defined type T with the __sample__ method
    """

    @abstractclassmethod
    def __sample__(self) -> T:
        pass


class NStepable(ABC, Generic[T]):
    """
    Classes that implement this interface can be used to produce an n-step-transition of user-defined type T
    """

    @abstractclassmethod
    def __get_n_step_info__(self) -> T:
        """
        Create an n-step transition
        """
        pass


class WithId(ABC):
    """
    Classes that implement this interface can be used to check if the id is present at a certain index in the buffer or not
    """

    @abstractclassmethod
    def __store__id__(self, index: int, id: str):
        pass

    @abstractclassmethod
    def __check_id__(self, index: int, id: str) -> bool:
        pass
