from abc import ABC

from torch import Tensor


class _CoModule(ABC):
    def __init_subclass__(cls) -> None:
        for fn, description in [
            ("forward", "frame-wise forward computation"),
            (
                "forward_regular",
                "clip-wise forward computation with state initialisation",
            ),
            (
                "forward_regular_unrolled",
                "clip-wise forward without state initialisation, but which is identical to the non-continual component",
            ),
        ]:
            assert callable(
                getattr(cls, fn, None)
            ), f"A CoModule should implement a `{fn}` function which performs {description}."

        assert (
            type(getattr(cls, "delay", None)) == property
        ), "A CoModule should implement a `delay` property."

    def forward(self, input: Tensor) -> Tensor:
        """Clip-wise forward computation with state initialisation"""
        ...  # pragma: no cover

    def forward_regular(self, input: Tensor) -> Tensor:
        """Clip-wise forward computation with state initialisation"""
        ...  # pragma: no cover

    def forward_regular_unrolled(self, input: Tensor) -> Tensor:
        """Clip-wise forward without state initialisation, but which is identical to the non-continual component"""
        ...  # pragma: no cover

    @property
    def delay(self) -> int:
        """Temporal delay of the module

        Returns:
            int: Temporal delay of the module
        """
        ...  # pragma: no cover

    def clean_state(self):
        """Clean module state
        This serves as a dummy function for modules which do not require state-cleanup
        """
        ...  # pragma: no cover
