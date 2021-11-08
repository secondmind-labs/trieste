from ..transforms import DataTransformModelWrapper
from .models import DeepGaussianProcess


class DeepGaussianProcesswithDataTransform(DataTransformModelWrapper, DeepGaussianProcess):
    """A wrapped `DeepGaussianProcess` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.
    """

    pass
