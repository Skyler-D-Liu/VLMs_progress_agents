"""Input errors."""


class TrainingDataError(ValueError):
    pass


class AssistantMaskError(TrainingDataError):
    pass
