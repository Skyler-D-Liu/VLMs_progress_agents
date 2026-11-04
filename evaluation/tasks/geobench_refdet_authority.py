"""Geobench refdet authority."""

SYSTEM_MESSAGE = (
    "You are evaluating aerial and satellite imagery. Follow the requested output schema "
    "exactly and do not provide an explanation."
)


def get_user_prompt(dataset, width, height, description):
    from vleo_runtime.localization import localization_prompt
    return localization_prompt(dataset, width, height, description)
