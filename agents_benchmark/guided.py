"""Guided."""


GUIDED_ORDER = (
    "dior_localization", "dota", "fair1m",
    "xview2", "fpcd",
    "rareplanes", "cowc", "neon", "animals",
    "landmarks", "patternnet",
    "dior_classification",
)

_COUNTING = """First inspect the whole image to judge the target size and density.
Then divide the image into overlapping grids and zoom in to count tile by tile.
For each tile, count only targets whose centers fall inside that tile, to avoid double counting.
Add the tile counts and quickly re-check the whole image once."""

_LOCALIZATION = """First break down the description into attributes, position, and relations.
List all candidate targets.
Eliminate candidates one by one using their position and relations.
Zoom in on the target edges to tighten the box boundaries.
Confirm that the coordinates are expressed in original-image pixels."""

_RECOGNITION = """First describe the key visible features in the image.
Then compare them with the options one by one and eliminate options that do not match."""

GUIDANCE = {
    "dior_localization": _LOCALIZATION,
    "dota": _LOCALIZATION,
    "fair1m": _LOCALIZATION,
    "xview2": """First count the buildings tile by tile in the pre-disaster image.
Then compare each building with the same location in the post-disaster image and zoom in when needed.
Assign each building a damage level according to the definitions in the task.
Check whether the four damage counts are approximately equal to the total number of buildings before the disaster.""",
    "fpcd": """First locate the same pond in both images.
Then compare separately whether water is present, whether the pond outline is present, and whether vegetation or earthwork has changed.
Finally compare the evidence with options A–E and eliminate inconsistent options.""",
    "rareplanes": _COUNTING,
    "cowc": _COUNTING,
    "neon": _COUNTING,
    "animals": _COUNTING,
    "landmarks": _RECOGNITION,
    "patternnet": _RECOGNITION,
    "dior_classification": """Zoom in on the red-box region while also looking at the surrounding environment, such as runways or water.
Use scale and shape to eliminate impossible categories.""",
}
