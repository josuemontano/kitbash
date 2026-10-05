"""Slugs and the data-block naming convention shared by prompts and checks."""

import re
import unicodedata

from attrs import frozen

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "item") -> str:
    """Lowercase snake_case identifier made of ASCII letters, digits and underscores."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = _NON_SLUG.sub("_", ascii_text.lower()).strip("_")
    if not slug:
        return fallback
    return slug if not slug[0].isdigit() else f"{fallback}_{slug}"


def unique_slug(text: str, taken: set[str]) -> str:
    """A slug that is not in ``taken``; the result is added to ``taken``."""
    base = slugify(text)
    candidate, n = base, 2
    while candidate in taken:
        candidate = f"{base}_{n:02d}"
        n += 1
    taken.add(candidate)
    return candidate


@frozen
class NamingConvention:
    """Patterns with ``{slug}`` and ``{part}`` placeholders, for example ``mat_{slug}_{part}``."""

    object: str = "{slug}"
    mesh: str = "{slug}_mesh"
    material: str = "mat_{slug}_{part}"
    image: str = "tex_{slug}_{part}"
    collection: str = "{slug}"

    def as_dict(self) -> dict[str, str]:
        return {
            "object": self.object,
            "mesh": self.mesh,
            "material": self.material,
            "image": self.image,
            "collection": self.collection,
        }

    def describe(self, slug: str) -> str:
        """Human readable summary used in prompts."""
        return "\n".join(
            f"- {kind}: `{pattern.format(slug=slug, part='<part>')}`" for kind, pattern in self.as_dict().items()
        )
