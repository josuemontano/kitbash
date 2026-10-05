# Task: find reference photos on the web

Use the web_search tool to find $count photos of this object:
- name: $name
- description: $description
- suggested search: $query

Each result must be a DIRECT link to an image file (.jpg, .jpeg, .png or .webp) that downloads without a
login, not an HTML page. Prefer product or catalogue photos showing the whole object, alone, on a plain
white or light background, in a front or three-quarter view, with a shape close to the description.

Respond with JSON only:
{"images": [{"url": "https://...", "title": "short description of the photo"}]}
