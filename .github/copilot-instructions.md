# Copilot Instructions

## Project Guidelines
- When implementing editable ImGui text fields, avoid logic that repopulates the input buffer every frame because it prevents user typing. Preserve in-progress text edits and avoid any per-frame buffer synchronization while the field is active.