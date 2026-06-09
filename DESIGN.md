# Design System

## Global Strategy
Restrained: We use clear borders, clean off-white/gray backgrounds, and a single accent color for primary actions to establish an environment of technical confidence.

## Palette
- **Canvas / Background**: Clean white (`#ffffff`) for main surfaces, with a slight neutral gray (`#f9fafb`) for app shells/body backgrounds.
- **Ink / Text**: High-contrast slate/charcoal for primary text (`#0f172a`), softer slate for secondary (`#64748b`).
- **Accent**: A strong, accessible blue (`#2563eb`) or brand-specific primary.
- **Borders / Dividers**: Soft neutral gray (`#e2e8f0`).

## Typography
- **Font Family**: Inter or standard system sans-serif (`ui-sans-serif, system-ui, sans-serif`).
- **Hierarchy**: Clear step sizing with bold weights for headings and medium for interaction labels. Max display clamp around `4rem`. Letter-spacing no tighter than `-0.02em`.
- **Text Wrap**: `balance` for headings, `pretty` for prose.

## Layout & Components
- **Spacing**: Generous rhythm, using standard `rem` steps (4, 8, 16, 24, 32px).
- **Cards & Containers**: Subtle 1px borders, gentle rounding (`8px` to `12px`), minimal to no box-shadow (avoid the ghost-card codex tell of border + heavy shadow).
- **Forms**: Clear inputs, explicit labels, distinct active/focus rings.

## Motion
- Immediate, crisp interactions. Snappy transitions (`150ms ease-out`).
- Avoid structural layout animations. Use standard opacities and subtle transforms for entry reveals.
