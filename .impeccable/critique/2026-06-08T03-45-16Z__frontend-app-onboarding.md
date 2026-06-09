---
target: frontend/app/(onboarding)
total_score: 31
p0_count: 0
p1_count: 1
timestamp: 2026-06-08T03-45-16Z
slug: frontend-app-onboarding
---
#### Design Health Score
> *Consult the Heuristics Scoring Guide.*

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | No overall progress bar to indicate how many steps remain. |
| 2 | Match System / Real World | 4 | Clear, domain-appropriate language. |
| 3 | User Control and Freedom | 3 | "Back" and "Skip" available, but no global "Exit" or "Skip all" option. |
| 4 | Consistency and Standards | 4 | Relies cleanly on the global theme tokens. |
| 5 | Error Prevention | 3 | Primary action disabled correctly when inputs are missing. |
| 6 | Recognition Rather Than Recall | 4 | All options clearly laid out visually. |
| 7 | Flexibility and Efficiency | 2 | Relies entirely on clicks; no keyboard shortcuts (like Cmd+Enter) to proceed. |
| 8 | Aesthetic and Minimalist Design | 3 | Clean and focused, though the gradient banner introduces arbitrary colors. |
| 9 | Error Recovery | 3 | Standard form validation. |
| 10| Help and Documentation | 2 | No inline help for what integrating a "Data Source" actually entails. |
| **Total** | | **31/40** | **Good** |

#### Anti-Patterns Verdict

**LLM assessment**: The UI successfully avoids major AI tells like "ghost cards" or sketchy SVG doodles. However, it relies heavily on the "centered wizard card on a glowing background" template, which is a very common SaaS default. 

**Deterministic scan**: The CLI detector found 0 issues across the onboarding components. The markup is clean.

**Visual overlays**: No reliable user-visible overlay is available in this environment; fallback source analysis was used.

#### Overall Impression
The flow is clean, legible, and functional. The single biggest opportunity is introducing a clear sense of progress and allowing power-users to bypass the wizard entirely.

#### What's Working
- **Theme integration**: Properly uses `bg-card`, `border-border`, and `text-foreground` to adapt perfectly to dark mode.
- **Clear focus**: One decision per screen keeps cognitive load low.

#### Priority Issues

- **[P1] Missing Progress Indicator**
  - **Why it matters**: Users feel trapped if they don't know how long an onboarding flow will take.
  - **Fix**: Add a minimal progress bar or "Step 1 of 3" indicator to the wizard header.
  - **Suggested command**: `$impeccable layout`

- **[P2] Hardcoded Gradient Banner**
  - **Why it matters**: The `from-orange-400 via-pink-500 to-purple-600` gradient in the Workspace Setup introduces off-brand colors that clash with the strict `RecRAG` dark mode tokens.
  - **Fix**: Re-map the gradient to use the project's `--chart-1` to `--chart-5` tokens or a `--primary` fade.
  - **Suggested command**: `$impeccable colorize`

- **[P2] No Global Exit / Skip All**
  - **Why it matters**: Expert users ("Alex") will be frustrated if they are forced to click through 3 screens to reach the app.
  - **Fix**: Add a persistent "Skip setup" button in the top right of the main layout header.
  - **Suggested command**: `$impeccable harden`

#### Persona Red Flags

**Alex (Power User)**: 
- No global "Skip all" option to jump straight to the dashboard.
- No `Cmd+Enter` keyboard shortcut to submit the workspace form.

**Sam (Accessibility-Dependent)**:
- The SVG icons in the role and tool selection lists lack `aria-hidden="true"`, which might cause screen readers to announce them unnecessarily.

#### Minor Observations
- The decorative blobs (`blur-[80px]`) are a bit overpowering in dark mode. Toning down their opacity from `/20` to `/10` might look more professional.

#### Questions to Consider
- Does RecRAG actually need to know the user's "Role" to function, or is it just for analytics? If the latter, should we distill the flow to just 2 steps?
