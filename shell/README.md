# Shared multi-profile shell

Tracked source for recruiter vs commercial static shells used by flagship demo projects in this repo.

## Purpose

The shell owns shared identity and review flow:

- applied ML / data systems positioning
- project hero frame
- technical focus sidebar
- review links
- related systems links
- footer and minimal interaction behavior

Project-specific repositories keep ownership of:

- the project body content (`shell/body/<project>.html`)
- API routes and app behavior
- repo-specific architecture notes
- demo-specific limitations and links in `shell/projects/<project>.json`

## Files

- `index.html`: HTML template with placeholders
- `shell.css`: shared shell layout and identity styling
- `demo-content.css`: shared body-content defaults
- `shell.js`: sidebar toggle and small shell behavior
- `profile.json`: default profile when `profiles/<name>.json` is missing (recruiter fallback)
- `profiles/*.json`: named shell profiles (`recruiter`, `commercial`)
- `projects/*.json`: per-project config (e.g. `batch-scoring.json`)
- `render-shell.mjs`: renders one project shell to a target directory

## Render flow (Batch Scoring)

From repository root:

Recruiter shell (e.g. batch-scoring.vahdetkaratas.com):

```bash
node shell/render-shell.mjs --project shell/projects/batch-scoring.json --body shell/body/batch-scoring.html --out layout-shell --profile recruiter
```

Commercial shell (e.g. batch-scoring.vahdetlabs.com):

```bash
node shell/render-shell.mjs --project shell/projects/batch-scoring.json --body shell/body/batch-scoring.html --out layout-shell-commercial --profile commercial
```

Each command writes:

- `<out>/index.html`
- `<out>/shell.css`, `demo-content.css`, `shell.js`, `favicon.svg`
- `<out>/profile.json` (copy of the selected profile)

The selected profile controls identity, portfolio URL, technical focus, review section title, and footer tagline. Project JSON can override hero copy, CTAs, and `profiles.recruiter` / `profiles.commercial` blocks for link sets.

## Consumption model

Copy-and-render: keep `shell/` tracked in the project repo, edit project JSON + body HTML, run `render-shell.mjs`, deploy the generated `layout-shell*` directories to your static host or sync into your packaging step.

## Notes

- Facebook / Vakasoft links are excluded from the shared identity layer.
- Project body content stays repo-specific so each artifact keeps its own technical narrative.
