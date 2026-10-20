# BANTAI

BANTAI is an interactive barangay incident-monitoring prototype for reporting garbage piles, potholes, and road surface damage.

## Current functionality

- Resident report submission and report tracking.
- Admin review queue for validating or rejecting observations.
- Incident records and status updates.
- Incident and hotspot rankings based on validated observations.
- Simulated video observations for demonstrating the review workflow.

This is a frontend demo with sample data. The resident/admin switch is a demonstration control, not authentication or authorization. Detection is simulated, and refreshing the page resets the session. There is no production backend or persistent database.

## Technology

React 19, TypeScript, Vite 8, and Tailwind CSS 4. The project originated in Figma Make and retains its Vite integration.

## Local development

Use Node.js 22. The included `.mise.toml` also specifies pnpm 10.34.3.

Install dependencies using the committed pnpm lockfile:

```powershell
pnpm install --frozen-lockfile
```

Start the local server, bound to your computer only:

```powershell
pnpm dev --host 127.0.0.1
```

Open <http://127.0.0.1:8443>. The configuration defaults to port 8443; the `PORT` environment variable can override it.

## Available commands

| Command | Purpose |
| --- | --- |
| `pnpm dev --host 127.0.0.1` | Start local development with hot reload |
| `pnpm build` | Generate a production build in `dist/` |
| `pnpm preview --host 127.0.0.1` | Preview the generated build locally |
| `pnpm format` | Format project files with oxfmt |

There is currently no test script in `package.json`.

## Project structure

- `src/main.tsx` - React entry point.
- `src/App.tsx` - Application root.
- `src/IncidentApp.tsx` - Resident and admin workflow interface.
- `src/IncidentPanels.tsx` - Report, review, incident, and simulation panels.
- `src/ReportForm.tsx` - Resident report form.
- `src/incidents.ts` - Sample data and incident workflow logic.
- `src/index.css` and `src/workflow.css` - Application styles.
- `.figma/make/site.json` - Site configuration imported by Vite; keep this file in the repository.
- `vite.config.ts` - Build and development configuration.

## Uploading to GitHub

Commit the source, configuration, README, `.gitignore`, and lockfiles. Exclude dependency folders, generated builds, local output, and secrets.

When uploading through GitHub's website, manually leave out ignored files and folders: browser uploads do not apply your local `.gitignore` automatically. In particular, do not upload `node_modules/`, `dist/`, `output/`, `tmp/`, or `.env` files containing secrets.
