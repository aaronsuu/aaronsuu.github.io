# Aaron Su — Portfolio Website: Project Brief

> **Purpose:** This document captures the full context of Aaron's portfolio website project so any future Claude session can pick up exactly where we left off — no re-explaining needed.

---

## Project Overview

A single-file HTML portfolio website for Aaron Su, an Electrical & Computer Engineering student at UW. The site is self-contained (one `index.html`, no build tools, no framework) with Google Fonts as the only external dependency.

**Current status:** Ready to deploy on GitHub Pages. Photo added, content populated, all features functional.

---

## File Structure

```
portfolio/
├── index.html          ← entire site (HTML + CSS + JS in one file)
├── images/
│   └── me.jpg          ← profile photo (600x600, cropped from Kelowna trip)
├── resume.pdf           ← not yet added — Aaron needs to drop this in
└── README.md           ← editing guide for VS Code workflow
```

---

## Design System

### Colors
| Token      | Dark Mode   | Light Mode  |
|------------|-------------|-------------|
| Background | `#252422`   | `#FFFCF2`   |
| Card       | `#302e2a`   | `#ffffff`   |
| Text       | `#FFFCF2`   | `#252422`   |
| Accent     | `#EB5E28`   | `#EB5E28`   |

### Fonts
- **Display:** Playfair Display (serif) — headings, names, titles
- **Body:** Source Sans 3 — paragraphs, descriptions
- **Mono:** IBM Plex Mono — dates, labels, tags, code

### Design Decisions
- Dark/light theme toggle with `localStorage` persistence
- Subtle grain texture overlay on `body::after`
- Orange (`#EB5E28`) as sole accent — used for links, labels, hover states, cursor
- Email encoded as `&#64;` to prevent Cloudflare email obfuscation (this caused corruption in earlier versions)
- All buttons and tags use `border-radius: 50px` (pill shape)
- Cards lift on hover with `translateY(-5px)` and shadow

---

## Pages & Features

### 1. Home (`pg-home`)
- **Hero:** Name, typewriter animation, tagline with UW link, profile photo (circular crop)
- **Status badge:** Green pulsing dot — "Currently: Researching at SEAL Lab, UW · Joining HKUST MINSys Lab Jun 2026"
- **Contact row:** Email, LinkedIn, GitHub links + orange "Resume" download button
- **Bio section:** Two-column layout with links to UW ECE, SEAL Lab, Richmond Fire-Rescue, EMALB Schedule 2 licensing. Includes a highlighted blockquote on the right.
- **Interests section ("When I'm Not Engineering"):** 2-column grid of paragraph blocks with emoji headings:
  - F1 (Mercedes, Ferrari, Verstappen)
  - Coffee (lattes, Wilton Benitez Colombian blend)
  - Transit Systems (Link, MTR, Tokyo Metro, JR, SkyTrain)
  - Photography (Sony A7II, Sigma 30mm f/1.4)
  - Music (Coldplay, Maroon 5, Jay Chou)
- **Education:** UW (2024–2028), J.N. Burnett (2019–2024)
- **Experience:** HKUST MINSys Lab (Jun 2026, upcoming), SEAL Lab (Sept 2025–present), EMR (2022–2024), FIRST Robotics CNC (2022–2024)
- **Skills grid:** Python, Java, R, MATLAB, Fusion 360, AutoCAD, KiCAD, MultiSIM, PCB Design, OOP, Microsoft Office, Adobe Creative Suite, ML, DevOps, LLMs, CNC Fabrication, 3D Printing, ESP32
- **CTA:** "See what I've built" → navigates to Projects

### 2. Projects (`pg-projects`)
- **Search bar:** Appears in navbar only on this page. Filters by title, description, and keywords. Matching tags highlight in orange.
- **Gallery grid:** Cards with thumbnail, title, 2-line description, keyword tags
- **Detail overlay:** Click a card → full-screen modal with hero image, writeup (supports HTML), keyword tags, additional image gallery
- **Current projects (from resume):**
  - Transit Clock (ESP32, API, IoT)
  - Line Following Car (KiCAD, PCB, PID)
  - SEAL Lab Custom LLMs (Python, ML, DevOps)
  - FIRST Robotics CNC (Fusion 360, CAD, Robotics)
  - One placeholder template
- **All project data** lives in the `PROJECTS` JavaScript array — easy to edit

### 3. Uses (`pg-uses`)
- **Structure:** "What I Use" page inspired by uses.tech
- **Categories:** Hardware, Software & Tools, Desk & Everyday Carry, Lab & Maker Gear
- **Status:** All placeholder text — Aaron will fill in later

### 4. Thanks / Acknowledgements (`pg-ack`)
- **Layout:** Centered page with avatar initials, name (linked), role, personal note
- **Current entries:**
  - **Arian Shamaei** — Friend & Mentor (LinkedIn link). "For inspiring me to push outside my comfort zone."
  - **Prof. Alexander Mamishev** — Professor & SEAL Lab Director (UW faculty page link). Advice on staying competitive in an AI-driven world.

---

## Typewriter Titles

Cycles through these with a typing → pause → deleting → next animation:

1. Machine Learning Engineer
2. Computer Vision Engineer
3. Electrical Engineer
4. Software Developer
5. Embedded Systems Engineer
6. Licensed Paramedic
7. Caffeine-Powered Problem Solver

Array location in code: `var TITLES=[...]`

---

## Keyboard Shortcuts

- `/` — focus search bar (navigates to Projects page if not already there)
- `Esc` — close project detail overlay

---

## Links Inventory

All external links in the site and what they point to:

| Context | URL |
|---------|-----|
| UW ECE | `https://www.ece.uw.edu/` |
| SEAL Lab | `http://uwseal.com/` |
| HKUST | `https://hkust.edu.hk/` |
| MINSys Lab (Dr. Ouyang) | `https://xmouyang.github.io/` |
| Richmond Fire-Rescue | `https://firerescue.richmond.ca/` |
| EMALB Licensing | `https://www2.gov.bc.ca/...licensing` |
| Prof. Mamishev | `https://people.ece.uw.edu/mamishev/` |
| Arian Shamaei | `https://www.linkedin.com/in/arianshamaei/` |
| Aaron LinkedIn | `https://linkedin.com/in/aaronsuuu/` |
| Mercedes F1 | `https://www.mercedesamgf1.com/` |
| Ferrari F1 | `https://www.ferrari.com/en-EN/formula1` |
| Wilton Benitez | `https://wiltonbenitez.com/` |
| Link Light Rail | `https://www.soundtransit.org/...stations` |
| MTR | `https://www.mtr.com.hk/` |
| Tokyo Metro | `https://www.tokyometro.jp/en/` |
| Sony A7II | `https://www.sony.co.uk/.../ilce-7m2-body-kit` |
| Coldplay Spotify | `https://open.spotify.com/artist/4gzpq5DPGxSnKTe4SA8HAU` |
| Jay Chou Spotify | `https://open.spotify.com/artist/2elBjNSdBE2Y3f0j1mjrql` |
| uses.tech | `https://uses.tech` |
| GitHub (placeholder) | `https://github.com/` ← **needs real username** |
| Resume download | `resume.pdf` ← **needs file** |

---

## Known TODOs

- [ ] **GitHub URL** — `https://github.com/` is still a placeholder, needs Aaron's real username
- [ ] **resume.pdf** — not yet provided, download button points to it
- [ ] **Uses page** — all placeholder content, Aaron will fill in later
- [ ] **Project thumbnails/images** — all empty strings, projects show hex placeholder icon
- [ ] **Deploy to GitHub Pages** — repo not yet created
- [ ] **Consider UW hosting** as secondary deployment (`students.washington.edu/awjsu/`)

---

## Technical Notes for Future Sessions

### Editing the file
- CSS class names are abbreviated for file size (e.g., `.nl` = nav link, `.ti` = timeline item, `.tg` = tag, `.sk` = skill chip)
- Theme variables are in `[data-theme="dark"]` and `[data-theme="light"]` blocks at top of `<style>`
- All dynamic content (projects, typewriter titles) is in `<script>` at the bottom
- Project data is a JS array of objects — `PROJECTS`
- Navigation is SPA-style via `go('pagename')` function toggling `.pg` divs

### Past issues encountered
- **Cloudflare email obfuscation** corrupted the file when fetched via `web_fetch` — injected `__cf_email__` spans and a script tag that broke the typewriter and theme toggle. Fixed by using `&#64;` for the @ symbol.
- **Incremental `str_replace` edits** across many turns caused drift — a full file rebuild was necessary to fix accumulated issues.
- **Recommendation:** For major changes, rebuild the full file rather than patching. For small content updates (adding a project, fixing a link), `str_replace` is fine.

### Aaron's preferences
- Wants links on important things (institutions, labs, teams) but not over-linked
- Prefers paragraph-style sections over chip/pill lists for personal content
- Values personality in the portfolio but wants it balanced with professionalism
- Communication style: direct, casual, values first-principles reasoning
- Will fact-check links — don't use unverified URLs

---

## Deployment Plan (GitHub Pages)

1. Create repo named `{username}.github.io`
2. Upload `index.html`, `images/me.jpg`, `resume.pdf`, `README.md`
3. Settings → Pages → Source: `main`, root
4. Live at `https://{username}.github.io` within ~60 seconds
5. Optional: add custom domain later (~$12/year from Cloudflare Registrar or Porkbun)

---

*Last updated: May 27, 2026*
