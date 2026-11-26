# Your organisation's logo

LUMEN shows the logo of the organisation that runs it at the top left of every
screen. The artwork is not part of this repository: whoever deploys LUMEN
supplies the official file, under the terms on which they may use it.

## The logo

1. Copy the file into this folder as `logo.svg` (or `logo.png`, `logo.webp`,
   `logo.jpg`). At most 512 KB. A horizontal logo on a transparent background
   reads best: it is shown 30 px high on white, and on a white plate in dark
   mode.
2. Write the manifest again and deploy as usual:

   ```bash
   scripts/write_manifest.sh
   ```

   The manifest lists every file that is uploaded; a logo that is not in it
   stays behind.

Instead of this folder, `LUMEN_BRAND_LOGO` can name a file anywhere on the
server.

## Name and colour

Two settings of the installation (environment variables; on Posit Connect the
*Vars* panel of the content):

| Setting | Example | Effect |
|---|---|---|
| `LUMEN_BRAND_NAME` | `Sample Aerospace` | the alternative text of the logo; without a logo, the name beside LUMEN's mark |
| `LUMEN_BRAND_COLOR` | `#0033a0` | the colour of the header rule, the mark and every primary button; `#rrggbb` only. The default is this deep blue, a screen approximation of Pantone 286 C; use the value from your organisation's brand guide |

## What git does with this folder

Everything here except this note is ignored by git, so that artwork you may
use but not publish does not reach a public repository by accident. For
git-backed publishing, use a private repository in which the logo is
committed (`git add -f brand/logo.svg`), or `LUMEN_BRAND_LOGO`.
