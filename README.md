# IB TWS

## Contributing

### Commit message style

This repository generates [CHANGELOG.md](CHANGELOG.md) automatically with
[git-cliff](https://git-cliff.org/) from commit messages. To make sure your
changes land in the right section of the changelog, please use
[Conventional Commits](https://www.conventionalcommits.org/) prefixes on every
commit (and on PR squash-merge titles):

- `feat:` — a new feature
- `fix:` — a bug fix
- `perf:` — a performance improvement
- `refactor:` — a code change that neither fixes a bug nor adds a feature
- `docs:` — documentation only changes
- `test:` — adding or correcting tests
- `build:` — build system or dependency changes
- `ci:` — CI configuration changes
- `chore:` — other maintenance tasks

Add `!` after the type (e.g. `feat!:`) or a `BREAKING CHANGE:` footer for
backwards-incompatible changes. Release bumps use `chore(release): vX.Y.Z`
and are intentionally omitted from the changelog.

Commits that do not follow this convention will still appear in the changelog
under an `Other` section, but using the prefixes above is strongly preferred.
