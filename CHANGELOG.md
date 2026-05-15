# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial unofficial IBKR client built on `ib_async` with auto-reconnect, watchdog, and structured logging.
- Configuration dataclass (`IBKRConfig`) centralising all tunables.
- Unit test suite covering connection lifecycle, reconnect/back-off, watchdog, error classification, and shutdown behaviour.
- GitHub Actions workflows: CI (lint + test), Release (tag-driven build + GitHub Release), CHANGELOG auto-update.
