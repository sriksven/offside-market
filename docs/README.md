# Offside Market — Docs

All project documentation lives in this folder. The repo's only top-level
markdown is `README.md` (user-facing intro, quickstart, sample numbers).

| #  | File                                     | Read this when…                                                  |
| -- | ---------------------------------------- | ---------------------------------------------------------------- |
| 1  | [`STATUS.md`](STATUS.md)                 | You want the TL;DR — what's done, what's next, public URLs.     |
| 2  | [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) | You're setting up a new machine, deploying, or debugging deps.   |
| 3  | [`FLOW.md`](FLOW.md)                     | You want the data + execution flow at a glance with diagrams.    |
| 4  | [`FILE_INDEX.md`](FILE_INDEX.md)         | You're looking for "where does X live? what does Y produce?"     |
| 5  | [`COMPLETED.md`](COMPLETED.md)           | You're checking what's done in detail, with known limitations.   |

If you only have time to read one file, read `STATUS.md`. Everything else
expands on what's in there.

## Update protocol

| Doc                 | Bump when…                                                                  |
| ------------------- | --------------------------------------------------------------------------- |
| `STATUS.md`         | A phase moves forward, a deliverable ships, a public URL is captured.       |
| `INFRASTRUCTURE.md` | A dependency, runtime version, or deployment target changes.                |
| `FLOW.md`           | A pipeline step is added/removed, a new artifact is written, a new endpoint or view ships. |
| `FILE_INDEX.md`     | A tracked file is added, renamed, or its public surface meaningfully changes. |
| `COMPLETED.md`      | A phase moves forward, a known limitation is fixed, or a new caveat appears. |

If you only have time for one update, update `STATUS.md` and `COMPLETED.md`. The other three stay accurate longer between changes.
