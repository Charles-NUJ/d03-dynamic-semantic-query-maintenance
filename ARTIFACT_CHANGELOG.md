# Artifact maintenance log

## Submission snapshot

The `icde2027-submission` tag points to commit `fc37d355d85ae83a9d2a8cccbdd3917fde975acb`, the public artifact state created during the submission grace period.

## Post-submission maintenance

The first maintenance update repairs artifact integrity checks after a fresh Git clone. The original manifest was created from CRLF working-tree bytes, while the Git objects exposed LF bytes for 99 text files. The update aligns the published file hashes with the bytes obtained from Git, updates dependent verification metadata, and excludes local execution products from the package inventory.

The update also documents the maintenance boundary in the README. It does not change the frozen revision windows, semantic predictions, query relations, method decisions, experimental measurements, algorithms, or paper claims.

A documentation-only follow-up names the three authors in the MIT copyright notice and adds a repository citation. It does not change the license terms or any scientific material.
