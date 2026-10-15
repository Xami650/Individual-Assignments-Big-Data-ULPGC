# BIG DATA — Individual Assignments (2026-2027)

GCID, ULPGC. Author: Máximo Prosperi Reynoso.

Each assignment is self-contained in its own directory, with its own README,
build instructions, data, analysis and report.

| Assignment | Topic | Status |
|---|---|---|
| [assignment1](assignment1/) | Basic matrix multiplication in Python, Java and C | submitted (tag `assignment1-v1.0`, [report](assignment1/report/main.pdf)) |
| assignment2 | — | not started |
| assignment3 | — | not started |
| assignment4 | — | not started |

## Note on the repository structure

All four assignments are kept in one repository, and assignment 1 holds three
implementations in different languages side by side. This is **not** the
recommended practice for multi-language software, where each implementation
would normally be its own project. It is done only to keep the whole class
submission together in one place. The root `pom.xml` lists
`assignment1/java` as a module only for IDE convenience. That project has no
parent POM and builds on its own (`mvn -f assignment1/java/pom.xml package`).

## License

MIT, see [LICENSE](LICENSE).
