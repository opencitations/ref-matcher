# Reference Matching Tool for OpenCitations

Repository for a bibliographic reference matching tool designed to identify and align references between [Crossref](https://www.crossref.org/) and [OpenCitations Meta](https://opencitations.net/meta). It implements a heuristic-based approach, enabling the retrieval and validation of bibliographic metadata even in cases of incomplete or inconsistent citation records.

### Key Features

- Extracts reference metadata from Crossref (JSON format)
- Queries the OpenCitations Meta SPARQL endpoint
- Applies heuristics to determine matches without relying on unique identifiers (DOIs)
- Includes an evaluation script that uses DOI matching post-hoc to validate results

