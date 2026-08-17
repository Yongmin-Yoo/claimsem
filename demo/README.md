# ClaimSem Demo Data

This directory contains a small synthetic patent dataset for testing the ClaimSem pipeline.

## Files

```text
demo/
├── README.md
└── demo_records.json
```

## Dataset

`demo_records.json` contains nine synthetic patent records across three CPC sections:

- `A`: human necessities
- `G`: physics and computing
- `H`: electricity and communications

The records include:

- single-root dependency chains
- branching dependencies
- multiple independent claims
- multiple-parent dependent claims
- CPC section, class, and subclass labels

The dataset is synthetic and is not intended for scientific evaluation.

## Record Format

```json
{
  "patent_id": "DEMO_G01",
  "claims": [
    {
      "claim_id": "1",
      "text": "A processor-implemented method ...",
      "parent_ids": []
    },
    {
      "claim_id": "2",
      "text": "The method of claim 1 ...",
      "parent_ids": ["1"]
    }
  ],
  "cpc": {
    "section": "G",
    "class": "G06",
    "subclass": "G06N"
  }
}
```

## Smoke Test

The smoke-test configuration is located at:

```text
configs/smoke_test.json
```

It uses deterministic synthetic embeddings and does not download PatentSBERTa-V2.

The smoke test validates:

1. record loading
2. dependency validation
3. claim-depth calculation
4. root- and depth-aware pooling
5. PCA
6. spherical K-means
7. CPC metric calculation
8. artifact saving

The demo verifies the software pipeline only. It does not reproduce the paper results.
