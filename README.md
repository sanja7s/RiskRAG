# RiskRAG

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2504.08952-blue)](https://arxiv.org/abs/2504.08952)

## Overview

RiskRAG is a retrieval-augmented generation tool for AI model risk reporting, as described in the paper [Rao et al., 2025](https://arxiv.org/abs/2504.08952). It helps identify, contextualize, and prioritize model-specific risks, offering actionable mitigation strategies.

This repository contains two main components:

- **retriever** — code for fetching and embedding risk-related data.
- **generator** — code for generating structured risk reports based on retrieved information.

## Installation

1. Clone the repo:
   ```bash
   git clone git@github.com:sanja7s/RiskRAG.git
   cd RiskRAG
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Retriever

Fetch and embed risk data. Ensure your Hugging Face token is available as an environment variable:  
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```  
Then run:
```bash
python retriever/get_embeddings.py \
  --input path/to/input/data \
  --output path/to/embeddings.json
```

### Generator

Generate a risk report from embeddings:
```bash
python generator/generate_report.py \
  --embeddings path/to/embeddings.json \
  --output path/to/risk_report.md
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or issue.

## Citation

If you use this code, please cite:
```bibtex
@misc{rao2025riskrag,
  title={RiskRAG: A Data-Driven Solution for Improved AI Model Risk Reporting},
  author={Rao, Pooja S. B. and Šćepanović, Sanja and Zhou, Ke and Bogucka, Edyta Paulina and Quercia, Daniele},
  year={2025},
  eprint={2504.08952},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

