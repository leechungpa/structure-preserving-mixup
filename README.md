<h1 align="center"> <p>A Generalized Theory of Mixup for Structure-Preserving Synthetic Data</p></h1>
<h3 align="center">
    <p>Chungpa Lee, Jongho Im, Joseph H.T. Kim</p>
    <p>Yonsei University</p>
    <a href="https://proceedings.mlr.press/v258/lee25b.html">
        <img src="https://img.shields.io/badge/paper-blue.svg" alt="paper">
    </a>
    <a href="https://arxiv.org/abs/2503.02645">
        <img src="https://img.shields.io/badge/arXiv-b31b1b.svg" alt="arXiv">
    </a>
</h3>

Mixup is a widely adopted data augmentation technique known for enhancing the generalization of machine learning models by interpolating between data points. Despite its success and popularity, limited attention has been given to understanding the statistical properties of the synthetic data it generates. In this paper, we delve into the theoretical underpinnings of mixup, specifically its effects on the statistical structure of synthesized data. We demonstrate that while mixup improves model performance, it can distort key statistical properties such as variance, potentially leading to unintended consequences in data synthesis. To address this, we propose a novel mixup method that incorporates a generalized and flexible weighting scheme, better preserving the original data’s structure. Through theoretical developments, we provide conditions under which our proposed method maintains the (co)variance and distributional properties of the original dataset. Numerical experiments confirm that the new approach not only preserves the statistical characteristics of the original data but also sustains model performance across repeated synthesis, alleviating concerns of model collapse identified in previous research.


## Usage

To sample mixup weights from the EpBeta distribution:

```python
from src.epbeta import get_epbeta_parameter, generate_epbeta_weight

epsilon_0 = 0.3
epsilon_1 = 0.3
delta = 0.01

alpha, beta = get_epbeta_parameter(
    epsilon_0=epsilon_0,
    epsilon_1=epsilon_1,
    delta=delta,
    return_all_parameter=False  # return only one (most accurate) pair
)

weights = generate_epbeta_weight(
    size=10,  # number of mixup weights to generate
    epsilon_0=epsilon_0,
    epsilon_1=epsilon_1,
    alpha=alpha,
    beta=beta,
    seed=1234,
    verbos=True
)
```

To generate synthetic datasets from a CSV file using the proposed Mixup method:

```bash
python -m src.mixup ./data/file_to_synthesize.csv
```


## Citation

```tex
@InProceedings{lee2025generalized,
  title = {A Generalized Theory of Mixup for Structure-Preserving Synthetic Data},
  author = {Lee, Chungpa and Im, Jongho and Kim, Joseph HT},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  year = {2025},
  url = {https://proceedings.mlr.press/v258/lee25b.html}
}
```