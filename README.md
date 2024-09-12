## Firefly Algorithm - Pytorch

Exploration into the Firefly algorithm (a generalized version of particle swarm optimization) in Pytorch. In particular interested in hybrid <a href="https://academic.oup.com/jcde/article/9/2/706/6566441">firefly + genetic algorithms</a>, or ones that are <a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423005298">gender-based</a>.

## Install

```bash
$ pip install -r requirements.txt
```

## Usage

Test on rosenbrock minimization

```bash
$ python firefly.py --use-genetic-algorithm 1
```

## Citations

```bibtex
@article{Yang2018WhyTF,
    title   = {Why the Firefly Algorithm Works?},
    author  = {Xin-She Yang and Xingshi He},
    journal = {ArXiv},
    year    = {2018},
    volume  = {abs/1806.01632},
    url     = {https://api.semanticscholar.org/CorpusID:46940737}
}
```

```bibtex
@article{article,
    author  = {El-Shorbagy, M. and Elrefaey, Adel},
    year    = {2022},
    month   = {04},
    pages   = {706-730},
    title   = {A hybrid genetic-firefly algorithm for engineering design problems},
    volume  = {Journal of Computational Design and Engineering, Volume 9},
    journal = {Journal of Computational Design and Engineering},
    doi     = {10.1093/jcde/qwac013}
}
```
