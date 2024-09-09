## Firefly Algorithm - Pytorch (wip)

Exploration into the Firefly algorithm (a generalized version of particle swarm optimization) in Pytorch. In particular interested in hybrid <a href="https://academic.oup.com/jcde/article/9/2/706/6566441">firefly + genetic algorithms</a>, or ones that are <a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423005298">gender-based</a>.

## Install

```bash
$ pip install -r requirements.txt
```

## Usage

Test on rosenbrock minimization

```bash
$ python firefly.py
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
