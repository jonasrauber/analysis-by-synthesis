# Analysis by Synthesis

This is a complete reimplementation of our [Analysis by Synthesis](https://github.com/bethgelab/AnalysisBySynthesis) model.
The experiments in the paper were done with the original implementation. To the best of my knowledge, both are equivalent
and I tried to carefully reproduce all results but I cannot make any guarantees.

## Results from the paper

Our paper can be found on [arXiv](https://arxiv.org/abs/1805.09190). It was accepted at ICLR 2019.

<p float="left">
  <img src="figures/L2_accuracy_distortion_curves.png" width="30%" />
  <img src="figures/Linf_accuracy_distortion_curves.png" width="30%" />
  <img src="figures/L0_accuracy_distortion_curves.png" width="30%" />
</p>


All data from the accuracy-distortion curves can be downloaded in raw form:

* [figures/L2_accuracy_distortion_curves.pickle](figures/L2_accuracy_distortion_curves.pickle)
* [figures/Linf_accuracy_distortion_curves.pickle](figures/Linf_accuracy_distortion_curves.pickle)
* [figures/L0_accuracy_distortion_curves.pickle](figures/L0_accuracy_distortion_curves.pickle)

## BibTex

```bibtex
@inproceedings{schott2018towards,
  title={Towards the first adversarially robust neural network model on {MNIST}},
  author={Lukas Schott and Jonas Rauber and Matthias Bethge and Wieland Brendel},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=S1EHOsC9tX},
}
```
