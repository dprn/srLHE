# srLHE

Codes for contrast enhancement via Wilson-Cowan and Local Histogram Equalization with sub-Riemannian interaction kernel. 

For an in-depth exposition of the algorithm, please have a look at the paper *A cortical-inspired sub-Riemannian model for Poggendorff-type visual illusions* by E. Baspinar, L. Calatroni, V. Franceschi, and D. Prandi (available [here](https://arxiv.org/abs/2012.14184)).

## How to reproduce the experiments of the paper

Clone this repository, and launch `experiments.jl`. That is:

````sh
git clone https://github.com/dprn/srLHE
cd srLHE
julia experiments.jl
````

To reproduce the experiments obtained via the previous (non sub-Riemannian) version of the algorithm, please use the `WCvsLHE` package (available [here](https://github.com/dprn/WCvsLHE)).
