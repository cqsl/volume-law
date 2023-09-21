# Neural Quantum States for volume-law entangled states
Accompanying code for the comment paper "Comment on *Can Neural Quantum States Learn Volume-Law Ground States?*".
[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)


## Content of the repository

- `dis_fermions.py`: functions for generating the SYK-type disordered fermionic Hamiltonian (DF) and for diagonalizing it.
- `intializers.py`: initializer functions for generating random matrices.
- `learning.py`: function performing the training of the Neural Quantum State (NQS) ansätze.
- `renyin.py`: function computing the Rényi-n entanglement entropy.
- `simple_model.py`: `flax` model for the two-layer perceptron NQS (FF).
- `sk.py`: file containing functions for generating the quantum Sherrington-Kirkpatrick Hamiltonian (QSK) and for diagonalizing it.
- `slater.py`: `flax` model for the Slater determinant NQS with backflow transformation (FF+SD).

- **infidelity**: containing the results of the simulations.
    - **energy_params_disf_bf**, **energy_params_disf_simple**, **energy_params_sk**: folder containing the parameters of the optimized NQSs used to compute the energy errors for the two models.
    - `df_entropy.out`: Rényi-2 entropy of the exact DF ground state. 
    - `disf_bf.out`: best infidelity for the optimization of FF+SD for the DF model. 
    - `disf_bf_energy_errors.py`: computing the relative energy error of the optimized FF+SD for the DF model. Data in `disf_bf_energy_errors.out`.
    - `disf_bf_runs.py`: optimizing FF+SD for the DF model. 
    - `disf_simple.out`: best infidelity for the optimization of FF for the DF model. 
    - `disf_simple_energy_errors.py`: computing the relative energy error of the optimized SD for the DF model. Data in `disf_simple_energy_errors.out`.
    - `disf_simple_runs.py`: optimizing FF for the DF model. 
    - `figure.pdf`: figure in the comment paper. 
    - `plot.py`: plotting the data and creating the figure.
    - `runs_entropy.py`: computing the Rényi-2 entropy of the exact DF and QSK ground states. 
    - `sk.out`: best infidelity for the optimization of FF for the QSK model. 
    - `sk_energy_errors.py`: computing the relative energy error of the optimized FF+SD for the DF model. Data in `sk_energy_errors.out`.
    - `sk_runs.py`: optimizing FF+SD for the DF model. 




