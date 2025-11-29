# **FNQS-ViT: Foundation Neural-Network Quantum States with Vision Transformers**

A high-performance variational Monte-Carlo (VMC) framework implementing
**Foundation Neural-Network Quantum States (FNQS)** using a **Vision Transformer (ViT) backbone**.
The package provides:

* A unified neural ansatz for many different quantum Hamiltonians
* Fully vectorized sampling with JAX
* Support for **full Hilbert space** and **fixed-$S^z$ sectors**
* Support for **Stochastic Reconfiguration (SR)**
* Modular components: patch embedding, ViT blocks, lattice utilities, samplers, local energy operators

This repository follows the methodology of
***Foundation Neural-Networks Quantum States as a Unified Ansatz for Multiple Hamiltonians***
and adapts the FNQS idea into a ViT-based architecture for 2D spin systems.

---

# **1. Overview**

FNQS-ViT is a neural-network ansatz for many-body quantum wavefunctions:

[
\Psi_\theta(\sigma) ;=; e^{\log\psi_\theta(\sigma)} ,
]

where the configuration (\sigma \in {-1,+1}^{L\times L}) is embedded into patches, processed by a Vision Transformer, and mapped to a complex scalar log-amplitude.

This allows expressive modeling of:

* Frustrated magnets (e.g., $J_1$–$J_2$ model)
* Large 2D lattices
* Multiple Hamiltonians using a **shared foundation model** (FNQS idea)

The package implements:

| Component                  | Description                                       |
| -------------------------- | ------------------------------------------------- |
| FNQS-ViT Model             | Patch embedding + ViT encoder + scalar output     |
| Vectorized Samplers        | Full space & fixed-$S^z$ sector versions          |
| J1–J2 Hamiltonian          | NN & NNN edges, periodic boundary options         |
| Local Energy               | Full batch (E_\text{loc}(\sigma)) for all samples |
| Stochastic Reconfiguration | Natural-gradient parameter updates                |
| Device Utilities           | Automatic GPU/CPU selection                       |

---

# **2. Theory Summary (Short)**

## **2.1 Variational Principle**

The energy of the neural-network quantum state is:

[
E(\theta) =
\frac{\langle \Psi_\theta | H | \Psi_\theta \rangle}
{\langle \Psi_\theta | \Psi_\theta \rangle}.
]

Monte-Carlo sampling draws configurations from:

[
p_\theta(\sigma) = \frac{|\Psi_\theta(\sigma)|^2}{Z}.
]

The **local energy** is:

[
E_\text{loc}(\sigma) =
\sum_{\sigma'} H_{\sigma,\sigma'}
\frac{\Psi_\theta(\sigma')}{\Psi_\theta(\sigma)}
;=;
E_{\text{diag}}(\sigma) +
\sum_{\sigma' \neq \sigma}
e^{\log\psi(\sigma') - \log\psi(\sigma)} .
]

The Monte-Carlo estimator:

[
E(\theta) \approx
\frac{1}{M} \sum_{k=1}^M E_\text{loc}(\sigma_k).
]

---

## **2.2 Log-derivatives**

For SR and gradient-based optimizers, log-derivatives are:

[
O_i(\sigma) = \frac{\partial \log\psi_\theta(\sigma)}{\partial\theta_i},
\quad
\vec{O}(\sigma) \in \mathbb{C}^P .
]

---

## **2.3 Stochastic Reconfiguration (SR)**

SR approximates natural-gradient descent on the manifold of quantum states:

[
S , \delta\theta = -\eta , G ,
]

where:

[
S_{ij}=
\big\langle O_i^\ast O_j \big\rangle

* \langle O_i^\ast\rangle\langle O_j\rangle,
  ]

[
G_i =
2 , \Re\Big(
\langle O_i^\ast E_\text{loc} \rangle

* \langle O_i^\ast\rangle \langle E_\text{loc}\rangle
  \Big).
  ]

The update:

[
\theta ;\leftarrow; \theta + \delta\theta.
]

---

## **2.4 J1–J2 Hamiltonian**

[
H = J_1 \sum_{\langle i,j\rangle}
\mathbf{S}_i \cdot \mathbf{S}*j
;+;
J_2 \sum*{\langle!\langle i,j\rangle!\rangle}
\mathbf{S}_i \cdot \mathbf{S}_j
]

In the Ising basis, off-diagonal terms flip pairs of spins.

---

# **3. FNQS-ViT Architecture**

FNQS-ViT uses the following pipeline:

### **3.1 Patch Embedding**

[
\sigma \in \mathbb{Z}_2^{L\times L}
\to \text{patch tokens: } (p_1,\dots,p_K)
]

### **3.2 Vision Transformer Encoder**

Standard ViT blocks:

* Multi-Head Self-Attention
* MLP with GELU
* LayerNorm
* Residual connections

### **3.3 Output Layer**

The transformer tokens are pooled and mapped to:

[
\log\psi_\theta(\sigma) \in \mathbb{C}.
]

Gamma-conditioning (FNQS idea):

* global or structured ( \gamma )-fields
* model learns features shared across Hamiltonians

---

# **4. Installation**

```bash
git clone https://github.com/<repo>/fnqs_vit.git
cd fnqs_vit
pip install -r requirements.txt
```

---

# **5. Usage Example**

### **Run VMC training on J1–J2 model**

```bash
python src/fnqs_vit/experiments/train_j1j2_fnqs.py
```

For fast debugging:

```bash
python src/fnqs_vit/experiments/train_j1j2_fnqs.py --fast
```

---

# **6. API Overview**

## **Model**

```python
from fnqs_vit.fnqs.model import FNQSViT

model = FNQSViT(
    depth=4,
    embed_dim=72,
    hidden_dim=288,
    num_heads=12,
    patch_size=(2,2),
    gamma_mode="structured",
)
```

## **Sampler (full space)**

```python
from fnqs_vit.vmc.sampler import sample_chain_batch
```

## **Sampler (fixed Sz sector)**

```python
from fnqs_vit.vmc.sampler_sz import sample_chain_batch_sz
```

## **Local energy**

```python
from fnqs_vit.vmc.local_energy import compute_local_energy_batch
```

## **SR update**

```python
from fnqs_vit.vmc.sr import compute_sr_matrices, sr_update
```

---

# **7. Directory Structure**

```
fnqs_vit/
    fnqs/               # FNQS-ViT architecture
    vmc/                # samplers, local energy, SR
    hamiltonians/       # J1-J2, lattice graphs
    experiments/        # training scripts, observables
    tests/              # unit tests for key modules
    utils/              # device management
```

---

# **8. Current Issues (Under Investigation)**

### **1. Residual imaginary part in local energy**

Energy should be strictly real for Hermitian Hamiltonians, but small imaginary components remain:

* may indicate asymmetric sampling
* vectorized flipping logic may be inconsistent
* log-ratios may accumulate phase error
* complex gradient split (real/imag) may need holomorphic logic

### **2. Sector sampler performance**

The $S^z$-conserving sampler uses pair flips; acceptance may be lower than single-flip proposals.

### **3. Variation between different runs**

Even with the same seed, VMC has sources of variance:

* non-deterministic JAX ops on CPU
* SR matrix inversion is sensitive
* multi-chain batch sampling introduces stronger correlations

Further debugging is planned.

---

# **9. Contributing**

Pull requests, tests, and extensions are welcome.

---

# **10. License**

MIT License.




