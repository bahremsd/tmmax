---
title: 'TMMax: High-performance modeling of multilayer thin-film structures using transfer matrix method with JAX'

tags:
  - Python
  - JAX
  - Optics
  - Photonics
  - Multilayer Thin-Film
  - Transfer Matrix Method

authors:
  - name: Bahrem Serhat Danis
    orcid: 0009-0002-9880-0446
    affiliation: "1"
  - name: Esra Zayim
    orcid: 0000-0001-5887-0293
    affiliation: "2"


affiliations:
 - name: Department of Electrical and Electronics Engineering, Koç University, Istanbul, 34450, Turkey
   index: 1
 - name: Physics Engineering Department, Istanbul Technical University, Istanbul, 34469, Turkey
   index: 2

date: 3 July 2025

bibliography: paper.bib

---

# Summary

Optical multilayer thin films are fundamental components that enable the precise control of reflectance, transmittance, and phase shift in the design of photonic systems. Rapid and accessible simulation of these structures is critically important for designing and analyzing complex coatings. While researchers commonly use the traditional transfer matrix method for designing these structures, its scalar approach to wavelength and angle of incidence causes redundant recalculations and inefficiencies in large-scale simulations. Furthermore, traditional method implementations do not support automatic differentiation, which limits their applicability in gradient-based inverse design approaches. Here, we present TMMax, a Python library that fully vectorizes and accelerates transfer matrix method using the high-performance machine learning library JAX. TMMax supports CPU, GPU, and TPU hardware, includes a publicly available material database. Our approach, demonstrated through benchmarking, allows us to model thin-film stacks with hundreds of layers within seconds. This illustrates that our method achieves a simulation speedup of up to hundreds of times over a baseline NumPy implementation. Our method enables optical engineers and thin-film researchers in optics and photonics to efficiently design complex dielectric multilayer structures through rapid and scalable simulations.

# Statement of need

The Transfer Matrix Method (TMM) models multilayer optical thin films by applying Snell’s law for light propagation and Fresnel equations to compute interface transmittance and reflectance.

\vspace{1.5em}
$$
\mathbf{M} = \mathbf{I}_0 \cdot  \prod_{i=1}^{N-2} \mathbf{M}_i \tag{1}
$$ 
\vspace{1.5em}

In TMM, the optical behavior of a multilayer structure composed of dielectric materials is obtained by computing the system matrix $\mathbf{M}$, as shown in Equation (1). This matrix calculation, commonly referred to as the Abeles TMM [@refId0], results from the successive multiplication of the transfer matrices of each layer ($\mathbf{M}_i$) [@katsidis2002general]. 

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth,keepaspectratio]{figure1.pdf}
\caption{Schematic of two strategies for calculating transmission, reflection, and absorption in multilayer thin-film simulations. The system (a) is modeled either by sequentially multiplying 2×2 transfer matrices for each wavelength and incidence angle (b) or by vectorizing these operations across both axes (c).}
\end{figure}


In traditional TMM implementations, the stack of layers in Figure 1a is simulated using a single wavelength and angle of incidence, as shown in Figure 1b, and nested loops over wavelengths and angles lead to redundant calculations [@byrnes2020multilayeropticalcalculations]. TMMax removes these redundancies by vectorizing wavelengths and angles and all intermediate TMM operations via JAX [@jax2018github]. As seen in the schematic of the vectorized implementation in Figure 1c, we vectorize all intermediate operations in TMM and subsequently apply the JAX’s just-in-time (JIT) decorator. Instead of running the mapped TMM code sequentially over each batch element of wavelength and angle of incidence, jax.jit fuses all operations across the batch into a single XLA-compiled [@xla2023github] kernel. This reduces function call overhead and provides a faster TMM implementation. TMMax replaces the conventional for-loop system-matrix calculation [@6131837] with JAX’s lax.scan, enabling JIT compilation and eliminating interpreter bottlenecks, while running efficiently on CPUs, GPUs, and TPUs without code changes. 

TMMax supports deep learning–based inverse design by keeping all computations on the GPU, avoiding costly CPU–GPU data transfers [@10.1117/1.OE.58.6.065103]. Whereas NumPy-based [@2020NumPy-Array] TMM packages that lack native gradients and require Autograd [@maclaurin2015autograd], TMMax natively computes gradients. Additionally, TMMax integrates a curated database of 30 extensively used dielectric materials, sourced from refractiveindex.info [@polyanskiy2024refractiveindex], thereby enabling optical engineers and thin-film researchers in optics and photonics to efficiently simulate complex multilayer structures through a scalable, JAX-accelerated implementation.

# Benchmarks

Runtime in TMM scales naturally with the number of layers, as well as the lengths of the wavelength and incidence-angle arrays, due to the increased number of transfer matrix multiplications. To benchmark TMMax, we used tmm library [@byrnes2020multilayeropticalcalculations] as a reference.

\begin{figure}[htbp]
\centering\includegraphics[width=0.75\textwidth]{figure2.pdf}
\caption{ Run time vs. layer count comparing tmm (orange) and TMMax (blue).}
\end{figure}

To assess how layer count affects computational performance, we sampled 20 multilayer structures ranging from 2 to 400 layers, with each layer randomly assigned one of seven materials and thicknesses between 100–500 nm. Spectral and angular domains were fixed at 20 points each, spanning 500–1000 nm and 0–π/2 radians, respectively. Figure 2 shows that while tmm runtime grows rapidly, TMMax scales efficiently, remaining nearly constant (~1.0–1.2 s) for low-layer structures and achieving speedups from 18× (2 layers) to 700× (400 layers).

\begin{figure*}[htbp]
\centering\includegraphics[width=\textwidth]{figure3.pdf}
\caption{The colormaps show the runtime performance of tmm and TMMax across varying simulation grid sizes, comparing 8- and 80-layer stacks in (a) and (b), respectively.}  
\end{figure*}
\vspace{1.0em}

We benchmarked the effects of wavelength and incident angle array sizes by sampling 20 values from 2 to 100, generating simulation grids from 2×2 to 100×100 for an 8-layer structure (Figure 3a). tmm runtime rises sharply with grid size, reaching ~138 s for 100×100, whereas TMMax remains below 3 s. For the smallest 2×2 grid, tmm is faster (~0.1 s vs. ~0.6 s) due to NumPy’s low overhead, while JAX incurs higher initialization costs. As layers increase to 80 (Figure 3b), tmm exceeds 760 s, but TMMax stays under 8 s, demonstrating superior efficiency and stability against both problem size and structural complexity.

We used Python’s timeit module to benchmark each simulation 50 times, with all comparisons run on a single Intel Core i9 core without GPU or multicore use for fairness.

# Installation

TMMax can be readily installed from the Python Package Index using `pip install tmmax`, which automatically handles all dependencies. For detailed installation instructions and platform compatibility, please refer to the [TMMax Documentation](https://tmmax.readthedocs.io/en/latest/index.html).

# Acknowledgements

This work was supported by the Scientific and Technological Research Council of Türkiye (TUBITAK) under the 2209-A Research Project Support Programme for Undergraduate Students, 2022 First-Term Call.

# References

