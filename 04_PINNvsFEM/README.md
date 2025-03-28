# Discussion on Physical Information Neural Network (PINN) and Finite element method (FEM) in geochemical thermal-chemical-mechanical multi-field coupling problems

## Abstract

In this study, we discussed the applications of Finite Element Method (FEM) and Physics-Informed Neural Networks (PINN) in addressing Thermo-Chemical-Mechanical (TCM) multi-field coupling problems in geochemistry. These problems involve strong interactions among thermal conduction, chemical reaction dynamics, material transport, and mechanical deformation, which are critical for understanding geodynamic processes, reservoir evolution, and mineral reaction kinetics. The paper provides a comprehensive comparison between FEM and PINN, highlighting their respective strengths, limitations, and applicability to TCM problems. Practical examples, such as geothermal systems and CO₂ geological storage, are used to illustrate the advantages of each method. The study concludes by emphasizing the complementary nature of FEM and PINN and proposes potential directions for integrating the two approaches for enhanced modeling capabilities.

### Keywords:
Thermo-Chemical-Mechanical (TCM), Finite Element Method (FEM), Physics-Informed Neural Networks (PINN), multi-field coupling, geochemistry, strong coupling, multi-scale modeling, dynamic boundary conditions, nonlinear behavior.

## Introduction

Thermo-Chemical-Mechanical (TCM) multi-field coupling problems play a significant role in geochemical research, with applications in geodynamics, underground reservoir evolution, and mineral reaction kinetics. These problems are characterized by strong field interactions, multi-scale phenomena, dynamic boundary conditions, and nonlinear behaviors. For instance, temperature influences chemical reaction rates (e.g., Arrhenius equation), while chemical reactions can alter material properties and induce mechanical deformation. Similarly, temperature changes can cause thermal expansion, generating stress that affects thermal conductivity.

To solve such complex problems, the paper focuses on two widely used numerical methods: Finite Element Method (FEM) and Physics-Informed Neural Networks (PINN). FEM is a mature, grid-based approach known for its high accuracy, stability, and extensive application in engineering and scientific fields. In contrast, PINN leverages deep learning to solve physical problems without grid discretization, making it highly flexible for complex geometries, high-dimensional problems, and data-driven scenarios.

This paper systematically analyzes the advantages and limitations of FEM and PINN in TCM problems. It also identifies current challenges, such as handling strong coupling, multi-scale interactions, and nonlinear behaviors. By comparing the methods and exploring their integration, the study aims to provide insights into solving geochemical TCM problems more efficiently and accurately.