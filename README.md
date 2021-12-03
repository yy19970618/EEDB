# Overview

We proposed three methodologies to enhance the existing learned database management works. We test our methods by the two kinds of widely used learned structure, deep learning(EEDL) and reinforcement learning(EERL). 

# Environment Preparation

Python 3.7

TensorFlow 2.3

PostgreSQL 12.6

Pytorch 1.1

Hypopg

# Run Code



1. Generate datasets (Census, Forest, Power, DMV)  from https://github.com/sfu-db/AreCELearnedYet 

2. Generate corresponding workloads for above four datasets based on the code from https://github.com/sfu-db/AreCELearnedYet

3. Download the open source benchmark TPC-H from http://www.tpc.org/tpch/.

4. Generate workloads according to the query templates of TPC-H.

5. Complete the pre-training and re-training of EEDL by using this script. 
`python  ./eedl/main.py`

6. Initial the environment and train the EERL model by using this script.
`python  ./eerl/main.py`
