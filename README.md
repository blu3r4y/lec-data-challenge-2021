# Team Blue Pressure - LEC Data Challenge 2021

[![License](https://img.shields.io/badge/License-AGPL%203.0-yellow?style=popout-square)](LICENSE.txt)

This is the source code of 🔵 **Team Blue Pressure** for the [LEC Data Challenge 2021](https://www.lec.at/research-area-2/lec-data-challenge-2021/?lang=en/).

## Installation

Create the `conda` environment, install `kedro`, and let it install all the other dependencies.

```
conda create -n ldc2021 python=3.8 conda-forge::kedro=0.17.4
conda activate ldc2021
kedro install
```

## Usage

Run the following pipelines to compute the data

```
kedro run --parallel --pipeline clean
kedro run --parallel --pipeline features
```

Run any of the models

```
kedro run --pipeline ridge
kedro run --pipeline mlp
```
