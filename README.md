# decor
Spreading code design: optimizing binary spreading codes, or pseudorandom noise
(PRN) codes to have low auto-correlation and cross-correlation. Contains
efficient implementations of bit-flip descent methods that iteratively
*decorrelate* a set of codes.

Accompanies the paper:

A. Yang, T. Mina, S. Boyd, and G. Gao. Large-Scale GNSS Spreading Code Optimization. 
Proceedings of the 37th International Technical Meeting of the Satellite Division 
of the Institute of Navigation (ION GNSS+ 2024)

Paper link: https://stanford.edu/~boyd/papers/code_design.html

## Overview
`decor` is a Python package for designing and optimizing binary spreading codes,
also known as pseudorandom noise (PRN) codes. These codes are optimized to have
low auto-correlation and cross-correlation properties, making them suitable for
applications in communication systems, such as CDMA and GPS.

## Features
- Generate binary spreading codes
- Optimize codes for low auto-correlation and cross-correlation
- Gold and Weil Codes

## Usage

Example usage:
```
import decor

# Generate 63 random binary spreading codes, each length 1023
# objective function with parameter p=6
codes = decor.random_code_family(63, 1023, p=6)

# Generate Gold codes
gold_codes = decor.gold_code_family(63, 1023, p=6)

# evaluate correlation values
codes.correlation()

# evaluate objective function
codes.objective()

# optimize codes
optimizer = AdaptiveKGreedyCodeOptimizer()
optimizer.optimize(codes, n_iter=1000)
```

The `scripts` directory contains an `example.py` file. To run the example, use:
```
python3 -m scripts.example
```

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
