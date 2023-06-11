# SIMD LOWESS utilities for Python

This is a minimal python library for using local regression, also known as 
Locally Weighted Scatterplot Smoothing ([LOWESS](https://en.wikipedia.org/wiki/Local_regression)).

In my day-to-day work I use this extensively and just out of sheer repetition I 
need something that is as fast as possible, so I'm making use of [OpenMP](www.openmp.org) 
and [AVX](https://en.wikipedia.org/wiki/AVX-512) instructions. 
