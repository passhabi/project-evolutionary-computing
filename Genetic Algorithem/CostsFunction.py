def sphere(x):
    # maximum and minimum of response boundary in each dimension:
    dimensions = 5
    min_boundary = -10
    max_boundary = +10

    function = sum(x ** 2)
    return function, min_boundary, max_boundary, dimensions
