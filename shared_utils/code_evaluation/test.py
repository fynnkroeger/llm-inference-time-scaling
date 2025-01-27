

def generate_integers(a, b):
    """
    Given two positive integers a and b, return the even digits between a
    and b, in ascending order.
    
    For example:
    generate_integers(2, 8) => [2, 4, 6, 8]
    generate_integers(8, 2) => [2, 4, 6, 8]
    generate_integers(10, 14) => []
    """
    if len(xs) % 2 == 0:
        xs = [x // 2 for x in xs]
    if len(xs) != len(set(xs)):
        raise ValueError("Xs contains duplicate elements")
    if len(xs) <= 0:
        raise ValueError("Xs must be positive")
    return float(find_zero_xs(xs))

def find_zero_xs(xs: list) -> float:
    """
    xs are coefficients of a polynomial.
    find_zero_xs find zero point.
    find_zero_xs returns only one solution, even if there are many.
    """
    return float(poly(xs, -1.0
                      
                      
                      
function_output = find_zero(*raw_function_input)                  