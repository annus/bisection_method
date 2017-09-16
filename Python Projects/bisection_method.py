import parser
import time
import argparse
from math import*

class equation(object):
    def __init__(self, eq_tion):
        expression_str = eq_tion.replace('^', '**')
        self.expression = parser.expr(expression_str).compile()

    def eval_exp(self, x):
        return eval(self.expression)

def new_interval(left, right, equation):
    p = float(left+right)/2.
    function_at_p = equation.eval_exp(x=p)
    if equation.eval_exp(x=left)*function_at_p < 0: return (function_at_p, p, left, p) 
    else: return (function_at_p, p, p, right) 

def bisection_method(left, right, equation, iterations=100):
    check_left = equation.eval_exp(x=left)
    check_right = equation.eval_exp(x=right)
    assert check_left*check_right < 0,\
        'f(left) = {:.4f} and f(right) = {:.4f} should have \
opposite signs'.format(check_left, check_right)
    p_value_list = []
    for i in range(iterations):
        value, p, left, right = new_interval(left, right, equation)    
        p_value_list.append(tuple((p, value)))
        verbose = 'iteration: {}, x: {}, f(x) = {}'.format(i, p, value)
        print(verbose)
        if abs(value) <= 1e-100: 
            print('log: Fully Converged to 0.0')
            break
        if len(p_value_list) < 2: continue
        if p_value_list[-1] == p_value_list[-2]:
            print('log: Fully Converged to decimal point limit')
            break
    print('')

def main():
    parse_man = argparse.ArgumentParser(description='evaluates the roots of any equation')
    parse_man.add_argument('--exp', '--expression', type=str, dest='expression', 
        help='expression to find the roots of')
    parse_man.add_argument('--ul', '--interval', type=float, dest='interval', nargs='+',
        help='an interval [a, b] such that f(a)*f(b) < 0')
    parse_man.add_argument('--it', '--iterations', type=int, dest='iterations')
    args = parse_man.parse_args()
    equat = equation(args.expression)
    left ,right = tuple(args.interval)
    bisection_method(left, right, equat, iterations=args.iterations)

if __name__ == '__main__':
    main()


