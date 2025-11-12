def linear(m,x,b):
    y = (m*x)+b
    return y
def slope_units(x_units,y_units):
    X = x_units.rstrip('s')
    Y = y_units.rstrip('s')
    return Y + '/' + X
def print_equation(m,b,y_units,x_units):
    print('Y =',m,slope_units(x_units,y_units),'x +',b,x_units.rstrip('s'))
