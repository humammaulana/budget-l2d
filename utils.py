def get_pos(landmark):
    return landmark.x, landmark.y

def get_line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = -1 * ((y2 * x1) - (y1 * x2)) / (x2 - x1)
    
    return m, b

def get_intersection(xa, ya, xb, yb, xc, yc, xd, yd):
    m1, b1 = get_line_equation(xa, ya, xb, yb)
    m2, b2 = get_line_equation(xc, yc, xd, yd)

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y

x, y = get_intersection(4,8,14,4,7,8,4,2)

print(f"{x}, {y}")