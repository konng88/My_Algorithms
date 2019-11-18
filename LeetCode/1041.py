def isRobotBounded(instructions) -> bool:
    instr_list = list(instructions)
    direction = 0
    position = [0,0]
    for i in instr_list:
        if i == 'L':
            direction += 1
        elif i == 'R':
            direction -= 1
        else:
            if direction % 4 == 1:
                position[0] -= 1
            elif direction % 4 == 3:
                position[0] += 1
            elif direction % 4 == 0:
                position[1] += 1
            elif direction % 4 == 2:
                position[1] -= 1
    return direction % 4 != 0 or position == [0,0]


print(isRobotBounded("GLRLLGLL"))
