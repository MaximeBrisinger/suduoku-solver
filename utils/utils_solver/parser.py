def parse_carto(file_name):
    file = open(file_name).readlines()
    edges = []
    nb_nodes = int(file[0].split(" ")[-2])
    for line in file[1:]:
        line_split = line.split("\n")[0].split(" ")
        edges.append((int(line_split[1]), int(line_split[2])))
    return nb_nodes, edges


def parse_sudoku(file_name):
    file = open(file_name).readlines()
    pre_assigned = dict()
    for line in file:
        line_split = line.split(",")
        pre_assigned[(int(line_split[0]), int(line_split[1]))] = int(line_split[2])
    return pre_assigned

