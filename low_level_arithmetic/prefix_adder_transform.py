from typing import Iterable, List, Tuple, Set, Dict


def prefix_nodes_to_ranges(nodes: Iterable[Tuple[int, int, int]]) -> Set[Tuple[int, int]]:
    """
    Transform a prefix-sum/adder node list into the set of (upper, lower) ranges each node covers.

    Each input item is (p0, p1, level). The node’s identity is the rightmost index max(p0, p1).
    If a parent hasn't been produced by an earlier node, it's treated as a singleton range (i, i).

    Returns:
        A set of (upper, lower) tuples for all nodes created.
    """
    # Normalize to list and process by level, then by parents for determinism (not required for correctness).
    nodes_list: List[Tuple[int, int, int]] = list(nodes)
    nodes_list.sort(key=lambda x: (x[2], min(x[0], x[1]), max(x[0], x[1])))

    # Map "node id" -> (upper, lower) range. Base case: unseen parent i means (i, i).
    id_to_range: Dict[int, Tuple[int, int]] = {}

    out: Set[Tuple[int, int]] = set()
    for p0, p1, _lvl in nodes_list:
        r0 = id_to_range.get(p0, (p0, p0))
        r1 = id_to_range.get(p1, (p1, p1))

        upper = max(r0[0], r1[0])  # same as max(p0, p1)
        lower = min(r0[1], r1[1])

        node_id = max(p0, p1)  # rightmost index labels the produced node
        id_to_range[node_id] = (upper, lower)
        out.add((upper, lower))

    return out


data_list_24 = [[0, 1, 1],
                [2, 3, 1],
                [4, 5, 1],
                [6, 7, 1],
                [8, 9, 1],
                [10, 11, 1],
                [12, 13, 1],
                [14, 15, 1],
                [16, 17, 1],
                [18, 19, 1],
                [20, 21, 1],
                [22, 23, 1],
                [1, 3, 2],
                [5, 7, 2],
                [9, 11, 2],
                [13, 15, 2],
                [17, 19, 2],
                [3, 7, 3],
                [11, 15, 3],
                [7, 15, 4],
                [7, 11, 5],
                [15, 19, 5],
                [3, 5, 6],
                [7, 9, 6],
                [11, 13, 6],
                [15, 17, 6],
                [19, 21, 6],
                [1, 2, 7],
                [3, 4, 7],
                [5, 6, 7],
                [7, 8, 7],
                [9, 10, 7],
                [11, 12, 7],
                [13, 14, 7],
                [15, 16, 7],
                [17, 18, 7],
                [19, 20, 7],
                [21, 22, 7],
                [21, 23, 7]]

data_list_32 = [[0, 1, 1],
               [2, 3, 1],
               [4, 5, 1],
               [6, 7, 1],
               [8, 9, 1],
               [10, 11, 1],
               [12, 13, 1],
               [14, 15, 1],
               [16, 17, 1],
               [18, 19, 1],
               [20, 21, 1],
               [22, 23, 1],
               [24, 25, 1],
               [26, 27, 1],
               [28, 29, 1],
               [30, 31, 1],
               [1, 3, 2],
               [3, 5, 3],
               [7, 9, 2],
               [9, 11, 3],
               [13, 15, 2],
               [15, 17, 3],
               [19, 21, 2],
               [21, 23, 3],
               [25, 27, 2],
               [27, 29, 3],
               [5, 11, 4],
               [17, 23, 4],
               [11, 17, 5],
               [11, 23, 5],
               [5, 7, 6],
               [5, 9, 6],
               [11, 13, 6],
               [11, 15, 6],
               [17, 19, 6],
               [17, 21, 6],
               [23, 25, 6],
               [23, 27, 6],
               [23, 29, 6],
               [1, 2, 7],
               [3, 4, 7],
               [5, 6, 7],
               [7, 8, 7],
               [9, 10, 7],
               [11, 12, 7],
               [13, 14, 7],
               [15, 16, 7],
               [17, 18, 7],
               [19, 20, 7],
               [21, 22, 7],
               [23, 24, 7],
               [25, 26, 7],
               [27, 28, 7],
               [29, 30, 7],
               [29, 31, 7]]

def get_multiscan_nodes_24() -> Set[Tuple[int, int]]:
    inp = data_list_24
    nodes = prefix_nodes_to_ranges(inp)
    return nodes

def get_multiscan_nodes_32() -> Set[Tuple[int, int]]:
    inp = data_list_32
    nodes = prefix_nodes_to_ranges(inp)
    return nodes



# data_list = [[0, 1, 1], [2, 3, 1], [4, 5, 1], [6, 7, 1], [1, 3, 2], [3, 5, 3], [1, 2, 4], [3, 4, 4], [5, 6, 4], [5, 7, 4]]


def test_prefix_merge_ranges():

    inp = data_list_24
    nodes = prefix_nodes_to_ranges(inp)
    print(nodes)
    print(len(data_list_24))
    print(len(nodes))
    # -> {(4, 0), (7, 0), (2, 0), (5, 4), (3, 0), (5, 0), (7, 6), (6, 0), (1, 0), (3, 2)}

if __name__ == "__main__":
    test_prefix_merge_ranges()
