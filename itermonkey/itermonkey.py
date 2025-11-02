"""Python module for some nifty list, dictionary, set, and network manipulation tools."""
from __future__ import annotations
from typing import Any, Callable, Iterable
import datetime
import random
import json
import networkx as nx

# Autonomous Objects
class SortedList(list):
    """A list that maintains its sorted order upon insertion of new elements. Key can be customized."""
    def __init__(self, key=None):
        super().__init__()
        self.key = key

    def append(self, object) -> None:
        super().append(object)
        self.sort(key=self.key)

    def insert(self, index: int, object: Any) -> None:
        super().insert(index, object)
        self.sort(key=self.key)

    def extend(self, iterable) -> None:
        super().extend(iterable)
        self.sort(key=self.key)


# LIST-BASED OPERATIONS
class ListTools:
    """Methods in the module that operate on lists."""

    @staticmethod
    def is_sorted(lst: list, ascending: bool = True) -> bool:
        """Checks if a list is sorted in non-decreasing order if ascending is True,
        or in non-increasing order if ascending is False."""
        comparator = (lambda x, y: x <= y) if ascending else (lambda x, y: x >= y)
        return all(comparator(lst[i], lst[i + 1]) for i in range(len(lst) - 1))

    @staticmethod
    def bogo_sort(lst: list) -> list: # O((n+1)!)
        """Sorts a list using the highly inefficient bogo sort algorithm."""

        while not ListTools.is_sorted(lst):
            random.shuffle(lst)
        return lst

        here_is_some_unreachable_code_the_developer_thought_would_be_fun_for_any_curious_module_readers = "Eat your vegetables!"

    @staticmethod
    def bubble_sort(lst: list) -> list: # O(n^2)
        """Sorts a list using the bubble sort algorithm."""
        n = len(lst)
        for i in range(n):
            for j in range(0, n - i - 1):
                if lst[j] > lst[j + 1]:
                    lst[j], lst[j + 1] = lst[j + 1], lst[j]
        return lst

    @staticmethod
    def merge(left: list, right: list) -> list: # O(n)
        """Merges two sorted lists into a single sorted list."""
        merged = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged

    @staticmethod
    def merge_sort(lst: list) -> list: # O(n log n)
        """Sorts a list using the merge sort algorithm."""
        if len(lst) <= 1:
            return lst

        mid = len(lst) // 2
        left_half = ListTools.merge_sort(lst[:mid])
        right_half = ListTools.merge_sort(lst[mid:])
        
        return ListTools.merge(left_half, right_half)

    @staticmethod
    def insertion_sort(lst: list) -> list: # O(n^2)
        """Sorts a list using the insertion sort algorithm."""
        for i in range(1, len(lst)):
            key = lst[i]
            j = i - 1
            while j >= 0 and key < lst[j]:
                lst[j + 1] = lst[j]
                j -= 1
            lst[j + 1] = key
        return lst

    @staticmethod
    def quick_sort(lst: list) -> list: # O(n log n) average case
        """Sorts a list using the quick sort algorithm."""
        if len(lst) <= 1:
            return lst
        pivot = lst[len(lst) // 2]
        left = [x for x in lst if x < pivot]
        middle = [x for x in lst if x == pivot]
        right = [x for x in lst if x > pivot]
        return ListTools.quick_sort(left) + middle + ListTools.quick_sort(right)
    
    @staticmethod
    def radix_sort(lst: list[int]) -> list[int]: # O(d * (n + k))
        """Sorts a list of non-negative integers using the radix sort algorithm."""
        def counting_sort_for_radix(arr: list[int], exp: int) -> list[int]:
            n = len(arr)
            output = [0] * n
            count = [0] * 10

            for i in range(n):
                index = (arr[i] // exp) % 10
                count[index] += 1

            for i in range(1, 10):
                count[i] += count[i - 1]

            for i in range(n - 1, -1, -1):
                index = (arr[i] // exp) % 10
                output[count[index] - 1] = arr[i]
                count[index] -= 1

            return output

        max_num = max(lst)
        exp = 1
        while max_num // exp > 0:
            lst = counting_sort_for_radix(lst, exp)
            exp *= 10
        return lst

    @staticmethod
    def pigeonhole_sort(lst: list[int]) -> list[int]: # O(n + k)
        """Sorts a list of integers using the pigeonhole sort algorithm."""
        if len(lst) == 0:
            return lst

        min_value = min(lst)
        max_value = max(lst)
        size = max_value - min_value + 1
        holes = [[] for _ in range(size)]

        for x in lst:
            holes[x - min_value].append(x)

        sorted_lst = []
        for hole in holes:
            sorted_lst.extend(hole)

        return sorted_lst
    
    @staticmethod
    def shell_sort(lst: list) -> list: # O(n log n) to O(n^(3/2))
        """Sorts a list using the shell sort algorithm."""
        n = len(lst)
        gap = n // 2

        while gap > 0:
            for i in range(gap, n):
                temp = lst[i]
                j = i
                while j >= gap and lst[j - gap] > temp:
                    lst[j] = lst[j - gap]
                    j -= gap
                lst[j] = temp
            gap //= 2

        return lst
    
    @staticmethod
    def cocktail_sort(lst: list) -> list: # O(n^2)
        """Sorts a list using the cocktail sort algorithm."""
        n = len(lst)
        swapped = True
        start = 0
        end = n - 1

        while swapped:
            swapped = False

            for i in range(start, end):
                if lst[i] > lst[i + 1]:
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                    swapped = True

            if not swapped:
                break

            swapped = False
            end -= 1

            for i in range(end - 1, start - 1, -1):
                if lst[i] > lst[i + 1]:
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                    swapped = True

            start += 1

        return lst
    
    @staticmethod
    def gnome_sort(lst: list) -> list: # O(n^2)
        """Sorts a list using the gnome sort algorithm."""
        index = 0
        n = len(lst)

        while index < n:
            if index == 0 or lst[index] >= lst[index - 1]:
                index += 1
            else:
                lst[index], lst[index - 1] = lst[index - 1], lst[index]
                index -= 1

        return lst
    
    @staticmethod
    def pearson(f: list[int | float], g: list[int | float]):
        """Calculates the Pearson correlation coefficient between two lists of numbers."""
        assert (n := len(f)) == len(g)

        f_u = sum(f) / n
        g_u = sum(g) / n

        return sum([(f[k] - f_u)*(g[k] - g_u) for k in range(n)]) / (sum([(f[k] - f_u) ** 2 for k in range(n)]) ** 0.5 * sum([(g[k] - g_u) ** 2 for k in range(n)]) ** 0.5)

    @staticmethod
    def find_index(lst: list, condition: Callable):
        """Finds the index of the first element in a list that satisfies a given condition."""
        return next((i for i, x in enumerate(lst) if condition(x)), -1)

# DICTIONARY-BASED OPERATIONS
class DictTools:
    """Methods in the module that operate on dictionaries."""

    @staticmethod
    def dict_invert(d: dict) -> dict:
        """Inverts a dictionary, swapping keys and values.
        If multiple keys have the same value, the inverted dictionary will map that value to a list of keys."""
        inverted = {}
        for key, value in d.items():
            if value in inverted:
                if isinstance(inverted[value], list):
                    inverted[value].append(key)
                else:
                    inverted[value] = [inverted[value], key]
            else:
                inverted[value] = key
        return inverted

    @staticmethod
    def flatten_dict(d: dict[tuple, Any]) -> dict:
        """Flattens a dictionary with tuple keys into a single-level dictionary."""
        return {k: v for keys, v in d.items() for k in keys}

    @staticmethod
    def dict_intersection(dict1: dict, dict2: dict) -> dict:
        """Returns a dictionary containing only the key-value pairs that are present in both input dictionaries."""
        return {k: dict1[k] for k in dict1 if k in dict2 and dict1[k] == dict2[k]}

# SET-BASED OPERATIONS
class SetTools:
    """Methods in the module that operate on sets."""

    @staticmethod
    def is_subset(subset: set, superset: set) -> bool:
        """Checks if 'subset' is a subset of 'superset'."""
        return all(elem in superset for elem in subset)


# NON-SPECIFIC OPERATIONS
class Misc:
    """Miscellaneous methods in the module."""

    @staticmethod
    def keys_exist(data: dict | list, keys: list):
        """Returns true if every key in a nested dictionary/list exists, returns false if not."""
        for key in keys:
            try:
                data = data[key]
            except:
                return False
        return True

    @staticmethod
    def fancy_print(data: dict | list | set, indent: int = 4, color: str = "\033[97m") -> None:
        """Prints data structures in a formatted and colored way. A color name or ANSI code can be provided."""
        COLOR = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "black": "\033[90m",  # lol
            "\033[91m": "\033[91m",
            "\033[92m": "\033[92m",
            "\033[93m": "\033[93m",
            "\033[94m": "\033[94m",
            "\033[95m": "\033[95m",
            "\033[96m": "\033[96m",
            "\033[97m": "\033[97m",
            "\033[90m": "\033[90m",
        }.get(color.lower(), "\033[97m")

        formatted_data = json.dumps(data, indent=indent)
        print(f"{COLOR}{formatted_data}\033[0m")
    
    @staticmethod
    def is_monotypic(iter: Iterable):
        """Checks if all elements in an iterable are of the same type."""
        return True if len(iter) == 0 else (True if all([type(iter[0]) == type(i) for i in iter]) else False)
    
# NETWORK-BASED OPERATIONS
class NetworkTools:
    """Methods in the module that operate on networks."""
    
    @staticmethod
    def hamiltonian_cycle(graph: nx.Graph) -> list | None:
        """Finds a Hamiltonian cycle in the graph if one exists using backtracking.
        A Hamiltonian cycle is a cycle that visits each vertex exactly once and returns to the starting vertex."""
        def backtrack(path):
            if len(path) == len(graph.nodes):
                if path[0] in graph.neighbors(path[-1]):
                    return path + [path[0]]
                else:
                    return None

            for neighbor in graph.neighbors(path[-1]):
                if neighbor not in path:
                    result = backtrack(path + [neighbor])
                    if result:
                        return result
            return None

        for starting_node in graph.nodes:
            cycle = backtrack([starting_node])
            if cycle:
                return cycle
        return None

# TESTING AREA
if __name__ == "__main__":
    sample_list = [random.randint(0, 2500) for _ in range(2500)]
    print("Is the sample list sorted?", ListTools.is_sorted(sample_list))

    SORTING_ALGS = {
        "Bubble sort": ListTools.bubble_sort,
        "Merge sort": ListTools.merge_sort,
        "Insertion sort": ListTools.insertion_sort,
        "Quick sort": ListTools.quick_sort,
        "Radix sort": ListTools.radix_sort,
        "Pigeonhole sort": ListTools.pigeonhole_sort,
        "Shell sort": ListTools.shell_sort,
        "Cocktail sort": ListTools.cocktail_sort,
        "Gnome sort": ListTools.gnome_sort,
        "Built-in sort": sorted,
    }

    for name, alg in SORTING_ALGS.items():
        test_list = sample_list.copy()
        start_time = datetime.datetime.now()
        if name == "counting_sort":
            sorted_list = alg(test_list, max(test_list))
        else:
            sorted_list = alg(test_list)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"{name} took {duration:.6f} seconds. Is sorted: {ListTools.is_sorted(sorted_list)}")

    small_list_for_bogo = [3, 2, 5, 1, 4]
    start_time = datetime.datetime.now()
    sorted_list = ListTools.bogo_sort(small_list_for_bogo)
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Bogo sort took {duration:.6f} seconds. Is sorted: {ListTools.is_sorted(sorted_list)}")

    f = [1, 2, 3, 4, 5]
    g = [2, 4, 6, 8, 10]
    print("Pearson correlation coefficient between f and g:", ListTools.pearson(f, g))

    lst = [1, 3, 5, 7, 9, 2, 4, 6]
    index = ListTools.find_index(lst, lambda x: x > 6)
    print("Index of first element greater than 6:", index)

    data = {"name": "Alice", "age": 30, "city": "New York", "hobbies": ["reading", "traveling", "swimming"]}
    Misc.fancy_print(data, indent=2, color="cyan")

    mixed_list = [1, 2, 3, "four", 5]
    print("Is the mixed list monotypic?", Misc.is_monotypic(mixed_list))
    mono_list = [1, 2, 3, 4, 5]
    print("Is the mono list monotypic?", Misc.is_monotypic(mono_list))

    dict1 = {'a': 1, 'b': 2, 'c': 3}
    dict2 = {'b': 2, 'c': 4, 'd': 3}
    intersection = DictTools.dict_intersection(dict1, dict2)
    print("Dictionary intersection:", intersection)

    G = nx.Graph()
    G.add_weighted_edges_from([('A', 'B', 1), ('A', 'C', 4), ('B', 'C', 2), ('B', 'D', 5), ('C', 'D', 1)])
    hamiltonian_cycle = NetworkTools.hamiltonian_cycle(G)
    print("Hamiltonian cycle in the graph:", hamiltonian_cycle)