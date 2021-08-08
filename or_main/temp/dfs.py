def combine(terms, item, combinations):
    last = (len(terms) == 1)
    n = len(terms[0])
    for i in range(n):
        new_item = item + [terms[0][i]]
        if last:
            combinations.append(new_item)
        else:
            combine(terms[1:], new_item, combinations)


a = [
    [1, 2, 3],
    [10, 11, 12],
    [100, 101, 102]
]

combinations = []
combine(a, [], combinations)

for combination in combinations:
    print(combination)
