'''
1. Найти кассеты с наибольшим остатком банкнот, которые меньше или равны запрашиваемой сумме. Если несколько - выбрать с наибольшим номиналом
2. Определить, можно ли выдать всю сумму этим номиналом (хватает ли купюр в кассете или нет)
3. Проверить, не превышает ли кол-во купюр максимума возможного для выдачи. Превышает - пункт 1 без данной кассеты (если нет других подходящих - нет вариантов выдачи)
4. Рассчитываем целую возможную часть для выдачи. Можем всю сумму - выдаём
5. Если банкнот для выдачи больше чем в кассете раскладываем по min_notes с наибольшего номинала. Если не удалось разложить и нет других кассет - нельзя выдать
6. Выдаём одну купюру текущего номинала и пересчитываем следующее: 1) [Мак кол-во банкнот на выдачу] - [выданная банкнота];
                                                                    2) [Запрошенная сумма] - [выданная сумма];
                                                                    3) Банкнотный состав на выдачу
7. Если сумма > 0 - перейти к пункту 1, иначе всё выдано


min_notes
1.
'''

import unittest
from idlelib import run


from openpyxl import load_workbook
from itertools import count

def min_notes(nom, residue, a_sum, max_b):
    count = 0
    if not a_sum:
        return 0, residue
    for inx, i in enumerate(nom):
        while residue[i] and a_sum:
            a_sum -= i
            count += 1
            residue[i] -= 1
        if a_sum < 0:
            a_sum += i
            count -= 1
            residue[i] += 1
            continue
        elif a_sum == 0 and count < max_b:
            return count
        elif a_sum == 0 and count >= max_b:
            return 'no'

def result_nominal(true_ct, result, rem_b, summ=0, max_b=25):
    nominal_ct = max(true_ct, key=true_ct.get)
    summ -= nominal_ct
    count = min_notes(result, rem_b, summ, max_b)
    if count == 'no':
        true_ct_copy = true_ct.copy()
        true_ct_copy.pop(nominal_ct)
        summ += nominal_ct
        nominal_ct, result = result_nominal(true_ct_copy, result, rem_b, summ, max_b)
    else:
        result[nominal_ct] += 1

    return nominal_ct, result

def right_ct(ct, rem_b, summ=0):
    true_ct = dict()
    true_ct = {i: rem_b[i] for i in ct if i < summ and rem_b[i]}
    return true_ct


def take_this_b(ct, rem_b, result, summ, max_b):
    # # Максимум выдаваемых банкнот
    # max_b = 25
    # # # Запрошенная сумма
    # summ = 1600
    # input(summ)
    while summ:
        true_ct = right_ct(ct, rem_b, summ)
        rem_b_copy = rem_b.copy()
        nominal_ct, result = result_nominal(true_ct, result, rem_b_copy, summ, max_b)
        max_b -= 1
        summ -= nominal_ct
        rem_b[nominal_ct] -= 1

    return rem_b, result

def main(ct, rem_b, summ, max_b):
# def main(ct, rem_b, summ, max_b):
    # Номинал
    # ct = (100, 50, 10, 5)
    # # Банкнот в кассетах
    # rem_b = {100: 1000, 50: 1988, 10: 1985, 5: 1983}
    print(rem_b)
    # max_b = 25
    # # Запрошенная сумма
    # summ = 1600
    # Банкнот на выдачу
    result = {100: 0, 50: 0, 20: 0, 10: 0, 5: 0}
    rem_b, result = take_this_b(ct, rem_b, result, summ, max_b)

    return result


#
class Test(unittest.TestCase):

    def test_right_ct(self):
        table = load_workbook('tests.xlsx')
        sheet = table.active
        named_range = sheet['A3:R781']
        i = 0
        for row in named_range:
            print(i)

            if not i == 9 and not i == 12 and not i == 15:
                ct = (row[0].value, row[1].value, row[2].value, row[3].value)
                rem_b = {row[0].value: row[4].value, row[1].value: row[5].value, row[2].value: row[6].value, row[3].value: row[7].value}
                max_b = row[9].value
                summ = row[10].value
                result = {row[0].value: row[13].value, row[1].value: row[14].value, row[2].value: row[15].value, row[3].value: row[16].value}
                self.assertEqual(main(ct, rem_b, summ, max_b), result)
            i += 1

if __name__ == "__main__":
    unittest.main()
#
# if __name__ == "__main__":
#     run.main()