def zad1():
    print("Podaj imię : ")
    imie = str(input())
    print("Podaj ilość zrobionych zadań : ")
    il_zad = int(input())
    if il_zad > 3:
        print(imie + ", bardzo dobrze!")
    else:
        print(imie + ", musisz więcej popracować w domu.")


zad1()