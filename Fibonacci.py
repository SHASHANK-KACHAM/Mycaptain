numberofterms = int(input("How many terms? "))
num1=0
num2=1 
count=0
while count < numberofterms:
       print(num1)
       temp = num1 + num2
       num1 = num2
       num2 = temp
       count += 1
