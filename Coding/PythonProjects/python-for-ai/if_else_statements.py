# if else with multiple condition
age = 25
has_license = True

# Both must be True
if age >= 18 and has_license:
    print("You can drive!")

# At least one must be True
weekend = True
holiday = False
if weekend or holiday:
    print("No work today!")

# Reverse the condition
raining = False
if not raining:
    print("Let's go outside!")

#####################################\
# Nested if statement

has_ticket = True
age = 15

if has_ticket:
    if age >= 18:
        print("Enjoy the movie!")
    else:
        print("Need adult supervision")
else:
    print("Buy a ticket first")