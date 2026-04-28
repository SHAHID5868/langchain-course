from typing import Optional, NewType, TypedDict
from typing import Type

RGB = NewType("RGB", tuple[int, int , int])
HSL = NewType("HSL", tuple[int, int, int])


class user(TypedDict):
    first_name: str
    last_name: str
    email: str
    age: int | None
    fav_color: RGB | None


def create_user(first_name: str, last_name: str, age: int | None = None,fav_color: RGB| None= None) -> user:
    email = f"{first_name.lower()}_{last_name.lower()}@example.com"
    str_age = str(age)

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "age": str_age,
        "fav_color": fav_color,
    }

user1 = create_user("Corey","Schafer", age=38, fav_color=RGB((109,123,134)))
user2 = create_user("John","Doe",fav_color=HSL((206,10,48)))
print(user1)
print(user2)
