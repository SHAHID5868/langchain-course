from functools import partial
from modulefinder import test
from operator import gt
from re import U
from pydantic import BaseModel, Field, ValidationError, EmailStr, HttpUrl, SecretStr, field_validator, model_validator, ValidationInfo
from uuid import uuid4, UUID
from datetime import UTC, datetime
from typing import Literal, Annotated
#using Literal status can only have these 3 values as input.
#using Annotated we can give conditions that the vaue has to be more or less than.

class User(BaseModel):
    # Uid: Annotated[int, Field(gt=0)] #using Annotated we have give condition that it should be greater than zero.
    Uid: UUID = Field(default_factory=uuid4)
    username: Annotated[str,Field(min_length=3, max_length=20)]
    email: EmailStr
    age: Annotated[int, Field(ge=13, le=130)] # ge -> Greather Than , le -> less Than, gt -> Greater Than
    website: HttpUrl | None = None
    password: SecretStr
    verified_at: datetime | None = None
    bio: str | None = None
    is_active: bool = True
    full_name: str | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores allowed)")
        return v.lower()
    
    @field_validator("website")
    @classmethod
    def add_https(cls, v: str | None) -> str | None:
        if  v and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v

# try:

#    user1 = User(Uid="test", username =corey, email="coreudfesdd@gmail.com", age= 36)
#    # print(user1.model_dump())
#    print(user1.model_dump_json(indent=2))
# except  ValidationError as e:
#     print(e)
class BlogPost(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=200)]
    content: Annotated[str, Field(min_length=10)]
    view_count: int = 0
    is_published: bool = False
    tag: list[str] = Field(default_factory=list)
    create_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    # create_at: datetime = Field(default_factory=partial(datetime.now(tz=UTC)))
    author_id:  str | int
    status: Literal["draft", "published", "archived"] = "draft" #using Literal status can only have these 3 values as input.
    slug: Annotated[str, Field(pattern=r"^[a-z0-9-]+$")]



# try:
#     post = BlogPost(title="Getting Started with Pydantic", content="Here's how to begin...", author_id="12345")
#     print(post)
# except ValidationError as e:
#     print(e)

# try:
#     user = User(
#         username="coreyms",
#         email="coreyMs@gmail.com",
#         age=14,
#         password="Secret123"
#     )
#     print(user)
# except ValidationError as e:
#     print(e)

user = User(
         username="coreyms_Schafer",
         email="coreyMs@gmail.com",
         age=14,
         password="Secret123",
         website="corey.com"
)
print(user)
# print(user.password.get_secret_value())


