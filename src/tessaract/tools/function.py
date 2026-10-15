from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

PropertyType: TypeAlias = Literal["string", "number", "integer", "boolean", "array", "object", "null"]


class Property(BaseModel):
    '''A JSON Schema property.

        type: a single type, or a list of types such as ["string", "null"] for a nullable value
        enum: allowed values
        items: the schema of each element when type is "array"
        properties / required / additionalProperties: the nested schema when type is "object"
    '''
    type: PropertyType | list[PropertyType]
    description: str | None = None
    enum: list[Any] | None = None
    items: "Property | None" = None
    properties: dict[str, "Property"] | None = None
    required: list[str] | None = None
    additionalProperties: bool | None = None

    def _types(self) -> set[str]:
        return set(self.type) if isinstance(self.type, list) else {self.type}

    @model_validator(mode="after")
    def _model_validator(self):
        types = self._types()

        if self.items is not None and "array" not in types:
            raise ValueError("items is only valid on properties of type 'array'")

        object_fields = (self.properties, self.required, self.additionalProperties)
        if any(value is not None for value in object_fields) and "object" not in types:
            raise ValueError(
                "properties, required and additionalProperties are only valid on properties of type 'object'"
            )

        invalid = set(self.required or []) - set(self.properties or {})
        if invalid:
            raise ValueError(
                f"required contains undefined properties: {invalid}"
            )

        return self

class InputSchema(BaseModel):
    model_config = ConfigDict(strict=True)
    type: Literal["object"] = "object"
    properties: dict[str, Property] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _model_validator(self):
        invalid = set(self.required) - set(self.properties)

        if invalid:
            raise ValueError(
                f"required contains undefined properties: {invalid}"
            )

        return self

class FunctionTool(BaseModel):
    name: str
    description: str
    input_schema: InputSchema | None = None
    strict: bool | None = None
    cache: bool = False
    provider_options: dict[str, Any] = Field(default_factory=dict)
