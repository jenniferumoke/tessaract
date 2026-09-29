import pytest
from pydantic import ValidationError

from tessaract import InputSchema, Property

def test_nullable_property_reports_both_types():
    prop = Property(type=["string", "null"])
    assert prop._types() == {"string", "null"}

def test_items_is_rejected_on_non_array_property():
    with pytest.raises(ValidationError, match="items is only valid"):
        Property(type="string", items=Property(type="string"))

@pytest.mark.parametrize(
    "field, value",
    [
        ("properties", {"a": Property(type="string")}),
        ("required", []),
        ("additionalProperties", False)
    ]
)

def test_object_only_fields_are_rejected_on_non_object(field, value):
    with pytest.raises(ValidationError, match="only valid on properties of type 'object'"):
        Property(type="string", **{field: value})

def test_input_schema_rejects_required_name_that_is_not_a_property():
    with pytest.raises(ValidationError, match="undefined properties"):
        InputSchema(properties={"city": Property(type="string")}, required=["country"])

def test_nullable_property_is_accepted():
    prop =  Property(
        type=["object", "null"],
        properties={"city": Property(type="string")},
        required=["city"]
    )

    assert prop._types() == {"object", "null"}
    assert prop.required == ["city"]