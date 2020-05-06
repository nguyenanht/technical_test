import typing as t

from marshmallow import Schema, fields
from marshmallow import ValidationError


class InvalidInputError(Exception):
    """Invalid model input."""


class FairMoneyDataRequestSchema(Schema):
    checking_balance = fields.Str()
    months_loan_duration = fields.Float()
    credit_history = fields.Str()
    purpose = fields.Str()
    amount = fields.Float()
    savings_balance = fields.Str()
    employment_length = fields.Str()
    installment_rate = fields.Float()
    personal_status = fields.Str()
    other_debtors = fields.Str()
    residence_history = fields.Float()
    property = fields.Str()
    age = fields.Float()
    installment_plan = fields.Str()
    housing = fields.Str()
    existing_credits = fields.Float()
    dependents = fields.Float()
    telephone = fields.Str()
    foreign_worker = fields.Str()
    job = fields.Str()


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = FairMoneyDataRequestSchema()

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors
