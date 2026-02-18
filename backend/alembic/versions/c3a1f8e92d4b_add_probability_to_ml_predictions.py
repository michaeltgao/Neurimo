"""add probability to ml_predictions

Revision ID: c3a1f8e92d4b
Revises: b21b516ce070
Create Date: 2026-02-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3a1f8e92d4b"
down_revision: Union[str, None] = "b21b516ce070"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("ml_predictions", sa.Column("probability", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("ml_predictions", "probability")
