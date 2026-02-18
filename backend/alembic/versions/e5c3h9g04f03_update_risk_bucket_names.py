"""update risk bucket names to moderate terminology

Revision ID: e5c3h9g04f03
Revises: d4b2g8f93e02
Create Date: 2026-02-04 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e5c3h9g04f03'
down_revision: Union[str, None] = 'd4b2g8f93e02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update existing risk bucket names in ml_predictions table
    # medium -> moderate
    op.execute("UPDATE ml_predictions SET asd_risk_bucket = 'moderate' WHERE asd_risk_bucket = 'medium'")
    # med-high -> moderate-high
    op.execute("UPDATE ml_predictions SET asd_risk_bucket = 'moderate-high' WHERE asd_risk_bucket = 'med-high'")


def downgrade() -> None:
    # Revert back to old names
    op.execute("UPDATE ml_predictions SET asd_risk_bucket = 'medium' WHERE asd_risk_bucket = 'moderate'")
    op.execute("UPDATE ml_predictions SET asd_risk_bucket = 'med-high' WHERE asd_risk_bucket = 'moderate-high'")
