"""recalculate risk buckets from probability

Revision ID: f6d4i0h15g04
Revises: e5c3h9g04f03
Create Date: 2026-02-04 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f6d4i0h15g04'
down_revision: Union[str, None] = 'e5c3h9g04f03'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Recalculate risk buckets based on probability values
    # This fixes any mismatched bucket assignments
    # Buckets: low (0-25), moderate (26-50), moderate-high (51-75), high (76-100)

    # low: probability <= 0.25
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'low'
        WHERE probability IS NOT NULL AND probability <= 0.25
    """)

    # moderate: probability > 0.25 AND <= 0.50
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'moderate'
        WHERE probability IS NOT NULL AND probability > 0.25 AND probability <= 0.50
    """)

    # moderate-high: probability > 0.50 AND <= 0.75
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'moderate-high'
        WHERE probability IS NOT NULL AND probability > 0.50 AND probability <= 0.75
    """)

    # high: probability > 0.75
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'high'
        WHERE probability IS NOT NULL AND probability > 0.75
    """)


def downgrade() -> None:
    # Can't reliably downgrade as we don't know what the original buckets were
    pass
