"""fix bucket assignments using rounded scores

Revision ID: g7e5j1i26h05
Revises: f6d4i0h15g04
Create Date: 2026-02-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'g7e5j1i26h05'
down_revision: Union[str, None] = 'f6d4i0h15g04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Recalculate risk buckets using ROUND(probability * 100) to match display
    # This ensures that a displayed score of 75 is always moderate-high

    # low: score <= 25
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'low'
        WHERE probability IS NOT NULL AND ROUND(probability * 100) <= 25
    """)

    # moderate: score 26-50
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'moderate'
        WHERE probability IS NOT NULL AND ROUND(probability * 100) > 25 AND ROUND(probability * 100) <= 50
    """)

    # moderate-high: score 51-75
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'moderate-high'
        WHERE probability IS NOT NULL AND ROUND(probability * 100) > 50 AND ROUND(probability * 100) <= 75
    """)

    # high: score > 75
    op.execute("""
        UPDATE ml_predictions
        SET asd_risk_bucket = 'high'
        WHERE probability IS NOT NULL AND ROUND(probability * 100) > 75
    """)


def downgrade() -> None:
    # Can't reliably downgrade as we don't know what the original buckets were
    pass
