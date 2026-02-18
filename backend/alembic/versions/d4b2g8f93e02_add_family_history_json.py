"""add family_history json

Revision ID: d4b2g8f93e02
Revises: c3a1f7e82d01
Create Date: 2026-02-04 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4b2g8f93e02'
down_revision: Union[str, None] = 'c3a1f7e82d01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('questionnaires', sa.Column('family_history', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('questionnaires', 'family_history')
