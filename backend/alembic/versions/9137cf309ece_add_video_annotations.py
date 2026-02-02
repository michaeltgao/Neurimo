"""add video annotations columns

Revision ID: 9137cf309ece
Revises: 504bcd3e91b8
Create Date: 2026-01-29
"""
from alembic import op
import sqlalchemy as sa

revision = "9137cf309ece"
down_revision = "504bcd3e91b8"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("videos", sa.Column("annotations_path", sa.String(), nullable=True))
    op.add_column("videos", sa.Column("annotations_version", sa.String(), nullable=True))

def downgrade():
    op.drop_column("videos", "annotations_version")
    op.drop_column("videos", "annotations_path")
