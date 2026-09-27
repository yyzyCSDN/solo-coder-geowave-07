"""Slip-sweep / blended acquisition: schedules, operators and separation."""
from __future__ import annotations
from .schedule import (
    BlendingSchedule, BlendedRecordSpec, SourceActivation, ScheduleIssue,
    random_blending_schedule,
    COLLISION, DUPLICATE_SOURCE, MISSING_ACTIVATION,
    BAD_SOURCE_REF, DELAY_RANGE, EMPTY_RECORD,
)
from .operator import (
    BlendedData, InputIssue, forward_blend, forward_cube, pseudo_deblend,
    adjoint_cube, ata_crosstalk_cube, shared_receiver_layout,
    check_unblended_input, check_blended_input,
    MISSING_SOURCE_RECORD, MISSING_RECEIVER, MISSING_RECORD,
    BAD_SAMPLE_COUNT, BAD_DT, NON_FINITE,
)
from .separation import (
    SeparationStatus, SeparationConfig, SeparationResult, IterationRecord,
    CrosstalkReport, separate,
)

__all__ = [
    'BlendingSchedule', 'BlendedRecordSpec', 'SourceActivation', 'ScheduleIssue',
    'random_blending_schedule',
    'COLLISION', 'DUPLICATE_SOURCE', 'MISSING_ACTIVATION',
    'BAD_SOURCE_REF', 'DELAY_RANGE', 'EMPTY_RECORD',
    'BlendedData', 'InputIssue', 'forward_blend', 'forward_cube', 'pseudo_deblend',
    'adjoint_cube', 'ata_crosstalk_cube', 'shared_receiver_layout',
    'check_unblended_input', 'check_blended_input',
    'MISSING_SOURCE_RECORD', 'MISSING_RECEIVER', 'MISSING_RECORD',
    'BAD_SAMPLE_COUNT', 'BAD_DT', 'NON_FINITE',
    'SeparationStatus', 'SeparationConfig', 'SeparationResult', 'IterationRecord',
    'CrosstalkReport', 'separate',
]
