#!/usr/bin/env python3


from typing import List, Optional

from instructor import OpenAISchema
from pydantic import BaseModel, Field, create_model


class AssessmentMetadata(BaseModel):
    """Metadata extracted from the assessment document"""

    resource: List[str] = Field(..., description="List of assessment resources")
    satellites: List[str] = Field(..., description="List of satellite names")
    full_solution: bool = Field(
        ...,
        description="True if the solution is available, False otherwise",
    )
    forthcoming: bool = Field(
        ...,
        description="True if the solution is anticipated to be available in the future",
    )
    limitation: str = Field(..., description="Limitations in assessment resources")
    missions: List[str] = Field(
        ...,
        description=" List of mission names for a given assessment ",
    )
    spatio_temporal_resolutions: List[str] = Field(
        ...,
        description="List of spatio temporal resolutions of data products",
    )


class NeedMetadata(BaseModel):
    """Metadata extracted from the needs document"""

    need_text: List[str] = Field(
        ...,
        description="List of needs  submitted by the agency. Should be concise and precise.",
    )
    phenomenon: List[str] = Field(
        ...,
        desciption="List of natural phenomena such as earthquakes, soil erosion, hurricanes, etc",
    )
    solution_category: List[str] = Field(..., description="List of solution categories")


class Metadata(OpenAISchema):
    """Metadata and entities extracted from the given document that consists of need and assessments data as well.
    Strictly don't generate unwanted metadata if not present.
    If entities can't be found, strictly don't extract them. Avoid outputing unknown values.
    """

    class Config:
        validation = False

    need_id: Optional[int] = Field(
        ...,
        description="id of need submitted by agency only if present in the document",
    )
    submitting_agency: Optional[str] = Field(
        ...,
        description="Name of agency/organization submitting the need document",
    )
    need_metadata: Optional[NeedMetadata] = Field(
        ...,
        description="Represents need metadata extracted from need document",
    )
    assessment_metadata: Optional[AssessmentMetadata] = Field(
        ...,
        description="Represents assessment metadata extracted from assessment document",
    )

    def dict_flattened(self, **kwargs):
        # Convert the main model to a dictionary
        flat_dict = super().dict(**kwargs)

        flat_dict.update(flat_dict.pop("need_metadata", {}))
        flat_dict.update(flat_dict.pop("assessment_metadata", {}))

        return flat_dict
