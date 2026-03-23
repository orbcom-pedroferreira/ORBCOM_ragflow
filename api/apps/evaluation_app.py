#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
RAG Evaluation API Endpoints

Provides REST API for RAG evaluation functionality including:
- Dataset management
- Test case management
- Evaluation execution
- Results retrieval
- Configuration recommendations
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from quart import request
from quart_schema import document_querystring, document_request, document_response, tag
from api.apps import login_required, current_user
from api.db.services.evaluation_service import EvaluationService
from api.utils.api_utils import (
    get_data_error_result,
    get_json_result,
    get_request_json,
    server_error_response,
    validate_request
)
from common.constants import RetCode


class ErrorResponse(BaseModel):
    code: int = 0
    data: Any | None = None
    message: str


class GenericSuccessResponse(BaseModel):
    code: int = 0
    data: Any | None = None
    message: str = "success"


class EvaluationDatasetCreateBodyDoc(BaseModel):
    name: str = Field(..., description="Dataset name.")
    description: str | None = Field(default=None, description="Dataset description.")
    kb_ids: list[str] = Field(..., description="Knowledge base IDs.")


class EvaluationDatasetListQueryDoc(BaseModel):
    page: int | None = Field(default=None, description="Page number.")
    page_size: int | None = Field(default=None, description="Items per page.")


class EvaluationDatasetUpdateBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, description="Dataset name.")
    description: str | None = Field(default=None, description="Dataset description.")
    kb_ids: list[str] | None = Field(default=None, description="Knowledge base IDs.")


class EvaluationCaseAddBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(..., description="Test question.")
    reference_answer: str | None = Field(default=None, description="Reference answer.")
    relevant_doc_ids: list[str] | None = Field(default=None, description="Relevant document IDs.")
    relevant_chunk_ids: list[str] | None = Field(default=None, description="Relevant chunk IDs.")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata.")


class EvaluationCaseImportBodyDoc(BaseModel):
    cases: list[dict[str, Any]] = Field(..., description="Test cases to import.")


class EvaluationRunStartBodyDoc(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID.")
    dialog_id: str = Field(..., description="Dialog ID.")
    name: str | None = Field(default=None, description="Run name.")


class EvaluationRunListQueryDoc(BaseModel):
    dataset_id: str | None = Field(default=None, description="Dataset ID filter.")
    dialog_id: str | None = Field(default=None, description="Dialog ID filter.")
    page: int | None = Field(default=None, description="Page number.")
    page_size: int | None = Field(default=None, description="Items per page.")


class EvaluationCompareBodyDoc(BaseModel):
    run_ids: list[str] = Field(..., description="Run IDs to compare.")


class EvaluationSingleBodyDoc(BaseModel):
    question: str = Field(..., description="Question to evaluate.")
    dialog_id: str = Field(..., description="Dialog ID.")
    reference_answer: str | None = Field(default=None, description="Reference answer.")
    relevant_chunk_ids: list[str] | None = Field(default=None, description="Relevant chunk IDs.")


# ==================== Dataset Management ====================

@manager.route('/dataset/create', methods=['POST'])  # noqa: F821
@login_required
@validate_request("name", "kb_ids")
@tag(["Evaluation Datasets"])
@document_request(EvaluationDatasetCreateBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def create_dataset():
    """
    Create a new evaluation dataset.
    
    Request body:
    {
        "name": "Dataset name",
        "description": "Optional description",
        "kb_ids": ["kb_id1", "kb_id2"]
    }
    """
    try:
        req = await get_request_json()
        name = req.get("name", "").strip()
        description = req.get("description", "")
        kb_ids = req.get("kb_ids", [])
        
        if not name:
            return get_data_error_result(message="Dataset name cannot be empty")
        
        if not kb_ids or not isinstance(kb_ids, list):
            return get_data_error_result(message="kb_ids must be a non-empty list")
        
        success, result = EvaluationService.create_dataset(
            name=name,
            description=description,
            kb_ids=kb_ids,
            tenant_id=current_user.id,
            user_id=current_user.id
        )
        
        if not success:
            return get_data_error_result(message=result)
        
        return get_json_result(data={"dataset_id": result})
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/list', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Datasets"])
@document_querystring(EvaluationDatasetListQueryDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def list_datasets():
    """
    List evaluation datasets for current tenant.
    
    Query params:
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20)
    """
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 20))
        
        result = EvaluationService.list_datasets(
            tenant_id=current_user.id,
            user_id=current_user.id,
            page=page,
            page_size=page_size
        )
        
        return get_json_result(data=result)
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/<dataset_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Datasets"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def get_dataset(dataset_id):
    """Get dataset details by ID"""
    try:
        dataset = EvaluationService.get_dataset(dataset_id)
        if not dataset:
            return get_data_error_result(
                message="Dataset not found",
                code=RetCode.DATA_ERROR
            )
        
        return get_json_result(data=dataset)
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/<dataset_id>', methods=['PUT'])  # noqa: F821
@login_required
@tag(["Evaluation Datasets"])
@document_request(EvaluationDatasetUpdateBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def update_dataset(dataset_id):
    """
    Update dataset.
    
    Request body:
    {
        "name": "New name",
        "description": "New description",
        "kb_ids": ["kb_id1", "kb_id2"]
    }
    """
    try:
        req = await get_request_json()
        
        # Remove fields that shouldn't be updated
        req.pop("id", None)
        req.pop("tenant_id", None)
        req.pop("created_by", None)
        req.pop("create_time", None)
        
        success = EvaluationService.update_dataset(dataset_id, **req)
        
        if not success:
            return get_data_error_result(message="Failed to update dataset")
        
        return get_json_result(data={"dataset_id": dataset_id})
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/<dataset_id>', methods=['DELETE'])  # noqa: F821
@login_required
@tag(["Evaluation Datasets"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def delete_dataset(dataset_id):
    """Delete dataset (soft delete)"""
    try:
        success = EvaluationService.delete_dataset(dataset_id)
        
        if not success:
            return get_data_error_result(message="Failed to delete dataset")
        
        return get_json_result(data={"dataset_id": dataset_id})
    except Exception as e:
        return server_error_response(e)


# ==================== Test Case Management ====================

@manager.route('/dataset/<dataset_id>/case/add', methods=['POST'])  # noqa: F821
@login_required
@validate_request("question")
@tag(["Evaluation Cases"])
@document_request(EvaluationCaseAddBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def add_test_case(dataset_id):
    """
    Add a test case to a dataset.
    
    Request body:
    {
        "question": "Test question",
        "reference_answer": "Optional ground truth answer",
        "relevant_doc_ids": ["doc_id1", "doc_id2"],
        "relevant_chunk_ids": ["chunk_id1", "chunk_id2"],
        "metadata": {"key": "value"}
    }
    """
    try:
        req = await get_request_json()
        question = req.get("question", "").strip()
        
        if not question:
            return get_data_error_result(message="Question cannot be empty")
        
        success, result = EvaluationService.add_test_case(
            dataset_id=dataset_id,
            question=question,
            reference_answer=req.get("reference_answer"),
            relevant_doc_ids=req.get("relevant_doc_ids"),
            relevant_chunk_ids=req.get("relevant_chunk_ids"),
            metadata=req.get("metadata")
        )
        
        if not success:
            return get_data_error_result(message=result)
        
        return get_json_result(data={"case_id": result})
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/<dataset_id>/case/import', methods=['POST'])  # noqa: F821
@login_required
@validate_request("cases")
@tag(["Evaluation Cases"])
@document_request(EvaluationCaseImportBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def import_test_cases(dataset_id):
    """
    Bulk import test cases.
    
    Request body:
    {
        "cases": [
            {
                "question": "Question 1",
                "reference_answer": "Answer 1",
                ...
            },
            {
                "question": "Question 2",
                ...
            }
        ]
    }
    """
    try:
        req = await get_request_json()
        cases = req.get("cases", [])
        
        if not cases or not isinstance(cases, list):
            return get_data_error_result(message="cases must be a non-empty list")
        
        success_count, failure_count = EvaluationService.import_test_cases(
            dataset_id=dataset_id,
            cases=cases
        )
        
        return get_json_result(data={
            "success_count": success_count,
            "failure_count": failure_count,
            "total": len(cases)
        })
    except Exception as e:
        return server_error_response(e)


@manager.route('/dataset/<dataset_id>/cases', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Cases"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def get_test_cases(dataset_id):
    """Get all test cases for a dataset"""
    try:
        cases = EvaluationService.get_test_cases(dataset_id)
        return get_json_result(data={"cases": cases, "total": len(cases)})
    except Exception as e:
        return server_error_response(e)


@manager.route('/case/<case_id>', methods=['DELETE'])  # noqa: F821
@login_required
@tag(["Evaluation Cases"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def delete_test_case(case_id):
    """Delete a test case"""
    try:
        success = EvaluationService.delete_test_case(case_id)
        
        if not success:
            return get_data_error_result(message="Failed to delete test case")
        
        return get_json_result(data={"case_id": case_id})
    except Exception as e:
        return server_error_response(e)


# ==================== Evaluation Execution ====================

@manager.route('/run/start', methods=['POST'])  # noqa: F821
@login_required
@validate_request("dataset_id", "dialog_id")
@tag(["Evaluation Runs"])
@document_request(EvaluationRunStartBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def start_evaluation():
    """
    Start an evaluation run.
    
    Request body:
    {
        "dataset_id": "dataset_id",
        "dialog_id": "dialog_id",
        "name": "Optional run name"
    }
    """
    try:
        req = await get_request_json()
        dataset_id = req.get("dataset_id")
        dialog_id = req.get("dialog_id")
        name = req.get("name")
        
        success, result = EvaluationService.start_evaluation(
            dataset_id=dataset_id,
            dialog_id=dialog_id,
            user_id=current_user.id,
            name=name
        )
        
        if not success:
            return get_data_error_result(message=result)
        
        return get_json_result(data={"run_id": result})
    except Exception as e:
        return server_error_response(e)


@manager.route('/run/<run_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Runs"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def get_evaluation_run(run_id):
    """Get evaluation run details"""
    try:
        result = EvaluationService.get_run_results(run_id)
        
        if not result:
            return get_data_error_result(
                message="Evaluation run not found",
                code=RetCode.DATA_ERROR
            )
        
        return get_json_result(data=result)
    except Exception as e:
        return server_error_response(e)


@manager.route('/run/<run_id>/results', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Runs"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def get_run_results(run_id):
    """Get detailed results for an evaluation run"""
    try:
        result = EvaluationService.get_run_results(run_id)
        
        if not result:
            return get_data_error_result(
                message="Evaluation run not found",
                code=RetCode.DATA_ERROR
            )
        
        return get_json_result(data=result)
    except Exception as e:
        return server_error_response(e)


@manager.route('/run/list', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Runs"])
@document_querystring(EvaluationRunListQueryDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def list_evaluation_runs():
    """
    List evaluation runs.
    
    Query params:
    - dataset_id: Filter by dataset (optional)
    - dialog_id: Filter by dialog (optional)
    - page: Page number (default: 1)
    - page_size: Items per page (default: 20)
    """
    try:
        # TODO: Implement list_runs in EvaluationService
        return get_json_result(data={"runs": [], "total": 0})
    except Exception as e:
        return server_error_response(e)


@manager.route('/run/<run_id>', methods=['DELETE'])  # noqa: F821
@login_required
@tag(["Evaluation Runs"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def delete_evaluation_run(run_id):
    """Delete an evaluation run"""
    try:
        # TODO: Implement delete_run in EvaluationService
        return get_json_result(data={"run_id": run_id})
    except Exception as e:
        return server_error_response(e)


# ==================== Analysis & Recommendations ====================

@manager.route('/run/<run_id>/recommendations', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Analysis"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def get_recommendations(run_id):
    """Get configuration recommendations based on evaluation results"""
    try:
        recommendations = EvaluationService.get_recommendations(run_id)
        return get_json_result(data={"recommendations": recommendations})
    except Exception as e:
        return server_error_response(e)


@manager.route('/compare', methods=['POST'])  # noqa: F821
@login_required
@validate_request("run_ids")
@tag(["Evaluation Analysis"])
@document_request(EvaluationCompareBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def compare_runs():
    """
    Compare multiple evaluation runs.
    
    Request body:
    {
        "run_ids": ["run_id1", "run_id2", "run_id3"]
    }
    """
    try:
        req = await get_request_json()
        run_ids = req.get("run_ids", [])
        
        if not run_ids or not isinstance(run_ids, list) or len(run_ids) < 2:
            return get_data_error_result(
                message="run_ids must be a list with at least 2 run IDs"
            )
        
        # TODO: Implement compare_runs in EvaluationService
        return get_json_result(data={"comparison": {}})
    except Exception as e:
        return server_error_response(e)


@manager.route('/run/<run_id>/export', methods=['GET'])  # noqa: F821
@login_required
@tag(["Evaluation Analysis"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def export_results(run_id):
    """Export evaluation results as JSON/CSV"""
    try:
        # format_type = request.args.get("format", "json")  # TODO: Use for CSV export
        
        result = EvaluationService.get_run_results(run_id)
        
        if not result:
            return get_data_error_result(
                message="Evaluation run not found",
                code=RetCode.DATA_ERROR
            )
        
        # TODO: Implement CSV export
        return get_json_result(data=result)
    except Exception as e:
        return server_error_response(e)


# ==================== Real-time Evaluation ====================

@manager.route('/evaluate_single', methods=['POST'])  # noqa: F821
@login_required
@validate_request("question", "dialog_id")
@tag(["Evaluation Analysis"])
@document_request(EvaluationSingleBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def evaluate_single():
    """
    Evaluate a single question-answer pair in real-time.
    
    Request body:
    {
        "question": "Test question",
        "dialog_id": "dialog_id",
        "reference_answer": "Optional ground truth",
        "relevant_chunk_ids": ["chunk_id1", "chunk_id2"]
    }
    """
    try:
        # req = await get_request_json()  # TODO: Use for single evaluation implementation
        
        # TODO: Implement single evaluation
        # This would execute the RAG pipeline and return metrics immediately
        
        return get_json_result(data={
            "answer": "",
            "metrics": {},
            "retrieved_chunks": []
        })
    except Exception as e:
        return server_error_response(e)
