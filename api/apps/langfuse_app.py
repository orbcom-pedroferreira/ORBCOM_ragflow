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


from api.apps import current_user, login_required
from langfuse import Langfuse
from pydantic import BaseModel, ConfigDict
from quart_schema import document_request, document_response, tag

from api.db.db_models import DB
from api.db.services.langfuse_service import TenantLangfuseService
from api.utils.api_utils import get_error_data_result, get_json_result, get_request_json, server_error_response, validate_request


class ErrorResponse(BaseModel):
    code: int
    message: str
    data: dict | None = None


class LangfuseApiKeyBodyDoc(BaseModel):
    secret_key: str
    public_key: str
    host: str


class LangfuseApiKeyRecordDoc(BaseModel):
    tenant_id: str | None = None
    secret_key: str | None = None
    public_key: str | None = None
    host: str | None = None
    project_id: str | None = None
    project_name: str | None = None
    model_config = ConfigDict(extra="allow")


class LangfuseApiKeyResponseDoc(BaseModel):
    code: int = 0
    data: LangfuseApiKeyRecordDoc | None = None
    message: str = "success"


class LangfuseDeleteResponseDoc(BaseModel):
    code: int = 0
    data: bool | None = None
    message: str = "success"


@manager.route("/api_key", methods=["POST", "PUT"])  # noqa: F821
@login_required
@validate_request("secret_key", "public_key", "host")
@tag(["Langfuse"])
@document_request(LangfuseApiKeyBodyDoc)
@document_response(LangfuseApiKeyResponseDoc)
@document_response(ErrorResponse, 400)
async def set_api_key():
    """Create or update Langfuse API keys."""
    req = await get_request_json()
    secret_key = req.get("secret_key", "")
    public_key = req.get("public_key", "")
    host = req.get("host", "")
    if not all([secret_key, public_key, host]):
        return get_error_data_result(message="Missing required fields")

    current_user_id = current_user.id
    langfuse_keys = dict(
        tenant_id=current_user_id,
        secret_key=secret_key,
        public_key=public_key,
        host=host,
    )

    langfuse = Langfuse(public_key=langfuse_keys["public_key"], secret_key=langfuse_keys["secret_key"], host=langfuse_keys["host"])
    if not langfuse.auth_check():
        return get_error_data_result(message="Invalid Langfuse keys")

    langfuse_entry = TenantLangfuseService.filter_by_tenant(tenant_id=current_user_id)
    with DB.atomic():
        try:
            if not langfuse_entry:
                TenantLangfuseService.save(**langfuse_keys)
            else:
                TenantLangfuseService.update_by_tenant(tenant_id=current_user_id, langfuse_keys=langfuse_keys)
            return get_json_result(data=langfuse_keys)
        except Exception as e:
            return server_error_response(e)


@manager.route("/api_key", methods=["GET"])  # noqa: F821
@login_required
@validate_request()
@tag(["Langfuse"])
@document_response(LangfuseApiKeyResponseDoc)
@document_response(ErrorResponse, 400)
def get_api_key():
    """Get Langfuse API keys for the current user."""
    current_user_id = current_user.id
    langfuse_entry = TenantLangfuseService.filter_by_tenant_with_info(tenant_id=current_user_id)
    if not langfuse_entry:
        return get_json_result(message="Have not record any Langfuse keys.")

    langfuse = Langfuse(public_key=langfuse_entry["public_key"], secret_key=langfuse_entry["secret_key"], host=langfuse_entry["host"])
    try:
        if not langfuse.auth_check():
            return get_error_data_result(message="Invalid Langfuse keys loaded")
    except langfuse.api.core.api_error.ApiError as api_err:
        return get_json_result(message=f"Error from Langfuse: {api_err}")
    except Exception as e:
        return server_error_response(e)

    langfuse_entry["project_id"] = langfuse.api.projects.get().dict()["data"][0]["id"]
    langfuse_entry["project_name"] = langfuse.api.projects.get().dict()["data"][0]["name"]

    return get_json_result(data=langfuse_entry)


@manager.route("/api_key", methods=["DELETE"])  # noqa: F821
@login_required
@validate_request()
@tag(["Langfuse"])
@document_response(LangfuseDeleteResponseDoc)
@document_response(ErrorResponse, 400)
def delete_api_key():
    """Delete Langfuse API keys for the current user."""
    current_user_id = current_user.id
    langfuse_entry = TenantLangfuseService.filter_by_tenant(tenant_id=current_user_id)
    if not langfuse_entry:
        return get_json_result(message="Have not record any Langfuse keys.")

    with DB.atomic():
        try:
            TenantLangfuseService.delete_model(langfuse_entry)
            return get_json_result(data=True)
        except Exception as e:
            return server_error_response(e)
