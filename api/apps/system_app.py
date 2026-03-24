#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
#  limitations under the License
#
import logging
from datetime import datetime
import json
from typing import Any

from api.apps import login_required, current_user

from api.db.db_models import APIToken
from api.db.services.api_service import APITokenService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import (
    get_json_result,
    get_data_error_result,
    server_error_response,
    generate_confirmation_token,
)
from common.versions import get_ragflow_version
from common.time_utils import current_timestamp, datetime_format
from timeit import default_timer as timer

from rag.utils.redis_conn import REDIS_CONN
from quart import jsonify
from pydantic import BaseModel, Field
from quart_schema import document_response, tag
from api.utils.health_utils import run_health_checks, get_oceanbase_status
from common import settings


class ErrorResponse(BaseModel):
    code: int
    message: str
    data: Any | None = None


class GenericSuccessResponse(BaseModel):
    code: int = 0
    data: Any | None = None
    message: str = "success"


class HealthComponentDoc(BaseModel):
    status: str | None = None
    elapsed: str | None = None
    type: str | None = None
    storage: str | None = None
    database: str | None = None
    error: str | None = None


class SystemStatusDoc(BaseModel):
    doc_engine: dict[str, Any] | None = None
    storage: HealthComponentDoc | None = None
    database: HealthComponentDoc | None = None
    redis: HealthComponentDoc | None = None
    task_executor_heartbeats: dict[str, Any] = Field(default_factory=dict)


class VersionResponseDoc(BaseModel):
    code: int = 0
    data: str
    message: str = "success"


class StatusResponseDoc(BaseModel):
    code: int = 0
    data: SystemStatusDoc
    message: str = "success"


class OceanBaseStatusResponseDoc(BaseModel):
    code: int = 0
    data: dict[str, Any]
    message: str = "success"


class ApiTokenRecordDoc(BaseModel):
    token: str
    beta: str | None = None
    name: str | None = None
    tenant_id: str | None = None
    create_time: int | None = None
    create_date: str | None = None
    update_time: int | None = None
    update_date: str | None = None


class ApiTokenListResponseDoc(BaseModel):
    code: int = 0
    data: list[ApiTokenRecordDoc]
    message: str = "success"


class ConfigResponseDataDoc(BaseModel):
    registerEnabled: bool | int
    disablePasswordLogin: bool | int


class ConfigResponseDoc(BaseModel):
    code: int = 0
    data: ConfigResponseDataDoc
    message: str = "success"


@manager.route("/version", methods=["GET"])  # noqa: F821
@login_required
@tag(["System"])
@document_response(VersionResponseDoc)
@document_response(ErrorResponse, 400)
def version():
    """Get the current version of the application."""
    return get_json_result(data=get_ragflow_version())


@manager.route("/status", methods=["GET"])  # noqa: F821
@login_required
@tag(["System"])
@document_response(StatusResponseDoc)
@document_response(ErrorResponse, 400)
def status():
    """Get the system status."""
    res = {}
    st = timer()
    try:
        res["doc_engine"] = settings.docStoreConn.health()
        res["doc_engine"]["elapsed"] = "{:.1f}".format((timer() - st) * 1000.0)
    except Exception as e:
        res["doc_engine"] = {
            "type": "unknown",
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    st = timer()
    try:
        settings.STORAGE_IMPL.health()
        res["storage"] = {
            "storage": settings.STORAGE_IMPL_TYPE.lower(),
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["storage"] = {
            "storage": settings.STORAGE_IMPL_TYPE.lower(),
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    st = timer()
    try:
        KnowledgebaseService.get_by_id("x")
        res["database"] = {
            "database": settings.DATABASE_TYPE.lower(),
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["database"] = {
            "database": settings.DATABASE_TYPE.lower(),
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    st = timer()
    try:
        if not REDIS_CONN.health():
            raise Exception("Lost connection!")
        res["redis"] = {
            "status": "green",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
        }
    except Exception as e:
        res["redis"] = {
            "status": "red",
            "elapsed": "{:.1f}".format((timer() - st) * 1000.0),
            "error": str(e),
        }

    task_executor_heartbeats = {}
    try:
        task_executors = REDIS_CONN.smembers("TASKEXE")
        now = datetime.now().timestamp()
        for task_executor_id in task_executors:
            heartbeats = REDIS_CONN.zrangebyscore(task_executor_id, now - 60 * 30, now)
            heartbeats = [json.loads(heartbeat) for heartbeat in heartbeats]
            task_executor_heartbeats[task_executor_id] = heartbeats
    except Exception:
        logging.exception("get task executor heartbeats failed!")
    res["task_executor_heartbeats"] = task_executor_heartbeats

    return get_json_result(data=res)


@manager.route("/healthz", methods=["GET"])  # noqa: F821
@tag(["System"])
@document_response(GenericSuccessResponse)
def healthz():
    result, all_ok = run_health_checks()
    return jsonify(result), (200 if all_ok else 500)


@manager.route("/ping", methods=["GET"])  # noqa: F821
@tag(["System"])
@document_response(GenericSuccessResponse)
async def ping():
    return "pong", 200


@manager.route("/oceanbase/status", methods=["GET"])  # noqa: F821
@login_required
@tag(["System"])
@document_response(OceanBaseStatusResponseDoc)
@document_response(ErrorResponse, 400)
def oceanbase_status():
    """Get OceanBase health status and performance metrics."""
    try:
        status_info = get_oceanbase_status()
        return get_json_result(data=status_info)
    except Exception as e:
        return get_json_result(
            data={
                "status": "error",
                "message": f"Failed to get OceanBase status: {str(e)}"
            },
            code=500
        )


@manager.route("/new_token", methods=["POST"])  # noqa: F821
@login_required
@tag(["API Tokens"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
def new_token():
    """Generate a new API token."""
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = [tenant for tenant in tenants if tenant.role == "owner"][0].tenant_id
        obj = {
            "tenant_id": tenant_id,
            "token": generate_confirmation_token(),
            "beta": generate_confirmation_token().replace("ragflow-", "")[:32],
            "create_time": current_timestamp(),
            "create_date": datetime_format(datetime.now()),
            "update_time": None,
            "update_date": None,
        }

        if not APITokenService.save(**obj):
            return get_data_error_result(message="Fail to new a dialog!")

        return get_json_result(data=obj)
    except Exception as e:
        return server_error_response(e)


@manager.route("/token_list", methods=["GET"])  # noqa: F821
@login_required
@tag(["API Tokens"])
@document_response(ApiTokenListResponseDoc)
@document_response(ErrorResponse, 400)
def token_list():
    """List all API tokens for the current user."""
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = [tenant for tenant in tenants if tenant.role == "owner"][0].tenant_id
        objs = APITokenService.query(tenant_id=tenant_id)
        objs = [o.to_dict() for o in objs]
        for o in objs:
            if not o["beta"]:
                o["beta"] = generate_confirmation_token().replace("ragflow-", "")[:32]
                APITokenService.filter_update([APIToken.tenant_id == tenant_id, APIToken.token == o["token"]], o)
        return get_json_result(data=objs)
    except Exception as e:
        return server_error_response(e)


@manager.route("/token/<token>", methods=["DELETE"])  # noqa: F821
@login_required
@tag(["API Tokens"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
def rm(token):
    """Remove an API token."""
    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")

        tenant_id = tenants[0].tenant_id
        APITokenService.filter_delete([APIToken.tenant_id == tenant_id, APIToken.token == token])
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route("/config", methods=["GET"])  # noqa: F821
@tag(["System"])
@document_response(ConfigResponseDoc)
def get_config():
    """Get system configuration."""
    return get_json_result(data={
        "registerEnabled": settings.REGISTER_ENABLED,
        "disablePasswordLogin": settings.DISABLE_PASSWORD_LOGIN,
    })
