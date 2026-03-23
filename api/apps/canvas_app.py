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
#  limitations under the License.
#
import copy
import inspect
import json
import logging
from functools import partial
from typing import Any
from quart import request, Response, make_response
from pydantic import BaseModel, ConfigDict, Field
from quart_schema import document_querystring, document_request, document_response, tag
from agent.component import LLM
from api.db import CanvasCategory
from api.db.services.canvas_service import CanvasTemplateService, UserCanvasService, API4ConversationService
from api.db.services.document_service import DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.pipeline_operation_log_service import PipelineOperationLogService
from api.db.services.task_service import queue_dataflow, CANVAS_DEBUG_DOC_ID, TaskService
from api.db.services.user_service import TenantService
from api.db.services.user_canvas_version import UserCanvasVersionService
from common.constants import RetCode
from common.misc_utils import get_uuid, thread_pool_exec
from api.utils.api_utils import (
    get_json_result,
    server_error_response,
    validate_request,
    get_data_error_result,
    get_request_json,
)
from agent.canvas import Canvas
from peewee import MySQLDatabase, PostgresqlDatabase
from api.db.db_models import APIToken, Task

from rag.flow.pipeline import Pipeline
from rag.nlp import search
from rag.utils.redis_conn import REDIS_CONN
from common import settings
from api.apps import login_required, current_user
from api.apps.services.canvas_replica_service import CanvasReplicaService
from api.db.services.canvas_service import completion as agent_completion


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int = 0
    data: Any | None = None
    message: str = "string"


class SuccessResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int = 0
    data: Any | None = None
    message: str = "success"


class CanvasRemoveBodyDoc(BaseModel):
    canvas_ids: list[str]


class CanvasSaveBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    dsl: Any
    title: str
    id: str | None = None
    release: bool | None = None
    canvas_category: str | None = None
    description: str | None = None
    permission: str | None = None
    avatar: str | None = None


class CanvasCompletionBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    query: str | None = ""
    files: list[Any] | None = None
    inputs: dict[str, Any] | None = None
    user_id: str | None = None


class CanvasAgentCompletionBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    return_trace: bool | None = False


class CanvasRerunBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    dsl: Any
    component_id: str


class CanvasResetBodyDoc(BaseModel):
    id: str


class CanvasUploadQueryDoc(BaseModel):
    url: str | None = Field(default=None, description="Optional source URL for upload metadata.")


class CanvasUploadBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: Any | None = Field(default=None, description="File or files to upload.")


class CanvasInputFormQueryDoc(BaseModel):
    id: str | None = Field(default=None, description="Canvas ID.")
    component_id: str | None = Field(default=None, description="Component ID.")


class CanvasDebugBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    component_id: str
    params: dict[str, Any]


class CanvasDbConnectBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    db_type: str
    database: str
    username: str
    host: str
    port: int | str
    password: str


class CanvasListQueryDoc(BaseModel):
    keywords: str | None = Field(default=None, description="Optional keyword filter.")
    page: int | None = Field(default=0, description="Page number.")
    page_size: int | None = Field(default=0, description="Items per page.")
    orderby: str | None = Field(default="create_time", description="Sort field.")
    desc: bool | None = Field(default=True, description="Whether to sort descending.")
    canvas_category: str | None = Field(default=None, description="Optional canvas category filter.")
    owner_ids: str | None = Field(default=None, description="Comma-separated owner IDs.")


class CanvasSettingBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    permission: str
    description: str | None = None
    avatar: str | None = None


class CanvasTraceQueryDoc(BaseModel):
    canvas_id: str | None = Field(default=None, description="Canvas ID.")
    message_id: str | None = Field(default=None, description="Message ID.")


class CanvasSessionsQueryDoc(BaseModel):
    user_id: str | None = Field(default=None, description="Filter by user ID.")
    page: int | None = Field(default=1, description="Page number.")
    page_size: int | None = Field(default=30, description="Items per page.")
    keywords: str | None = Field(default=None, description="Optional keyword filter.")
    from_date: str | None = Field(default=None, description="Start date filter.")
    to_date: str | None = Field(default=None, description="End date filter.")
    orderby: str | None = Field(default="update_time", description="Sort field.")
    desc: bool | None = Field(default=True, description="Whether to sort descending.")
    exp_user_id: str | None = Field(default=None, description="Experimental user ID.")
    dsl: bool | None = Field(default=True, description="Whether to include DSL in session records.")


class CanvasSessionCreateBodyDoc(BaseModel):
    name: str | None = None


class CanvasDownloadQueryDoc(BaseModel):
    id: str | None = Field(default=None, description="File ID.")
    created_by: str | None = Field(default=None, description="Creator user ID.")


@manager.route('/templates', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvas Templates"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def templates():
    return get_json_result(data=[c.to_dict() for c in CanvasTemplateService.get_all()])


@manager.route('/rm', methods=['POST'])  # noqa: F821
@validate_request("canvas_ids")
@login_required
@tag(["Canvases"])
@document_request(CanvasRemoveBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def rm():
    req = await get_request_json()
    for i in req["canvas_ids"]:
        if not UserCanvasService.accessible(i, current_user.id):
            return get_json_result(
                data=False, message='Only owner of canvas authorized for this operation.',
                code=RetCode.OPERATING_ERROR)
        UserCanvasService.delete_by_id(i)
    return get_json_result(data=True)


@manager.route('/set', methods=['POST'])  # noqa: F821
@validate_request("dsl", "title")
@login_required
@tag(["Canvases"])
@document_request(CanvasSaveBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def save():
    req = await get_request_json()
    req['release'] = bool(req.get("release", ""))
    try:
        req["dsl"] = CanvasReplicaService.normalize_dsl(req["dsl"])
    except ValueError as e:
        return get_data_error_result(message=str(e))
    cate = req.get("canvas_category", CanvasCategory.Agent)
    if "id" not in req:
        req["user_id"] = current_user.id
        if UserCanvasService.query(user_id=current_user.id, title=req["title"].strip(), canvas_category=cate):
            return get_data_error_result(message=f"{req['title'].strip()} already exists.")
        req["id"] = get_uuid()
        if not UserCanvasService.save(**req):
            return get_data_error_result(message="Fail to save canvas.")
    else:
        if not UserCanvasService.accessible(req["id"], current_user.id):
            return get_json_result(
                data=False, message='Only owner of canvas authorized for this operation.',
                code=RetCode.OPERATING_ERROR)
        UserCanvasService.update_by_id(req["id"], req)
    # save version
    UserCanvasVersionService.save_or_replace_latest(
        user_canvas_id=req["id"],
        dsl=req["dsl"],
        title=UserCanvasVersionService.build_version_title(getattr(current_user, "nickname", current_user.id), req.get("title")),
        release=req.get("release"),
    )
    replica_ok = CanvasReplicaService.replace_for_set(
        canvas_id=req["id"],
        tenant_id=str(current_user.id),
        runtime_user_id=str(current_user.id),
        dsl=req["dsl"],
        canvas_category=req.get("canvas_category", cate),
        title=req.get("title", ""),
    )
    if not replica_ok:
        return get_data_error_result(message="canvas saved, but replica sync failed.")
    return get_json_result(data=req)


@manager.route('/get/<canvas_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvases"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def get(canvas_id):
    if not UserCanvasService.accessible(canvas_id, current_user.id):
        return get_data_error_result(message="canvas not found.")
    e, c = UserCanvasService.get_by_canvas_id(canvas_id)
    if not e:
        return get_data_error_result(message="canvas not found.")
    try:
        # DELETE
        CanvasReplicaService.bootstrap(
            canvas_id=canvas_id,
            tenant_id=str(current_user.id),
            runtime_user_id=str(current_user.id),
            dsl=c.get("dsl"),
            canvas_category=c.get("canvas_category", CanvasCategory.Agent),
            title=c.get("title", ""),
        )
    except ValueError as e:
        return get_data_error_result(message=str(e))

    # Get the last publication time (latest released version's update_time)
    last_publish_time = None
    versions = UserCanvasVersionService.list_by_canvas_id(canvas_id)
    if versions:
        released_versions = [v for v in versions if v.release]
        if released_versions:
            # Sort by update_time descending and get the latest
            released_versions.sort(key=lambda x: x.update_time, reverse=True)
            last_publish_time = released_versions[0].update_time

    # Add last_publish_time to response data
    if isinstance(c, dict):
        c["last_publish_time"] = last_publish_time
    else:
        # If c is a model object, convert to dict first
        c = c.to_dict()
        c["last_publish_time"] = last_publish_time

    # For pipeline type, get associated datasets
    if c.get("canvas_category") == CanvasCategory.DataFlow:
        datasets = list(KnowledgebaseService.query(pipeline_id=canvas_id))
        c["datasets"] = [{"id": d.id, "name": d.name, "avatar": d.avatar} for d in datasets]

    return get_json_result(data=c)


@manager.route('/getsse/<canvas_id>', methods=['GET'])  # type: ignore # noqa: F821
@tag(["Canvas Execution"])
@document_response(ErrorResponse, 400)
def getsse(canvas_id):
    token = request.headers.get('Authorization').split()
    if len(token) != 2:
        return get_data_error_result(message='Authorization is not valid!')
    token = token[1]
    objs = APIToken.query(beta=token)
    if not objs:
        return get_data_error_result(message='Authentication error: API key is invalid!"')
    tenant_id = objs[0].tenant_id
    if not UserCanvasService.query(user_id=tenant_id, id=canvas_id):
        return get_json_result(
            data=False,
            message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR
        )
    e, c = UserCanvasService.get_by_id(canvas_id)
    if not e or c.user_id != tenant_id:
        return get_data_error_result(message="canvas not found.")
    return get_json_result(data=c.to_dict())


@manager.route('/completion', methods=['POST'])  # noqa: F821
@validate_request("id")
@login_required
@tag(["Canvas Execution"])
@document_request(CanvasCompletionBodyDoc)
@document_response(ErrorResponse, 400)
async def run():
    req = await get_request_json()
    query = req.get("query", "")
    files = req.get("files", [])
    inputs = req.get("inputs", {})
    tenant_id = str(current_user.id)
    runtime_user_id = req.get("user_id") or tenant_id
    user_id = str(runtime_user_id)
    if not await thread_pool_exec(UserCanvasService.accessible, req["id"], tenant_id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)

    replica_payload = CanvasReplicaService.load_for_run(
        canvas_id=req["id"],
        tenant_id=tenant_id,
        runtime_user_id=user_id,
    )

    if not replica_payload:
        return get_data_error_result(message="canvas replica not found, please call /get/<canvas_id> first.")

    replica_dsl = replica_payload.get("dsl", {})
    canvas_title = replica_payload.get("title", "")
    canvas_category = replica_payload.get("canvas_category", CanvasCategory.Agent)
    dsl_str = json.dumps(replica_dsl, ensure_ascii=False)

    _, cvs = await thread_pool_exec(UserCanvasService.get_by_id, req["id"])
    if cvs.canvas_category == CanvasCategory.DataFlow:
        task_id = get_uuid()
        Pipeline(dsl_str, tenant_id=tenant_id, doc_id=CANVAS_DEBUG_DOC_ID, task_id=task_id, flow_id=req["id"])
        ok, error_message = await thread_pool_exec(queue_dataflow, user_id, req["id"], task_id, CANVAS_DEBUG_DOC_ID, files[0], 0)
        if not ok:
            return get_data_error_result(message=error_message)
        return get_json_result(data={"message_id": task_id})

    try:
        canvas = Canvas(dsl_str, tenant_id, canvas_id=req["id"])
    except Exception as e:
        return server_error_response(e)

    async def sse():
        nonlocal canvas, user_id
        try:
            async for ans in canvas.run(query=query, files=files, user_id=user_id, inputs=inputs):
                yield "data:" + json.dumps(ans, ensure_ascii=False) + "\n\n"

            commit_ok = CanvasReplicaService.commit_after_run(
                canvas_id=req["id"],
                tenant_id=tenant_id,
                runtime_user_id=user_id,
                dsl=json.loads(str(canvas)),
                canvas_category=canvas_category,
                title=canvas_title,
            )
            if not commit_ok:
                logging.error(
                    "Canvas runtime replica commit failed: canvas_id=%s tenant_id=%s runtime_user_id=%s",
                    req["id"],
                    tenant_id,
                    user_id,
                )

        except Exception as e:
            logging.exception(e)
            canvas.cancel_task()
            yield "data:" + json.dumps({"code": 500, "message": str(e), "data": False}, ensure_ascii=False) + "\n\n"

    resp = Response(sse(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    #resp.call_on_close(lambda: canvas.cancel_task())
    return resp


@manager.route("/<canvas_id>/completion", methods=["POST"])  # noqa: F821
@login_required
@tag(["Canvas Execution"])
@document_request(CanvasAgentCompletionBodyDoc)
@document_response(ErrorResponse, 400)
async def exp_agent_completion(canvas_id):
    tenant_id = current_user.id
    req = await get_request_json()
    return_trace = bool(req.get("return_trace", False))
    async def generate():
        trace_items = []
        async for answer in agent_completion(tenant_id=tenant_id, agent_id=canvas_id, **req):
            if isinstance(answer, str):
                try:
                    ans = json.loads(answer[5:])  # remove "data:"
                except Exception:
                    continue

            event = ans.get("event")
            if event == "node_finished":
                if return_trace:
                    data = ans.get("data", {})
                    trace_items.append(
                        {
                            "component_id": data.get("component_id"),
                            "trace": [copy.deepcopy(data)],
                        }
                    )
                    ans.setdefault("data", {})["trace"] = trace_items
                    answer = "data:" + json.dumps(ans, ensure_ascii=False) + "\n\n"
                yield answer

            if event not in ["message", "message_end"]:
                continue

            yield answer

        yield "data:[DONE]\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers.add_header("Cache-control", "no-cache")
    resp.headers.add_header("Connection", "keep-alive")
    resp.headers.add_header("X-Accel-Buffering", "no")
    resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
    return resp
    

@manager.route('/rerun', methods=['POST'])  # noqa: F821
@validate_request("id", "dsl", "component_id")
@login_required
@tag(["Canvas Execution"])
@document_request(CanvasRerunBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def rerun():
    req = await get_request_json()
    doc = PipelineOperationLogService.get_documents_info(req["id"])
    if not doc:
        return get_data_error_result(message="Document not found.")
    doc = doc[0]
    if 0 < doc["progress"] < 1:
        return get_data_error_result(message=f"`{doc['name']}` is processing...")

    if settings.docStoreConn.index_exist(search.index_name(current_user.id), doc["kb_id"]):
        settings.docStoreConn.delete({"doc_id": doc["id"]}, search.index_name(current_user.id), doc["kb_id"])
    doc["progress_msg"] = ""
    doc["chunk_num"] = 0
    doc["token_num"] = 0
    DocumentService.clear_chunk_num_when_rerun(doc["id"])
    DocumentService.update_by_id(id, doc)
    TaskService.filter_delete([Task.doc_id == id])

    dsl = req["dsl"]
    dsl["path"] = [req["component_id"]]
    PipelineOperationLogService.update_by_id(req["id"], {"dsl": dsl})
    queue_dataflow(tenant_id=current_user.id, flow_id=req["id"], task_id=get_uuid(), doc_id=doc["id"], priority=0, rerun=True)
    return get_json_result(data=True)


@manager.route('/cancel/<task_id>', methods=['PUT'])  # noqa: F821
@login_required
@tag(["Canvas Execution"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def cancel(task_id):
    try:
        REDIS_CONN.set(f"{task_id}-cancel", "x")
    except Exception as e:
        logging.exception(e)
    return get_json_result(data=True)


@manager.route('/reset', methods=['POST'])  # noqa: F821
@validate_request("id")
@login_required
@tag(["Canvas Execution"])
@document_request(CanvasResetBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def reset():
    req = await get_request_json()
    if not UserCanvasService.accessible(req["id"], current_user.id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)
    try:
        e, user_canvas = UserCanvasService.get_by_id(req["id"])
        if not e:
            return get_data_error_result(message="canvas not found.")

        canvas = Canvas(json.dumps(user_canvas.dsl), current_user.id, canvas_id=user_canvas.id)
        canvas.reset()
        req["dsl"] = json.loads(str(canvas))
        UserCanvasService.update_by_id(req["id"], {"dsl": req["dsl"]})
        return get_json_result(data=req["dsl"])
    except Exception as e:
        return server_error_response(e)


@manager.route("/upload/<canvas_id>", methods=["POST"])  # noqa: F821
@tag(["Canvas Execution"])
@document_querystring(CanvasUploadQueryDoc)
@document_request(CanvasUploadBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def upload(canvas_id):
    e, cvs = UserCanvasService.get_by_canvas_id(canvas_id)
    if not e:
        return get_data_error_result(message="canvas not found.")

    user_id = cvs["user_id"]
    files = await request.files
    file_objs = files.getlist("file") if files and files.get("file") else []
    try:
        if len(file_objs) == 1:
            return get_json_result(data=FileService.upload_info(user_id, file_objs[0], request.args.get("url")))
        results = [FileService.upload_info(user_id, f) for f in file_objs]
        return get_json_result(data=results)
    except Exception as e:
        return server_error_response(e)


@manager.route('/input_form', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvas Utilities"])
@document_querystring(CanvasInputFormQueryDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def input_form():
    cvs_id = request.args.get("id")
    cpn_id = request.args.get("component_id")
    try:
        e, user_canvas = UserCanvasService.get_by_id(cvs_id)
        if not e:
            return get_data_error_result(message="canvas not found.")
        if not UserCanvasService.query(user_id=current_user.id, id=cvs_id):
            return get_json_result(
                data=False, message='Only owner of canvas authorized for this operation.',
                code=RetCode.OPERATING_ERROR)

        canvas = Canvas(json.dumps(user_canvas.dsl), current_user.id, canvas_id=user_canvas.id)
        return get_json_result(data=canvas.get_component_input_form(cpn_id))
    except Exception as e:
        return server_error_response(e)


@manager.route('/debug', methods=['POST'])  # noqa: F821
@validate_request("id", "component_id", "params")
@login_required
@tag(["Canvas Utilities"])
@document_request(CanvasDebugBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def debug():
    req = await get_request_json()
    if not UserCanvasService.accessible(req["id"], current_user.id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)
    try:
        e, user_canvas = UserCanvasService.get_by_id(req["id"])
        canvas = Canvas(json.dumps(user_canvas.dsl), current_user.id, canvas_id=user_canvas.id)
        canvas.reset()
        canvas.message_id = get_uuid()
        component = canvas.get_component(req["component_id"])["obj"]
        component.reset()

        if isinstance(component, LLM):
            component.set_debug_inputs(req["params"])
        component.invoke(**{k: o["value"] for k,o in req["params"].items()})
        outputs = component.output()
        for k in outputs.keys():
            if isinstance(outputs[k], partial):
                txt = ""
                iter_obj = outputs[k]()
                if inspect.isasyncgen(iter_obj):
                    async for c in iter_obj:
                        txt += c
                else:
                    for c in iter_obj:
                        txt += c
                outputs[k] = txt
        return get_json_result(data=outputs)
    except Exception as e:
        return server_error_response(e)


@manager.route('/test_db_connect', methods=['POST'])  # noqa: F821
@validate_request("db_type", "database", "username", "host", "port", "password")
@login_required
@tag(["Canvas Utilities"])
@document_request(CanvasDbConnectBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def test_db_connect():
    req = await get_request_json()
    try:
        if req["db_type"] in ["mysql", "mariadb"]:
            db = MySQLDatabase(req["database"], user=req["username"], host=req["host"], port=req["port"],
                               password=req["password"])
        elif req["db_type"] == "oceanbase":
            db = MySQLDatabase(req["database"], user=req["username"], host=req["host"], port=req["port"],
                               password=req["password"], charset="utf8mb4")
        elif req["db_type"] == 'postgres':
            db = PostgresqlDatabase(req["database"], user=req["username"], host=req["host"], port=req["port"],
                                    password=req["password"])
        elif req["db_type"] == 'mssql':
            import pyodbc
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={req['host']},{req['port']};"
                f"DATABASE={req['database']};"
                f"UID={req['username']};"
                f"PWD={req['password']};"
            )
            db = pyodbc.connect(connection_string)
            cursor = db.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        elif req["db_type"] == 'IBM DB2':
            import ibm_db
            conn_str = (
                f"DATABASE={req['database']};"
                f"HOSTNAME={req['host']};"
                f"PORT={req['port']};"
                f"PROTOCOL=TCPIP;"
                f"UID={req['username']};"
                f"PWD={req['password']};"
            )
            redacted_conn_str = (
                f"DATABASE={req['database']};"
                f"HOSTNAME={req['host']};"
                f"PORT={req['port']};"
                f"PROTOCOL=TCPIP;"
                f"UID={req['username']};"
                f"PWD=****;"
            )
            logging.info(redacted_conn_str)
            conn = ibm_db.connect(conn_str, "", "")
            stmt = ibm_db.exec_immediate(conn, "SELECT 1 FROM sysibm.sysdummy1")
            ibm_db.fetch_assoc(stmt)
            ibm_db.close(conn)
            return get_json_result(data="Database Connection Successful!")
        elif req["db_type"] == 'trino':
            def _parse_catalog_schema(db_name: str):
                if not db_name:
                    return None, None
                if "." in db_name:
                    catalog_name, schema_name = db_name.split(".", 1)
                elif "/" in db_name:
                    catalog_name, schema_name = db_name.split("/", 1)
                else:
                    catalog_name, schema_name = db_name, "default"
                return catalog_name, schema_name
            try:
                import trino
                import os
            except Exception as e:
                return server_error_response(f"Missing dependency 'trino'. Please install: pip install trino, detail: {e}")

            catalog, schema = _parse_catalog_schema(req["database"])
            if not catalog:
                return server_error_response("For Trino, 'database' must be 'catalog.schema' or at least 'catalog'.")

            http_scheme = "https" if os.environ.get("TRINO_USE_TLS", "0") == "1" else "http"

            auth = None
            if http_scheme == "https" and req.get("password"):
                auth = trino.BasicAuthentication(req.get("username") or "ragflow", req["password"])

            conn = trino.dbapi.connect(
                host=req["host"],
                port=int(req["port"] or 8080),
                user=req["username"] or "ragflow",
                catalog=catalog,
                schema=schema or "default",
                http_scheme=http_scheme,
                auth=auth
            )
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
            cur.close()
            conn.close()
            return get_json_result(data="Database Connection Successful!")
        else:
            return server_error_response("Unsupported database type.")
        if req["db_type"] != 'mssql':
            db.connect()
        db.close()

        return get_json_result(data="Database Connection Successful!")
    except Exception as e:
        return server_error_response(e)


#api get list version dsl of canvas
@manager.route('/getlistversion/<canvas_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvases"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def getlistversion(canvas_id):
    try:
        versions =sorted([c.to_dict() for c in UserCanvasVersionService.list_by_canvas_id(canvas_id)], key=lambda x: x["update_time"]*-1)
        return get_json_result(data=versions)
    except Exception as e:
        return get_data_error_result(message=f"Error getting history files: {e}")


#api get version dsl of canvas
@manager.route('/getversion/<version_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvases"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def getversion( version_id):
    try:
        e, version = UserCanvasVersionService.get_by_id(version_id)
        if version:
            return get_json_result(data=version.to_dict())
    except Exception as e:
        return get_json_result(data=f"Error getting history file: {e}")


@manager.route('/list', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvases"])
@document_querystring(CanvasListQueryDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def list_canvas():
    keywords = request.args.get("keywords", "")
    page_number = int(request.args.get("page", 0))
    items_per_page = int(request.args.get("page_size", 0))
    orderby = request.args.get("orderby", "create_time")
    canvas_category = request.args.get("canvas_category")
    if request.args.get("desc", "true").lower() == "false":
        desc = False
    else:
        desc = True
    owner_ids = [id for id in request.args.get("owner_ids", "").strip().split(",") if id]
    if not owner_ids:
        tenants = TenantService.get_joined_tenants_by_user_id(current_user.id)
        tenants = [m["tenant_id"] for m in tenants]
        tenants.append(current_user.id)
        canvas, total = UserCanvasService.get_by_tenant_ids(
            tenants, current_user.id, page_number,
            items_per_page, orderby, desc, keywords, canvas_category)
    else:
        tenants = owner_ids
        canvas, total = UserCanvasService.get_by_tenant_ids(
            tenants, current_user.id, 0,
            0, orderby, desc, keywords, canvas_category)
    return get_json_result(data={"canvas": canvas, "total": total})


@manager.route('/setting', methods=['POST'])  # noqa: F821
@validate_request("id", "title", "permission")
@login_required
@tag(["Canvases"])
@document_request(CanvasSettingBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def setting():
    req = await get_request_json()
    req["user_id"] = current_user.id

    if not UserCanvasService.accessible(req["id"], current_user.id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)

    e,flow = UserCanvasService.get_by_id(req["id"])
    if not e:
        return get_data_error_result(message="canvas not found.")
    flow = flow.to_dict()
    flow["title"] = req["title"]

    for key in ["description", "permission", "avatar"]:
        if value := req.get(key):
            flow[key] = value

    num= UserCanvasService.update_by_id(req["id"], flow)
    return get_json_result(data=num)


@manager.route('/trace', methods=['GET'])  # noqa: F821
@tag(["Canvas Utilities"])
@document_querystring(CanvasTraceQueryDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def trace():
    cvs_id = request.args.get("canvas_id")
    msg_id = request.args.get("message_id")
    try:
        binary = REDIS_CONN.get(f"{cvs_id}-{msg_id}-logs")
        if not binary:
            return get_json_result(data={})

        return get_json_result(data=json.loads(binary.encode("utf-8")))
    except Exception as e:
        logging.exception(e)


@manager.route('/<canvas_id>/sessions', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvas Sessions"])
@document_querystring(CanvasSessionsQueryDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def sessions(canvas_id):
    tenant_id = current_user.id
    if not UserCanvasService.accessible(canvas_id, tenant_id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)

    user_id = request.args.get("user_id")
    page_number = int(request.args.get("page", 1))
    items_per_page = int(request.args.get("page_size", 30))
    keywords = request.args.get("keywords")
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")
    orderby = request.args.get("orderby", "update_time")
    exp_user_id = request.args.get("exp_user_id")
    if request.args.get("desc") == "False" or request.args.get("desc") == "false":
        desc = False
    else:
        desc = True

    if exp_user_id:
        sess = API4ConversationService.get_names(canvas_id, exp_user_id)
        return get_json_result(data={"total": len(sess), "sessions": sess})
    
    # dsl defaults to True in all cases except for False and false
    include_dsl = request.args.get("dsl") != "False" and request.args.get("dsl") != "false"
    total, sess = API4ConversationService.get_list(canvas_id, tenant_id, page_number, items_per_page, orderby, desc,
                                             None, user_id, include_dsl, keywords, from_date, to_date, exp_user_id=exp_user_id)
    try:
        return get_json_result(data={"total": total, "sessions": sess})
    except Exception as e:
        return server_error_response(e)


@manager.route('/<canvas_id>/sessions', methods=['PUT'])  # noqa: F821
@login_required
@tag(["Canvas Sessions"])
@document_request(CanvasSessionCreateBodyDoc)
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
async def set_session(canvas_id):
    req = await get_request_json()
    tenant_id = current_user.id
    e, cvs = UserCanvasService.get_by_id(canvas_id)
    assert e, "Agent not found."
    if not isinstance(cvs.dsl, str):
        cvs.dsl = json.dumps(cvs.dsl, ensure_ascii=False)
    session_id=get_uuid()
    canvas = Canvas(cvs.dsl, tenant_id, canvas_id, canvas_id=cvs.id)
    canvas.reset()
    # Get the version title for this canvas (using latest, not necessarily released)
    version_title = UserCanvasVersionService.get_latest_version_title(cvs.id, release_mode=False)
    conv = {
        "id": session_id,
        "name": req.get("name", ""),
        "dialog_id": cvs.id,
        "user_id": tenant_id,
        "exp_user_id": tenant_id,
        "message": [],
        "source": "agent",
        "dsl": cvs.dsl,
        "reference": [],
        "version_title": version_title
    }
    API4ConversationService.save(**conv)
    return get_json_result(data=conv)


@manager.route('/<canvas_id>/sessions/<session_id>', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvas Sessions"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def get_session(canvas_id, session_id):
    tenant_id = current_user.id
    if not UserCanvasService.accessible(canvas_id, tenant_id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)
    _, conv = API4ConversationService.get_by_id(session_id)
    return get_json_result(data=conv.to_dict())


@manager.route('/<canvas_id>/sessions/<session_id>', methods=['DELETE'])  # noqa: F821
@login_required
@tag(["Canvas Sessions"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def del_session(canvas_id, session_id):
    tenant_id = current_user.id
    if not UserCanvasService.accessible(canvas_id, tenant_id):
        return get_json_result(
            data=False, message='Only owner of canvas authorized for this operation.',
            code=RetCode.OPERATING_ERROR)
    return get_json_result(data=API4ConversationService.delete_by_id(session_id))


@manager.route('/prompts', methods=['GET'])  # noqa: F821
@login_required
@tag(["Canvas Utilities"])
@document_response(SuccessResponse)
@document_response(ErrorResponse, 400)
def prompts():
    from rag.prompts.generator import ANALYZE_TASK_SYSTEM, ANALYZE_TASK_USER, NEXT_STEP, REFLECT, CITATION_PROMPT_TEMPLATE

    return get_json_result(data={
        "task_analysis": ANALYZE_TASK_SYSTEM +"\n\n"+ ANALYZE_TASK_USER,
        "plan_generation": NEXT_STEP,
        "reflection": REFLECT,
        #"context_summary": SUMMARY4MEMORY,
        #"context_ranking": RANK_MEMORY,
        "citation_guidelines": CITATION_PROMPT_TEMPLATE
    })


@manager.route('/download', methods=['GET'])  # noqa: F821
@tag(["Canvas Utilities"])
@document_querystring(CanvasDownloadQueryDoc)
@document_response(ErrorResponse, 400)
async def download():
    id = request.args.get("id")
    created_by = request.args.get("created_by")
    blob = FileService.get_blob(created_by, id)
    return await make_response(blob)
