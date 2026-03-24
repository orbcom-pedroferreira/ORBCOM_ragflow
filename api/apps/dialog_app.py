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
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from quart import request
from quart_schema import document_querystring, document_request, document_response, tag
from api.db.services import duplicate_name
from api.db.services.dialog_service import DialogService
from common.constants import StatusEnum
from api.db.services.tenant_llm_service import TenantLLMService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import TenantService, UserTenantService
from api.utils.api_utils import get_data_error_result, get_json_result, get_request_json, server_error_response, validate_request
from api.utils.tenant_utils import ensure_tenant_model_id_for_params
from common.misc_utils import get_uuid
from common.constants import RetCode
from api.apps import login_required, current_user
import logging


class ErrorResponse(BaseModel):
    code: int = 0
    data: Any | None = None
    message: str


class GenericSuccessResponse(BaseModel):
    code: int = 0
    data: Any | None = None
    message: str = "success"


class DialogGetQueryDoc(BaseModel):
    dialog_id: str = Field(..., description="Dialog ID.")


class DialogSetBodyDoc(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_config: dict[str, Any] = Field(..., description="Prompt configuration.")
    dialog_id: str | None = Field(default=None, description="Dialog ID for updates.")
    name: str | None = Field(default=None, description="Dialog name.")
    description: str | None = Field(default=None, description="Dialog description.")
    icon: str | None = Field(default=None, description="Dialog icon.")
    top_n: int | None = Field(default=None, description="Top-N retrieval count.")
    top_k: int | None = Field(default=None, description="Top-K retrieval depth.")
    rerank_id: str | None = Field(default=None, description="Rerank model ID.")
    tenant_rerank_id: int | str | None = Field(default=None, description="Tenant rerank model ID.")
    similarity_threshold: float | None = Field(default=None, description="Similarity threshold.")
    vector_similarity_weight: float | None = Field(default=None, description="Vector similarity weight.")
    llm_setting: dict[str, Any] | None = Field(default=None, description="LLM settings.")
    meta_data_filter: dict[str, Any] | None = Field(default=None, description="Metadata filter configuration.")
    kb_ids: list[str] | None = Field(default=None, description="Knowledge base IDs.")
    llm_id: str | None = Field(default=None, description="LLM ID.")
    tenant_llm_id: int | str | None = Field(default=None, description="Tenant LLM ID.")


class DialogListNextQueryDoc(BaseModel):
    keywords: str | None = Field(default=None, description="Optional keyword filter.")
    page: int | None = Field(default=None, description="Page number.")
    page_size: int | None = Field(default=None, description="Items per page.")
    parser_id: str | None = Field(default=None, description="Parser ID filter.")
    orderby: str | None = Field(default=None, description="Sort field.")
    desc: str | bool | None = Field(default=None, description="Whether to sort descending.")


class DialogListNextBodyDoc(BaseModel):
    owner_ids: list[str] | None = Field(default=None, description="Owner tenant IDs.")


class DialogDeleteBodyDoc(BaseModel):
    dialog_ids: list[str] = Field(..., description="Dialog IDs to delete.")


@manager.route('/set', methods=['POST'])  # noqa: F821
@validate_request("prompt_config")
@login_required
@tag(["Dialogs"])
@document_request(DialogSetBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def set_dialog():
    req = await get_request_json()
    dialog_info = ensure_tenant_model_id_for_params(current_user.id, req)
    dialog_id = dialog_info.get("dialog_id", "")
    is_create = not dialog_id
    name = dialog_info.get("name", "New Dialog")
    if not isinstance(name, str):
        return get_data_error_result(message="Dialog name must be string.")
    if name.strip() == "":
        return get_data_error_result(message="Dialog name can't be empty.")
    if len(name.encode("utf-8")) > 255:
        return get_data_error_result(message=f"Dialog name length is {len(name)} which is larger than 255")

    name = name.strip()
    if is_create:
        # only for chat creating
        existing_names = {
            d.name.casefold()
            for d in DialogService.query(tenant_id=current_user.id, status=StatusEnum.VALID.value)
            if d.name
        }
        if name.casefold() in existing_names:
            def _name_exists(name: str, **_kwargs) -> bool:
                return name.casefold() in existing_names

            name = duplicate_name(_name_exists, name=name)

    description = dialog_info.get("description", "A helpful dialog")
    icon = dialog_info.get("icon", "")
    top_n = dialog_info.get("top_n", 6)
    top_k = dialog_info.get("top_k", 1024)
    rerank_id = dialog_info.get("rerank_id", "")
    if not rerank_id:
        dialog_info["rerank_id"] = ""
    similarity_threshold = dialog_info.get("similarity_threshold", 0.1)
    vector_similarity_weight = dialog_info.get("vector_similarity_weight", 0.3)
    llm_setting = dialog_info.get("llm_setting", {})
    meta_data_filter = dialog_info.get("meta_data_filter", {})
    prompt_config = dialog_info["prompt_config"]

    # Set default parameters for datasets with knowledge retrieval
    # All datasets with {knowledge} in system prompt need "knowledge" parameter to enable retrieval
    kb_ids = dialog_info.get("kb_ids", [])
    parameters = prompt_config.get("parameters")
    logging.debug(f"set_dialog: kb_ids={kb_ids}, parameters={parameters}, is_create={not is_create}")
    # Check if parameters is missing, None, or empty list
    if kb_ids and not parameters:
        # Check if system prompt uses {knowledge} placeholder
        if "{knowledge}" in prompt_config.get("system", ""):
            # Set default parameters for any dataset with knowledge placeholder
            prompt_config["parameters"] = [{"key": "knowledge", "optional": False}]
            logging.debug(f"Set default parameters for datasets with knowledge placeholder: {kb_ids}")

    if not is_create:
        # only for chat updating
        if not dialog_info.get("kb_ids", []) and not prompt_config.get("tavily_api_key") and "{knowledge}" in prompt_config.get("system", ""):
            return get_data_error_result(message="Please remove `{knowledge}` in system prompt since no dataset / Tavily used here.")

    for p in prompt_config.get("parameters", []):
        if p["optional"]:
            continue
        if prompt_config.get("system", "").find("{%s}" % p["key"]) < 0:
            return get_data_error_result(
                message="Parameter '{}' is not used".format(p["key"]))

    try:
        e, tenant = TenantService.get_by_id(current_user.id)
        if not e:
            return get_data_error_result(message="Tenant not found!")
        kbs = KnowledgebaseService.get_by_ids(dialog_info.get("kb_ids", []))
        embd_ids = [TenantLLMService.split_model_name_and_factory(kb.embd_id)[0] for kb in kbs]  # remove vendor suffix for comparison
        embd_count = len(set(embd_ids))
        if embd_count > 1:
            return get_data_error_result(message=f'Datasets use different embedding models: {[kb.embd_id for kb in kbs]}"')

        llm_id = dialog_info.get("llm_id", tenant.llm_id)
        tenant_llm_id = dialog_info.get("tenant_llm_id", tenant.tenant_llm_id)
        if not dialog_id:
            dia = {
                "id": get_uuid(),
                "tenant_id": current_user.id,
                "name": name,
                "kb_ids": dialog_info.get("kb_ids", []),
                "description": description,
                "llm_id": llm_id,
                "tenant_llm_id": tenant_llm_id,
                "llm_setting": llm_setting,
                "prompt_config": prompt_config,
                "meta_data_filter": meta_data_filter,
                "top_n": top_n,
                "top_k": top_k,
                "rerank_id": rerank_id,
                "tenant_rerank_id": dialog_info.get("tenant_rerank_id", 0),
                "similarity_threshold": similarity_threshold,
                "vector_similarity_weight": vector_similarity_weight,
                "icon": icon
            }
            if not DialogService.save(**dia):
                return get_data_error_result(message="Fail to new a dialog!")
            return get_json_result(data=dia)
        else:
            del dialog_info["dialog_id"]
            if "kb_names" in dialog_info:
                del dialog_info["kb_names"]
            if not DialogService.update_by_id(dialog_id, dialog_info):
                return get_data_error_result(message="Dialog not found!")
            e, dia = DialogService.get_by_id(dialog_id)
            if not e:
                return get_data_error_result(message="Fail to update a dialog!")
            dia = dia.to_dict()
            dia.update(dialog_info)
            dia["kb_ids"], dia["kb_names"] = get_kb_names(dia["kb_ids"])
            return get_json_result(data=dia)
    except Exception as e:
        return server_error_response(e)


@manager.route('/get', methods=['GET'])  # noqa: F821
@login_required
@tag(["Dialogs"])
@document_querystring(DialogGetQueryDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
def get():
    dialog_id = request.args["dialog_id"]
    try:
        e, dia = DialogService.get_by_id(dialog_id)
        if not e:
            return get_data_error_result(message="Dialog not found!")
        dia = dia.to_dict()
        dia["kb_ids"], dia["kb_names"] = get_kb_names(dia["kb_ids"])
        return get_json_result(data=dia)
    except Exception as e:
        return server_error_response(e)


def get_kb_names(kb_ids):
    ids, nms = [], []
    for kid in kb_ids:
        e, kb = KnowledgebaseService.get_by_id(kid)
        if not e or kb.status != StatusEnum.VALID.value:
            continue
        ids.append(kid)
        nms.append(kb.name)
    return ids, nms


@manager.route('/list', methods=['GET'])  # noqa: F821
@login_required
@tag(["Dialogs"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
def list_dialogs():
    try:
        conversations = DialogService.query(
            tenant_id=current_user.id,
            status=StatusEnum.VALID.value,
            reverse=True,
            order_by=DialogService.model.create_time)
        conversations = [d.to_dict() for d in conversations]
        for conversation in conversations:
            conversation["kb_ids"], conversation["kb_names"] = get_kb_names(conversation["kb_ids"])
        return get_json_result(data=conversations)
    except Exception as e:
        return server_error_response(e)


@manager.route('/next', methods=['POST'])  # noqa: F821
@login_required
@tag(["Dialogs"])
@document_querystring(DialogListNextQueryDoc)
@document_request(DialogListNextBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def list_dialogs_next():
    args = request.args
    keywords = args.get("keywords", "")
    page_number = int(args.get("page", 0))
    items_per_page = int(args.get("page_size", 0))
    parser_id = args.get("parser_id")
    orderby = args.get("orderby", "create_time")
    if args.get("desc", "true").lower() == "false":
        desc = False
    else:
        desc = True

    req = await get_request_json()
    owner_ids = req.get("owner_ids", [])
    try:
        if not owner_ids:
            # tenants = TenantService.get_joined_tenants_by_user_id(current_user.id)
            # tenants = [tenant["tenant_id"] for tenant in tenants]
            tenants = [] # keep it here
            dialogs, total = DialogService.get_by_tenant_ids(
                tenants, current_user.id, page_number,
                items_per_page, orderby, desc, keywords, parser_id)
        else:
            tenants = owner_ids
            dialogs, total = DialogService.get_by_tenant_ids(
                tenants, current_user.id, 0,
                0, orderby, desc, keywords, parser_id)
            dialogs = [dialog for dialog in dialogs if dialog["tenant_id"] in tenants]
            total = len(dialogs)
            if page_number and items_per_page:
                dialogs = dialogs[(page_number-1)*items_per_page:page_number*items_per_page]
        return get_json_result(data={"dialogs": dialogs, "total": total})
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])  # noqa: F821
@login_required
@validate_request("dialog_ids")
@tag(["Dialogs"])
@document_request(DialogDeleteBodyDoc)
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
async def rm():
    req = await get_request_json()
    dialog_list=[]
    tenants = UserTenantService.query(user_id=current_user.id)
    try:
        for id in req["dialog_ids"]:
            for tenant in tenants:
                if DialogService.query(tenant_id=tenant.tenant_id, id=id):
                    break
            else:
                return get_json_result(
                    data=False, message='Only owner of dialog authorized for this operation.',
                    code=RetCode.OPERATING_ERROR)
            dialog_list.append({"id": id,"status":StatusEnum.INVALID.value})
        DialogService.update_many_by_id(dialog_list)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
