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


from quart import Response
from pydantic import BaseModel
from quart_schema import document_response, tag

from api.apps import login_required
from api.utils.api_utils import get_json_result
from agent.plugin import GlobalPluginManager


class ErrorResponse(BaseModel):
    code: int
    message: str
    data: dict | None = None


class GenericSuccessResponse(BaseModel):
    code: int = 0
    data: list[dict] | None = None
    message: str = "success"


@manager.route('/llm_tools', methods=['GET'])  # noqa: F821
@login_required
@tag(["Plugins"])
@document_response(GenericSuccessResponse)
@document_response(ErrorResponse, 400)
def llm_tools() -> Response:
    """List LLM tool plugins available to the current user."""
    tools = GlobalPluginManager.get_llm_tools()
    tools_metadata = [t.get_metadata() for t in tools]

    return get_json_result(data=tools_metadata)
