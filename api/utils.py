import codecs
import json
from json import JSONDecodeError

from flask import request, Response
from werkzeug import http


def validate_content_type():
    if 'Content-Type' not in request.headers:
        exception_msg = "Expects Content-Type to be application/json"
        raise Exception(exception_msg)
    content_type = http.parse_options_header(request.headers['Content-Type'])[0]
    if content_type != 'application/json':
        exception_msg = "Expects Content-Type to be application/json"
        raise Exception(exception_msg)


def get_body():
    validate_content_type()
    try:
        if request.data[:3] == codecs.BOM_UTF8:
            return json.loads(request.data.decode("utf-8-sig"))
        else:
            return json.loads(request.data.decode("utf-8"))

    except JSONDecodeError:
        raise Exception("Request body is not in valid JSON format")


class RecommendResponse(Response):
    def __init__(self, body, status_code, jsonfy=True):
        if body is not None:
            if jsonfy:
                super().__init__(json.dumps(body, ensure_ascii=False), status=status_code, mimetype='application/json')
            else:
                super().__init__(body, status=status_code)
        else:
            super().__init__(json.dumps(dict()), status=status_code, mimetype='application/json')