from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai.models import ModelRequestParameters


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(
        function_tools=[], output_mode='text', allow_text_output=True, output_tools=[], output_object=None
    )
    assert TypeAdapter(ModelRequestParameters).dump_python(params) == snapshot(
        {
            'function_tools': [],
            'builtin_tools': [],
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [],
            'allow_text_output': True,
            'allow_image_output': False,
        }
    )
