# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: wide_and_deep.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='wide_and_deep.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x13wide_and_deep.proto\"\x8d\x02\n\x05\x41\x64ult\x12\x0b\n\x03\x61ge\x18\x01 \x01(\x05\x12\x14\n\x0c\x63\x61pital_gain\x18\x02 \x01(\x05\x12\x14\n\x0c\x63\x61pital_loss\x18\x03 \x01(\x05\x12\x11\n\teducation\x18\x04 \x01(\t\x12\x15\n\reducation_num\x18\x05 \x01(\x05\x12\x0e\n\x06gender\x18\x06 \x01(\t\x12\x16\n\x0ehours_per_week\x18\x07 \x01(\x05\x12\x16\n\x0emarital_status\x18\x08 \x01(\t\x12\x16\n\x0enative_country\x18\t \x01(\t\x12\x12\n\noccupation\x18\n \x01(\t\x12\x0c\n\x04race\x18\x0b \x01(\t\x12\x14\n\x0crelationship\x18\x0c \x01(\t\x12\x11\n\tworkclass\x18\r \x01(\t\"H\n\x0eSingleResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x1b\n\x04\x64\x61ta\x18\x02 \x01(\x0b\x32\r.SingleResult\x12\x0b\n\x03msg\x18\x03 \x01(\t\".\n\x0cSingleResult\x12\x10\n\x08\x63\x61tegory\x18\x01 \x01(\t\x12\x0c\n\x04prob\x18\x02 \x01(\x01\x32\x37\n\x0fWideAndDeepGrpc\x12$\n\x07Predict\x12\x06.Adult\x1a\x0f.SingleResponse\"\x00\x62\x06proto3'
)




_ADULT = _descriptor.Descriptor(
  name='Adult',
  full_name='Adult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='age', full_name='Adult.age', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='capital_gain', full_name='Adult.capital_gain', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='capital_loss', full_name='Adult.capital_loss', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='education', full_name='Adult.education', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='education_num', full_name='Adult.education_num', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gender', full_name='Adult.gender', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hours_per_week', full_name='Adult.hours_per_week', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='marital_status', full_name='Adult.marital_status', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='native_country', full_name='Adult.native_country', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='occupation', full_name='Adult.occupation', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='race', full_name='Adult.race', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='relationship', full_name='Adult.relationship', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='workclass', full_name='Adult.workclass', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=293,
)


_SINGLERESPONSE = _descriptor.Descriptor(
  name='SingleResponse',
  full_name='SingleResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='SingleResponse.code', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='SingleResponse.data', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msg', full_name='SingleResponse.msg', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=295,
  serialized_end=367,
)


_SINGLERESULT = _descriptor.Descriptor(
  name='SingleResult',
  full_name='SingleResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='category', full_name='SingleResult.category', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prob', full_name='SingleResult.prob', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=369,
  serialized_end=415,
)

_SINGLERESPONSE.fields_by_name['data'].message_type = _SINGLERESULT
DESCRIPTOR.message_types_by_name['Adult'] = _ADULT
DESCRIPTOR.message_types_by_name['SingleResponse'] = _SINGLERESPONSE
DESCRIPTOR.message_types_by_name['SingleResult'] = _SINGLERESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Adult = _reflection.GeneratedProtocolMessageType('Adult', (_message.Message,), {
  'DESCRIPTOR' : _ADULT,
  '__module__' : 'wide_and_deep_pb2'
  # @@protoc_insertion_point(class_scope:Adult)
  })
_sym_db.RegisterMessage(Adult)

SingleResponse = _reflection.GeneratedProtocolMessageType('SingleResponse', (_message.Message,), {
  'DESCRIPTOR' : _SINGLERESPONSE,
  '__module__' : 'wide_and_deep_pb2'
  # @@protoc_insertion_point(class_scope:SingleResponse)
  })
_sym_db.RegisterMessage(SingleResponse)

SingleResult = _reflection.GeneratedProtocolMessageType('SingleResult', (_message.Message,), {
  'DESCRIPTOR' : _SINGLERESULT,
  '__module__' : 'wide_and_deep_pb2'
  # @@protoc_insertion_point(class_scope:SingleResult)
  })
_sym_db.RegisterMessage(SingleResult)



_WIDEANDDEEPGRPC = _descriptor.ServiceDescriptor(
  name='WideAndDeepGrpc',
  full_name='WideAndDeepGrpc',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=417,
  serialized_end=472,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='WideAndDeepGrpc.Predict',
    index=0,
    containing_service=None,
    input_type=_ADULT,
    output_type=_SINGLERESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_WIDEANDDEEPGRPC)

DESCRIPTOR.services_by_name['WideAndDeepGrpc'] = _WIDEANDDEEPGRPC

# @@protoc_insertion_point(module_scope)
