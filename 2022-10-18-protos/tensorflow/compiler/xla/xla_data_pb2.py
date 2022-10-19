# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/xla/xla_data.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&tensorflow/compiler/xla/xla_data.proto\x12\x03xla\"\xb7\x01\n\rPaddingConfig\x12=\n\ndimensions\x18\x01 \x03(\x0b\x32).xla.PaddingConfig.PaddingConfigDimension\x1ag\n\x16PaddingConfigDimension\x12\x18\n\x10\x65\x64ge_padding_low\x18\x01 \x01(\x03\x12\x19\n\x11\x65\x64ge_padding_high\x18\x02 \x01(\x03\x12\x18\n\x10interior_padding\x18\x03 \x01(\x03\"\x1f\n\tTileProto\x12\x12\n\ndimensions\x18\x01 \x03(\x03\"\xa2\x02\n\x0bLayoutProto\x12*\n\x0f\x64im_level_types\x18\t \x03(\x0e\x32\x11.xla.DimLevelType\x12\x16\n\x0eminor_to_major\x18\x01 \x03(\x03\x12\x1d\n\x05tiles\x18\x06 \x03(\x0b\x32\x0e.xla.TileProto\x12\x14\n\x0cmemory_space\x18\x08 \x01(\x03\x12\'\n\x0ephysical_shape\x18\n \x01(\x0b\x32\x0f.xla.ShapeProtoJ\x04\x08\x02\x10\x03J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x07\x10\x08R\x11padded_dimensionsR\rpadding_valueR\x06\x66ormatR\x13max_sparse_elementsR\x14\x65lement_size_in_bits\"\xbd\x01\n\nShapeProto\x12(\n\x0c\x65lement_type\x18\x02 \x01(\x0e\x32\x12.xla.PrimitiveType\x12\x12\n\ndimensions\x18\x03 \x03(\x03\x12%\n\x0ctuple_shapes\x18\x04 \x03(\x0b\x32\x0f.xla.ShapeProto\x12 \n\x06layout\x18\x05 \x01(\x0b\x32\x10.xla.LayoutProto\x12\x1c\n\x14is_dynamic_dimension\x18\x06 \x03(\x08J\x04\x08\x01\x10\x02R\x04rank\"r\n\x11ProgramShapeProto\x12#\n\nparameters\x18\x01 \x03(\x0b\x32\x0f.xla.ShapeProto\x12\x1f\n\x06result\x18\x02 \x01(\x0b\x32\x0f.xla.ShapeProto\x12\x17\n\x0fparameter_names\x18\x03 \x03(\t\"D\n\x10\x43omputationStats\x12\x12\n\nflop_count\x18\x01 \x01(\x01\x12\x1c\n\x14transcendental_count\x18\x02 \x01(\x01\"\xff\x03\n\nOpMetadata\x12\x0f\n\x07op_type\x18\x01 \x01(\t\x12\x0f\n\x07op_name\x18\x02 \x01(\t\x12\x13\n\x0bsource_file\x18\x03 \x01(\t\x12\x13\n\x0bsource_line\x18\x04 \x01(\x05\x12*\n\x0cprofile_type\x18\x05 \x03(\x0e\x32\x10.xla.ProfileTypeB\x02\x18\x01\x12\x18\n\x10\x63reation_pass_id\x18\x06 \x01(\x03\x12 \n\x18logical_creation_pass_id\x18\x07 \x01(\x03\x12\'\n\x1fsize_of_generated_code_in_bytes\x18\x08 \x01(\x03\x12+\n#size_of_memory_working_set_in_bytes\x18\t \x01(\x03\x12\x31\n\x0cprofile_info\x18\n \x01(\x0b\x32\x1b.xla.OpMetadata.ProfileInfo\x1a\xad\x01\n\x0bProfileInfo\x12&\n\x0cprofile_type\x18\x01 \x03(\x0e\x32\x10.xla.ProfileType\x12\x18\n\x10relative_speedup\x18\x02 \x01(\x01\x12*\n\x0eprofile_source\x18\x03 \x01(\x0e\x32\x12.xla.ProfileSource\x12\x30\n\x11\x63ompilation_event\x18\x04 \x01(\x0e\x32\x15.xla.CompilationEventJ\x04\x08\x0b\x10\x0c\"\xe3\x01\n\x10\x45xecutionProfile\x12\x1d\n\x15\x63ompilation_cache_hit\x18\x01 \x01(\x08\x12\x17\n\x0f\x63ompile_time_ms\x18\x02 \x01(\x03\x12\x1b\n\x13\x63ompute_cycle_count\x18\x03 \x01(\x03\x12\x17\n\x0f\x63ompute_time_ns\x18\x04 \x01(\x03\x12$\n\x1c\x63ompute_and_transfer_time_ns\x18\x05 \x01(\x03\x12 \n\x18\x65xecutable_size_in_bytes\x18\x06 \x01(\x03\x12\x19\n\x11profile_cache_hit\x18\x07 \x01(\x08\"!\n\x0f\x45xecutionHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\"\"\n\x10GlobalDataHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\"4\n\x0c\x44\x65viceHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\x12\x14\n\x0c\x64\x65vice_count\x18\x02 \x01(\x03\"\xb4\x01\n\rChannelHandle\x12\x0e\n\x06handle\x18\x01 \x01(\x03\x12,\n\x04type\x18\x02 \x01(\x0e\x32\x1e.xla.ChannelHandle.ChannelType\"e\n\x0b\x43hannelType\x12\x18\n\x14\x43HANNEL_TYPE_INVALID\x10\x00\x12\x14\n\x10\x44\x45VICE_TO_DEVICE\x10\x01\x12\x12\n\x0e\x44\x45VICE_TO_HOST\x10\x02\x12\x12\n\x0eHOST_TO_DEVICE\x10\x03\"\xc5\x01\n\x15\x44\x65viceAssignmentProto\x12\x15\n\rreplica_count\x18\x01 \x01(\x05\x12\x19\n\x11\x63omputation_count\x18\x02 \x01(\x05\x12I\n\x13\x63omputation_devices\x18\x03 \x03(\x0b\x32,.xla.DeviceAssignmentProto.ComputationDevice\x1a/\n\x11\x43omputationDevice\x12\x1a\n\x12replica_device_ids\x18\x01 \x03(\x05\"\xc4\x02\n\x0cLiteralProto\x12\x1e\n\x05shape\x18\x01 \x01(\x0b\x32\x0f.xla.ShapeProto\x12\r\n\x05preds\x18\x02 \x03(\x08\x12\x0b\n\x03s8s\x18\x0f \x01(\x0c\x12\x0b\n\x03u8s\x18\x03 \x01(\x0c\x12\x0c\n\x04s32s\x18\x04 \x03(\x05\x12\x0c\n\x04s64s\x18\x05 \x03(\x03\x12\x0c\n\x04u32s\x18\x06 \x03(\r\x12\x0c\n\x04u64s\x18\x07 \x03(\x04\x12\x0c\n\x04\x66\x33\x32s\x18\x08 \x03(\x02\x12\x0c\n\x04\x66\x36\x34s\x18\t \x03(\x01\x12\x0c\n\x04\x63\x36\x34s\x18\x0c \x03(\x02\x12\r\n\x05\x63\x31\x32\x38s\x18\x12 \x03(\x01\x12)\n\x0etuple_literals\x18\n \x03(\x0b\x32\x11.xla.LiteralProto\x12\x0c\n\x04\x66\x31\x36s\x18\x0b \x01(\x0c\x12\r\n\x05\x62\x66\x31\x36s\x18\r \x01(\x0c\x12\x0c\n\x04u16s\x18\x10 \x01(\x0c\x12\x0c\n\x04s16s\x18\x11 \x01(\x0c\x12\x16\n\x0esparse_indices\x18\x0e \x03(\x03\"\xa3\x01\n\x0fWindowDimension\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0e\n\x06stride\x18\x02 \x01(\x03\x12\x13\n\x0bpadding_low\x18\x03 \x01(\x03\x12\x14\n\x0cpadding_high\x18\x04 \x01(\x03\x12\x17\n\x0fwindow_dilation\x18\x05 \x01(\x03\x12\x15\n\rbase_dilation\x18\x06 \x01(\x03\x12\x17\n\x0fwindow_reversal\x18\x07 \x01(\x08\"2\n\x06Window\x12(\n\ndimensions\x18\x01 \x03(\x0b\x32\x14.xla.WindowDimension\"~\n\x16GatherDimensionNumbers\x12\x13\n\x0boffset_dims\x18\x01 \x03(\x03\x12\x1c\n\x14\x63ollapsed_slice_dims\x18\x02 \x03(\x03\x12\x17\n\x0fstart_index_map\x18\x03 \x03(\x03\x12\x18\n\x10index_vector_dim\x18\x04 \x01(\x03\"\x93\x01\n\x17ScatterDimensionNumbers\x12\x1a\n\x12update_window_dims\x18\x01 \x03(\x03\x12\x1c\n\x14inserted_window_dims\x18\x02 \x03(\x03\x12$\n\x1cscatter_dims_to_operand_dims\x18\x03 \x03(\x03\x12\x18\n\x10index_vector_dim\x18\x04 \x01(\x03\"\xd8\x02\n\x1b\x43onvolutionDimensionNumbers\x12\x1d\n\x15input_batch_dimension\x18\x07 \x01(\x03\x12\x1f\n\x17input_feature_dimension\x18\x08 \x01(\x03\x12 \n\x18input_spatial_dimensions\x18\x0b \x03(\x03\x12&\n\x1ekernel_input_feature_dimension\x18\x03 \x01(\x03\x12\'\n\x1fkernel_output_feature_dimension\x18\x04 \x01(\x03\x12!\n\x19kernel_spatial_dimensions\x18\x06 \x03(\x03\x12\x1e\n\x16output_batch_dimension\x18\t \x01(\x03\x12 \n\x18output_feature_dimension\x18\n \x01(\x03\x12!\n\x19output_spatial_dimensions\x18\x0c \x03(\x03\"\x99\x01\n\x13\x44otDimensionNumbers\x12\"\n\x1alhs_contracting_dimensions\x18\x01 \x03(\x03\x12\"\n\x1arhs_contracting_dimensions\x18\x02 \x03(\x03\x12\x1c\n\x14lhs_batch_dimensions\x18\x03 \x03(\x03\x12\x1c\n\x14rhs_batch_dimensions\x18\x04 \x03(\x03\"\xdf\x01\n\x16TriangularSolveOptions\x12\x11\n\tleft_side\x18\x01 \x01(\x08\x12\r\n\x05lower\x18\x02 \x01(\x08\x12\x15\n\runit_diagonal\x18\x03 \x01(\x08\x12:\n\x0btranspose_a\x18\x04 \x01(\x0e\x32%.xla.TriangularSolveOptions.Transpose\"P\n\tTranspose\x12\x15\n\x11TRANSPOSE_INVALID\x10\x00\x12\x10\n\x0cNO_TRANSPOSE\x10\x01\x12\r\n\tTRANSPOSE\x10\x02\x12\x0b\n\x07\x41\x44JOINT\x10\x03\" \n\x0f\x43holeskyOptions\x12\r\n\x05lower\x18\x01 \x01(\x08\"o\n\x12\x46rontendAttributes\x12-\n\x03map\x18\x01 \x03(\x0b\x32 .xla.FrontendAttributes.MapEntry\x1a*\n\x08MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x80\x03\n\nOpSharding\x12\"\n\x04type\x18\x01 \x01(\x0e\x32\x14.xla.OpSharding.Type\x12#\n\ntile_shape\x18\x02 \x01(\x0b\x32\x0f.xla.ShapeProto\x12\"\n\x1atile_assignment_dimensions\x18\x03 \x03(\x03\x12\x1f\n\x17tile_assignment_devices\x18\x04 \x03(\x03\x12(\n\x0ftuple_shardings\x18\x05 \x03(\x0b\x32\x0f.xla.OpSharding\x12\"\n\x1areplicate_on_last_tile_dim\x18\x06 \x01(\x08\x12!\n\x08metadata\x18\x07 \x03(\x0b\x32\x0f.xla.OpMetadata\x12,\n\x0elast_tile_dims\x18\x08 \x03(\x0e\x32\x14.xla.OpSharding.Type\"E\n\x04Type\x12\x0e\n\nREPLICATED\x10\x00\x12\x0b\n\x07MAXIMAL\x10\x01\x12\t\n\x05TUPLE\x10\x02\x12\t\n\x05OTHER\x10\x03\x12\n\n\x06MANUAL\x10\x04\"#\n\x0cReplicaGroup\x12\x13\n\x0breplica_ids\x18\x01 \x03(\x03\".\n\x0cSourceTarget\x12\x0e\n\x06source\x18\x01 \x01(\x03\x12\x0e\n\x06target\x18\x02 \x01(\x03\"\x90\x01\n\x0fPrecisionConfig\x12\x39\n\x11operand_precision\x18\x01 \x03(\x0e\x32\x1e.xla.PrecisionConfig.Precision\"B\n\tPrecision\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x08\n\x04HIGH\x10\x01\x12\x0b\n\x07HIGHEST\x10\x02\x12\x11\n\rPACKED_NIBBLE\x10\x03\":\n\x14ParameterReplication\x12\"\n\x1areplicated_at_leaf_buffers\x18\x01 \x03(\x08\"{\n\x16WhileLoopBackendConfig\x12\x44\n\x10known_trip_count\x18\x01 \x01(\x0b\x32*.xla.WhileLoopBackendConfig.KnownTripCount\x1a\x1b\n\x0eKnownTripCount\x12\t\n\x01n\x18\x01 \x01(\x03\"q\n\x1f\x43ustomCallOutputOperandAliasing\x12\x1a\n\x12output_shape_index\x18\x01 \x03(\x03\x12\x15\n\roperand_index\x18\x02 \x01(\x03\x12\x1b\n\x13operand_shape_index\x18\x03 \x03(\x03*\xda\x01\n\rPrimitiveType\x12\x1a\n\x16PRIMITIVE_TYPE_INVALID\x10\x00\x12\x08\n\x04PRED\x10\x01\x12\x06\n\x02S8\x10\x02\x12\x07\n\x03S16\x10\x03\x12\x07\n\x03S32\x10\x04\x12\x07\n\x03S64\x10\x05\x12\x06\n\x02U8\x10\x06\x12\x07\n\x03U16\x10\x07\x12\x07\n\x03U32\x10\x08\x12\x07\n\x03U64\x10\t\x12\x07\n\x03\x46\x31\x36\x10\n\x12\x07\n\x03\x46\x33\x32\x10\x0b\x12\x08\n\x04\x42\x46\x31\x36\x10\x10\x12\x07\n\x03\x46\x36\x34\x10\x0c\x12\x07\n\x03\x43\x36\x34\x10\x0f\x12\x08\n\x04\x43\x31\x32\x38\x10\x12\x12\t\n\x05TUPLE\x10\r\x12\x0f\n\x0bOPAQUE_TYPE\x10\x0e\x12\t\n\x05TOKEN\x10\x11*D\n\x0c\x44imLevelType\x12\r\n\tDIM_DENSE\x10\x00\x12\x12\n\x0e\x44IM_COMPRESSED\x10\x01\x12\x11\n\rDIM_SINGLETON\x10\x02*=\n\x0bProfileType\x12\x0b\n\x07INVALID\x10\x00\x12\n\n\x06WINDOW\x10\x01\x12\x08\n\x04\x46LAG\x10\x02\x12\x0b\n\x07INTEGER\x10\x03*j\n\rProfileSource\x12!\n\x1dPROFILE_SOURCE_UNKNOWN_SOURCE\x10\x00\x12\x1b\n\x17PROFILE_SOURCE_EMBEDDED\x10\x01\x12\x19\n\x15PROFILE_SOURCE_REMOTE\x10\x02*\x85\x01\n\x10\x43ompilationEvent\x12#\n\x1f\x43OMPILATION_EVENT_UNKNOWN_EVENT\x10\x00\x12\'\n#COMPILATION_EVENT_FIRST_COMPILATION\x10\x01\x12#\n\x1f\x43OMPILATION_EVENT_RECOMPILATION\x10\x02*G\n\x0bPaddingType\x12\x13\n\x0fPADDING_INVALID\x10\x00\x12\x11\n\rPADDING_VALID\x10\x01\x12\x10\n\x0cPADDING_SAME\x10\x02*1\n\x07\x46\x66tType\x12\x07\n\x03\x46\x46T\x10\x00\x12\x08\n\x04IFFT\x10\x01\x12\x08\n\x04RFFT\x10\x02\x12\t\n\x05IRFFT\x10\x03*F\n\x12RandomDistribution\x12\x0f\n\x0bRNG_INVALID\x10\x00\x12\x0f\n\x0bRNG_UNIFORM\x10\x01\x12\x0e\n\nRNG_NORMAL\x10\x02*E\n\x0fRandomAlgorithm\x12\x0f\n\x0bRNG_DEFAULT\x10\x00\x12\x11\n\rRNG_THREE_FRY\x10\x01\x12\x0e\n\nRNG_PHILOX\x10\x02\x42\x03\xf8\x01\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.xla.xla_data_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _OPMETADATA.fields_by_name['profile_type']._options = None
  _OPMETADATA.fields_by_name['profile_type']._serialized_options = b'\030\001'
  _FRONTENDATTRIBUTES_MAPENTRY._options = None
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_options = b'8\001'
  _PRIMITIVETYPE._serialized_start=4808
  _PRIMITIVETYPE._serialized_end=5026
  _DIMLEVELTYPE._serialized_start=5028
  _DIMLEVELTYPE._serialized_end=5096
  _PROFILETYPE._serialized_start=5098
  _PROFILETYPE._serialized_end=5159
  _PROFILESOURCE._serialized_start=5161
  _PROFILESOURCE._serialized_end=5267
  _COMPILATIONEVENT._serialized_start=5270
  _COMPILATIONEVENT._serialized_end=5403
  _PADDINGTYPE._serialized_start=5405
  _PADDINGTYPE._serialized_end=5476
  _FFTTYPE._serialized_start=5478
  _FFTTYPE._serialized_end=5527
  _RANDOMDISTRIBUTION._serialized_start=5529
  _RANDOMDISTRIBUTION._serialized_end=5599
  _RANDOMALGORITHM._serialized_start=5601
  _RANDOMALGORITHM._serialized_end=5670
  _PADDINGCONFIG._serialized_start=48
  _PADDINGCONFIG._serialized_end=231
  _PADDINGCONFIG_PADDINGCONFIGDIMENSION._serialized_start=128
  _PADDINGCONFIG_PADDINGCONFIGDIMENSION._serialized_end=231
  _TILEPROTO._serialized_start=233
  _TILEPROTO._serialized_end=264
  _LAYOUTPROTO._serialized_start=267
  _LAYOUTPROTO._serialized_end=557
  _SHAPEPROTO._serialized_start=560
  _SHAPEPROTO._serialized_end=749
  _PROGRAMSHAPEPROTO._serialized_start=751
  _PROGRAMSHAPEPROTO._serialized_end=865
  _COMPUTATIONSTATS._serialized_start=867
  _COMPUTATIONSTATS._serialized_end=935
  _OPMETADATA._serialized_start=938
  _OPMETADATA._serialized_end=1449
  _OPMETADATA_PROFILEINFO._serialized_start=1270
  _OPMETADATA_PROFILEINFO._serialized_end=1443
  _EXECUTIONPROFILE._serialized_start=1452
  _EXECUTIONPROFILE._serialized_end=1679
  _EXECUTIONHANDLE._serialized_start=1681
  _EXECUTIONHANDLE._serialized_end=1714
  _GLOBALDATAHANDLE._serialized_start=1716
  _GLOBALDATAHANDLE._serialized_end=1750
  _DEVICEHANDLE._serialized_start=1752
  _DEVICEHANDLE._serialized_end=1804
  _CHANNELHANDLE._serialized_start=1807
  _CHANNELHANDLE._serialized_end=1987
  _CHANNELHANDLE_CHANNELTYPE._serialized_start=1886
  _CHANNELHANDLE_CHANNELTYPE._serialized_end=1987
  _DEVICEASSIGNMENTPROTO._serialized_start=1990
  _DEVICEASSIGNMENTPROTO._serialized_end=2187
  _DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE._serialized_start=2140
  _DEVICEASSIGNMENTPROTO_COMPUTATIONDEVICE._serialized_end=2187
  _LITERALPROTO._serialized_start=2190
  _LITERALPROTO._serialized_end=2514
  _WINDOWDIMENSION._serialized_start=2517
  _WINDOWDIMENSION._serialized_end=2680
  _WINDOW._serialized_start=2682
  _WINDOW._serialized_end=2732
  _GATHERDIMENSIONNUMBERS._serialized_start=2734
  _GATHERDIMENSIONNUMBERS._serialized_end=2860
  _SCATTERDIMENSIONNUMBERS._serialized_start=2863
  _SCATTERDIMENSIONNUMBERS._serialized_end=3010
  _CONVOLUTIONDIMENSIONNUMBERS._serialized_start=3013
  _CONVOLUTIONDIMENSIONNUMBERS._serialized_end=3357
  _DOTDIMENSIONNUMBERS._serialized_start=3360
  _DOTDIMENSIONNUMBERS._serialized_end=3513
  _TRIANGULARSOLVEOPTIONS._serialized_start=3516
  _TRIANGULARSOLVEOPTIONS._serialized_end=3739
  _TRIANGULARSOLVEOPTIONS_TRANSPOSE._serialized_start=3659
  _TRIANGULARSOLVEOPTIONS_TRANSPOSE._serialized_end=3739
  _CHOLESKYOPTIONS._serialized_start=3741
  _CHOLESKYOPTIONS._serialized_end=3773
  _FRONTENDATTRIBUTES._serialized_start=3775
  _FRONTENDATTRIBUTES._serialized_end=3886
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_start=3844
  _FRONTENDATTRIBUTES_MAPENTRY._serialized_end=3886
  _OPSHARDING._serialized_start=3889
  _OPSHARDING._serialized_end=4273
  _OPSHARDING_TYPE._serialized_start=4204
  _OPSHARDING_TYPE._serialized_end=4273
  _REPLICAGROUP._serialized_start=4275
  _REPLICAGROUP._serialized_end=4310
  _SOURCETARGET._serialized_start=4312
  _SOURCETARGET._serialized_end=4358
  _PRECISIONCONFIG._serialized_start=4361
  _PRECISIONCONFIG._serialized_end=4505
  _PRECISIONCONFIG_PRECISION._serialized_start=4439
  _PRECISIONCONFIG_PRECISION._serialized_end=4505
  _PARAMETERREPLICATION._serialized_start=4507
  _PARAMETERREPLICATION._serialized_end=4565
  _WHILELOOPBACKENDCONFIG._serialized_start=4567
  _WHILELOOPBACKENDCONFIG._serialized_end=4690
  _WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT._serialized_start=4663
  _WHILELOOPBACKENDCONFIG_KNOWNTRIPCOUNT._serialized_end=4690
  _CUSTOMCALLOUTPUTOPERANDALIASING._serialized_start=4692
  _CUSTOMCALLOUTPUTOPERANDALIASING._serialized_end=4805
# @@protoc_insertion_point(module_scope)
